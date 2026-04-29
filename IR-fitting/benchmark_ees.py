"""Benchmark reversible solvers on short Langevin MD training rollouts.

Trains each solver for a fixed number of steps under a matched NFE budget and
reports the loss curve.  The default is intentionally short and conservative:
long Langevin trajectories can make reversible-adjoint reconstruction unstable
before the optimiser has any chance to do useful work.

Two Langevin rollout modes are available:

``sde``
    Uses Diffrax's Brownian SDE path
    dv = (F/m - gamma v) dt + sigma dW
    assembled as MultiTerm(ODETerm, ControlTerm(VirtualBrownianTree)).

``fixed_noise``
    Pre-samples Langevin kicks and treats them as a deterministic forcing term.
    This is useful for reproducible solver comparisons and debugging.

Loss (squared dipole velocity, proxy for IR autocorrelation) is accumulated
inside the augmented state and normalised by trajectory length and molecule
count. Produces: IR-fitting/benchmark_ees_loss.png by default.
"""

import os

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import pickle
import time
from collections.abc import Callable
from typing import ClassVar, cast

import dmff
import equinox as eqx
import jax
jax.config.update(
    "jax_debug_nans", os.environ.get("EES_DEBUG_NANS", "").lower() in {"1", "true", "yes"}
)
jax.config.update(
    "jax_debug_infs", os.environ.get("EES_DEBUG_INFS", "").lower() in {"1", "true", "yes"}
)
jax.config.update(
    "jax_disable_jit", os.environ.get("EES_DISABLE_JIT", "").lower() in {"1", "true", "yes"}
)
jax.config.update(
    "jax_traceback_filtering",
    "off" if os.environ.get("EES_FULL_TRACEBACK", "").lower() in {"1", "true", "yes"} else "auto",
)
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax
from equinox.internal import ω
from jax import grad, jit, random, tree_util, value_and_grad, vmap

dmff.PRECISION = "float"
dmff.update_jax_precision(dmff.PRECISION)

import lineax as lx
from diffrax import (
    AbstractERK,
    AbstractReversibleSolver,
    ConstantStepSize,
    ControlTerm,
    Euler,
    LocalLinearInterpolation,
    Midpoint,
    MultiTerm,
    ODETerm,
    ReversibleAdjoint,
    ReversibleHeun,
    SaveAt,
    VirtualBrownianTree,
    diffeqsolve,
)
from diffrax._solution import update_result
from diffrax_lowstorage import EES25
from difftraj_ees import EES_Loss_Generator
from eann import EANNForce

ω = cast(Callable, ω)

# ── Config ────────────────────────────────────────────────────────────────────


def _env_flag(name, default=False):
    value = os.environ.get(name)
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes", "on"}


def _env_int(name, default):
    value = os.environ.get(name)
    return default if value is None else int(value)


def _env_float(name, default):
    value = os.environ.get(name)
    return default if value is None else float(value)


DEBUG_FINITE = _env_flag("EES_CHECK_FINITE")


def _finite_guard(x, label):
    if not DEBUG_FINITE:
        return x
    return eqx.error_if(
        x,
        jnp.any(~jnp.isfinite(x)),
        f"{label} contains non-finite values",
    )


def _array_leaves(tree):
    return [leaf for leaf in tree_util.tree_leaves(tree) if hasattr(leaf, "shape")]


def _tree_all_finite(tree):
    leaves = _array_leaves(tree)
    if not leaves:
        return True
    checks = [jnp.all(jnp.isfinite(leaf)) for leaf in leaves]
    return bool(jax.device_get(jnp.all(jnp.stack(checks))))


def _tree_global_norm(tree):
    leaves = _array_leaves(tree)
    if not leaves:
        return 0.0
    max_abs = max(float(jnp.max(jnp.abs(leaf))) for leaf in leaves)
    if max_abs == 0.0:
        return 0.0
    scaled_total = sum(float(jnp.sum(jnp.square(leaf / max_abs))) for leaf in leaves)
    return max_abs * float(np.sqrt(scaled_total))


def _safe_clip_by_global_norm(max_norm):
    def init_fn(params):
        del params
        return optax.EmptyState()

    def update_fn(updates, state, params=None):
        del params
        leaves = _array_leaves(updates)
        if not leaves:
            return updates, state
        max_abs = jnp.max(jnp.stack([jnp.max(jnp.abs(leaf)) for leaf in leaves]))
        dtype = max_abs.dtype
        denom = jnp.maximum(max_abs, jnp.finfo(dtype).tiny)
        scaled_total = sum(jnp.sum(jnp.square(leaf / denom)) for leaf in leaves)
        scale = (jnp.asarray(max_norm, dtype=dtype) / denom) / (
            jnp.sqrt(scaled_total) + jnp.finfo(dtype).eps
        )
        scale = jnp.where(max_abs > 0, jnp.minimum(1.0, scale), 1.0)
        clipped = tree_util.tree_map(lambda leaf: leaf * scale, updates)
        return clipped, state

    return optax.GradientTransformation(init_fn, update_fn)


def _assert_finite_scalar(x, label):
    value = float(x)
    if not np.isfinite(value):
        raise FloatingPointError(f"{label} is non-finite: {value}")
    return value


def _assert_tree_finite(tree, label):
    if not _tree_all_finite(tree):
        raise FloatingPointError(f"{label} contains non-finite values")


seed = 1234
T = 298.15
rc = 0.6
TOTAL_TIME = _env_float("EES_TOTAL_TIME", 0.1)
NFE_BUDGET = _env_int(
    "EES_NFE_BUDGET", 252
)  # divisible by Euler(2), Midpoint(4), EES25/ReversibleHeun(3)
nout = 1
batch_size = _env_int("EES_BATCH_SIZE", 4)
n_train_steps = _env_int("EES_N_TRAIN_STEPS", 20)
lr = _env_float("EES_LR", 5e-4)
grad_clip = _env_float("EES_GRAD_CLIP", 1.0)
gamma_langevin = _env_float("EES_GAMMA", 1.0)
rollout_mode = os.environ.get("EES_ROLLOUT_MODE", "sde").strip().lower()
output_path = os.environ.get(
    "EES_OUTPUT", os.path.join("IR-fitting", "benchmark_ees_loss.png")
)
key = random.PRNGKey(seed)
solver_filter = {
    name.strip()
    for name in os.environ.get("EES_SOLVERS", "").split(",")
    if name.strip()
}

if rollout_mode not in {"sde", "fixed_noise"}:
    raise ValueError("EES_ROLLOUT_MODE must be either 'sde' or 'fixed_noise'.")

# ── System setup ──────────────────────────────────────────────────────────────

from openmm.app import PDBFile

pdb1 = PDBFile("IR-fitting/water64.pdb")
pos1 = jnp.array(pdb1.getPositions()._value)
natoms1 = len(pos1)
box1 = jnp.array(pdb1.topology.getPeriodicBoxVectors()._value)
atomtype = ["H", "O"]
species1 = [atomtype.index(atom.element.symbol) for atom in pdb1.topology.atoms()]
elem_indices1 = jnp.array(species1)
cov_map1 = jnp.array(np.loadtxt("IR-fitting/cov_map")[:192, :192], dtype=jnp.int32)
m_O = 15.99943
m_H = 1.007947
mass = jnp.tile(jnp.array([m_O, m_H, m_H]), natoms1 // 3)
print(f"Loaded water64.pdb: {natoms1} atoms")

# ── EANN model and params ─────────────────────────────────────────────────────

n_elem = 2
eann_force1 = EANNForce(n_elem, elem_indices1, n_gto=12, rc=6, sizes=(64, 64))


def _load_legacy_params(path):
    # Pickles from older JAX store ShapedArray with a named_shape field that
    # current JAX rejects; monkey-patch __init__ for the duration of the load.
    import jax._src.core as _jax_core

    _orig_init = _jax_core.ShapedArray.__init__

    def _compat_init(self, shape, dtype, weak_type=False, named_shape=None, **kwargs):
        _orig_init(self, shape, dtype, weak_type, **kwargs)

    _jax_core.ShapedArray.__init__ = _compat_init
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    finally:
        _jax_core.ShapedArray.__init__ = _orig_init


_param_candidates = ["IR-fitting/params_eann4.pickle", "params.pickle"]
params = None
for _pfile in _param_candidates:
    if os.path.exists(_pfile):
        params = {"energy": _load_legacy_params(_pfile)}
        print(f"Loaded EANN params from {_pfile}")
        break
if params is None:
    params = {"energy": eann_force1.params}
    print("WARNING: no params pickle found — using random-init EANN params.")
    print("  Random-init gives ~zero forces → constant loss → zero gradients.")
    print("  Copy params_eann4.pickle here for a meaningful gradient benchmark.")


@jit
def efunc1(pos, box, pairs, params):
    energy = jnp.array(
        eann_force1.get_energy(pos * 10, box * 10, pairs, params["energy"])
    )
    return _finite_guard(energy, "efunc1.energy")


# ── Observable ────────────────────────────────────────────────────────────────


def f_nout_simple(state):
    pos = state["pos"]
    n_mol = pos.shape[-2] // 3
    mol_pos = pos.reshape(pos.shape[0], n_mol, 3, 3)
    weights = jnp.array([1.0, -0.5, -0.5])
    dipole = jnp.einsum("bmad,a->bmd", mol_pos, weights)
    return jnp.sum(dipole, axis=1)  # (batch, 3)


# ── UReversible wrapper (local copy; installed Diffrax doesn't init `solver`) ─


class _PatchedUReversible(AbstractReversibleSolver):
    solver: AbstractERK
    coupling_parameter: float
    interpolation_cls: ClassVar[Callable[..., LocalLinearInterpolation]] = (
        LocalLinearInterpolation
    )

    @property
    def term_structure(self):
        return self.solver.term_structure

    @property
    def term_compatible_contr_kwargs(self):
        return self.solver.term_compatible_contr_kwargs

    @property
    def root_finder(self):
        return self.solver.root_finder

    @property
    def root_find_max_steps(self):
        return self.solver.root_find_max_steps

    def __init__(self, solver: AbstractERK, coupling_parameter: float = 0.999):
        if hasattr(solver, "disable_fsal"):
            solver = eqx.tree_at(lambda s: s.disable_fsal, solver, True)
        self.solver = solver
        self.coupling_parameter = coupling_parameter

    def order(self, terms):
        return self.solver.order(terms)

    def strong_order(self, terms):
        return self.solver.strong_order(terms)

    def init(self, terms, t0, t1, y0, args):
        del terms, t0, t1, args
        return y0

    def step(self, terms, t0, t1, y0, args, solver_state, made_jump):
        del made_jump
        z0 = solver_state
        step_z0, _, _, _, result1 = self.solver.step(
            terms, t0, t1, z0, args, None, True
        )
        y1 = (self.coupling_parameter * (ω(y0) - ω(z0)) + ω(step_z0)).ω
        step_y1, y_error, _, _, result2 = self.solver.step(
            terms, t1, t0, y1, args, None, True
        )
        z1 = (ω(y1) + ω(z0) - ω(step_y1)).ω
        dense_info = dict(y0=y0, y1=y1)
        return y1, y_error, dense_info, z1, update_result(result1, result2)

    def backward_step(
        self, terms, t0, t1, y1, args, ts_state, solver_state, made_jump
    ):
        del made_jump, ts_state
        z1 = solver_state
        step_y1, _, _, _, result1 = self.solver.step(
            terms, t1, t0, y1, args, None, True
        )
        z0 = (ω(z1) - ω(y1) + ω(step_y1)).ω
        step_z0, _, _, _, result2 = self.solver.step(
            terms, t0, t1, z0, args, None, True
        )
        y0 = ((1 / self.coupling_parameter) * (ω(y1) - ω(step_z0)) + ω(z0)).ω
        dense_info = dict(y0=y0, y1=y1)
        return y0, dense_info, z0, update_result(result1, result2)

    def func(self, terms, t0, y0, args):
        return self.solver.func(terms, t0, y0, args)


# ── NFE-matched discretization ────────────────────────────────────────────────

SOLVER_NFE_PER_STEP = {
    "Euler": 2,
    "Midpoint": 4,
    "ReversibleHeun": 3,
    "EES25": 3,
}


def _solver_dt_nsteps(solver_name):
    nfe_per_step = SOLVER_NFE_PER_STEP[solver_name]
    if NFE_BUDGET % nfe_per_step != 0:
        raise ValueError(
            f"NFE budget {NFE_BUDGET} is not divisible by {solver_name} cost {nfe_per_step}."
        )
    nsteps = NFE_BUDGET // nfe_per_step
    return TOTAL_TIME / nsteps, nsteps


# ── Shared generator helpers (provides potential and mass) ────────────────────

print("\nBuilding EES25 generator...")
gen_ees = EES_Loss_Generator(
    # Only `mass`, `potential`, and `regularize_pos` are used below, so the
    # benchmark does not need an EES25-compatible NFE budget just to build this.
    f_nout_simple,
    box1,
    pos1,
    mass,
    dt=TOTAL_TIME,
    nsteps=1,
    nout=nout,
    cov_map=cov_map1,
    rc=rc,
    efunc=efunc1,
)

# ── Initial conditions ────────────────────────────────────────────────────────

KB = 1.380649e-23
NA = 6.0221408e23
kT = KB * T * NA

key, subkey = random.split(key)
mass_col = jnp.tile(mass.reshape([len(mass), 1]), (1, 3))
vel_init = (
    random.normal(subkey, shape=(batch_size, natoms1, 3))
    * jnp.sqrt(kT / mass_col * 1e3)
    / 1e3
)
pos_init = jnp.tile(pos1[None], (batch_size, 1, 1))
key, subkey = random.split(key)
pos_init = pos_init + random.normal(subkey, shape=pos_init.shape) * 0.001

state_init = {"pos": pos_init, "vel": vel_init}


# ── Langevin SDE ──────────────────────────────────────────────────────────────
# Units (GROMACS): nm, ps, g/mol.
# Fluctuation-dissipation: sigma = sqrt(2 gamma kT / m)
#   kT in J/mol, m in g/mol → sigma [nm/ps/sqrt(ps)] = sqrt(2 gamma kT·1e-3 / m)


def make_langevin_fwd_bwd(
    solver,
    dt,
    nsteps,
    gamma=1.0,
    noise_key=None,
    mode="sde",
):
    if noise_key is None:
        noise_key = random.PRNGKey(99)
    if mode not in {"sde", "fixed_noise"}:
        raise ValueError("mode must be either 'sde' or 'fixed_noise'.")

    mass_2d = gen_ees.mass  # (natoms, 3)
    potential = gen_ees.potential
    sigma = jnp.sqrt(2.0 * gamma * kT * 1e-3 / mass_2d)

    t_end = nsteps * dt
    n_mol = natoms1 // 3
    loss_normalizer = jnp.array(max(t_end * n_mol * 3.0, 1e-12))

    def dipole_vel(vel):
        n_mol = vel.shape[0] // 3
        mol_vel = vel.reshape(n_mol, 3, 3)
        weights = jnp.array([1.0, -0.5, -0.5])
        return jnp.sum(jnp.einsum("mad,a->md", mol_vel, weights), axis=0)

    def drift_field(t, y, args):
        params = args
        pos, vel, loss_acc = y
        pos = _finite_guard(pos, "drift.pos")
        vel = _finite_guard(vel, "drift.vel")
        loss_acc = _finite_guard(loss_acc, "drift.loss_acc")
        force = -grad(potential, argnums=0)(pos, params)
        force = _finite_guard(force, "drift.force")
        dv = dipole_vel(vel)
        dv = _finite_guard(dv, "drift.dipole_vel")
        acc = force / mass_2d - gamma * vel
        acc = _finite_guard(acc, "drift.acc")
        loss_rate = jnp.sum(dv**2)
        loss_rate = _finite_guard(loss_rate, "drift.loss_rate")
        return (vel, acc, loss_rate)

    def diffusion_field(t, y, args):
        pos, vel, loss_acc = y
        del t, args, loss_acc
        return lx.DiagonalLinearOperator((jnp.zeros_like(pos), sigma, jnp.array(0.0)))

    def fixed_noise_field(t, y, args):
        params, noise_acc = args
        pos, vel, loss_acc = y
        pos = _finite_guard(pos, "fixed_noise.pos")
        vel = _finite_guard(vel, "fixed_noise.vel")
        loss_acc = _finite_guard(loss_acc, "fixed_noise.loss_acc")
        force = -grad(potential, argnums=0)(pos, params)
        force = _finite_guard(force, "fixed_noise.force")
        dv = dipole_vel(vel)
        dv = _finite_guard(dv, "fixed_noise.dipole_vel")
        # The forcing is piecewise constant over the solver grid. This is a
        # deterministic reparameterisation of the random kicks, not an SDE term.
        i = jnp.minimum(jnp.floor(t / dt).astype(jnp.int32), nsteps - 1)
        acc = force / mass_2d - gamma * vel + noise_acc[i]
        acc = _finite_guard(acc, "fixed_noise.acc")
        loss_rate = jnp.sum(dv**2)
        loss_rate = _finite_guard(loss_rate, "fixed_noise.loss_rate")
        return (vel, acc, loss_rate)

    def solve_sde_one(pos0, vel0, p, bm_key):
        bm = VirtualBrownianTree(
            t0=0.0,
            t1=t_end,
            tol=dt / 2,
            shape=(
                jax.ShapeDtypeStruct(pos0.shape, pos0.dtype),
                jax.ShapeDtypeStruct(vel0.shape, vel0.dtype),
                jax.ShapeDtypeStruct((), vel0.dtype),
            ),
            key=bm_key,
        )
        term = MultiTerm(ODETerm(drift_field), ControlTerm(diffusion_field, bm))
        sol = diffeqsolve(
            term,
            solver,
            t0=0.0,
            t1=t_end,
            dt0=dt,
            y0=(pos0, vel0, jnp.array(0.0)),
            args=p,
            saveat=SaveAt(t1=True),
            stepsize_controller=ConstantStepSize(),
            adjoint=ReversibleAdjoint(),
            max_steps=nsteps + 16,
        )
        final_pos = _finite_guard(sol.ys[0][0], "solve_one.final_pos")
        final_vel = _finite_guard(sol.ys[1][0], "solve_one.final_vel")
        final_loss = _finite_guard(sol.ys[2][0], "solve_one.final_loss")
        del final_pos, final_vel
        return final_loss / loss_normalizer

    def solve_fixed_noise_one(pos0, vel0, p, noise_acc):
        sol = diffeqsolve(
            ODETerm(fixed_noise_field),
            solver,
            t0=0.0,
            t1=t_end,
            dt0=dt,
            y0=(pos0, vel0, jnp.array(0.0)),
            args=(p, noise_acc),
            saveat=SaveAt(t1=True),
            stepsize_controller=ConstantStepSize(),
            adjoint=ReversibleAdjoint(),
            max_steps=nsteps + 16,
        )
        final_pos = _finite_guard(sol.ys[0][0], "solve_fixed_noise_one.final_pos")
        final_vel = _finite_guard(sol.ys[1][0], "solve_fixed_noise_one.final_vel")
        final_loss = _finite_guard(sol.ys[2][0], "solve_fixed_noise_one.final_loss")
        del final_pos, final_vel
        return final_loss / loss_normalizer

    def fwd_bwd(state_init, params):
        pos_reg = vmap(gen_ees.regularize_pos)(state_init["pos"])
        pos_reg = _finite_guard(pos_reg, "fwd_bwd.pos_reg")
        pos_all = jnp.concatenate([pos_reg, pos_reg])
        vel_all = jnp.concatenate([-state_init["vel"], state_init["vel"]])
        pos_all = _finite_guard(pos_all, "fwd_bwd.pos_all")
        vel_all = _finite_guard(vel_all, "fwd_bwd.vel_all")
        keys = random.split(noise_key, pos_all.shape[0])

        def solve_for_params(p):
            if mode == "sde":
                loss_accs = vmap(lambda p0, v0, k: solve_sde_one(p0, v0, p, k))(
                    pos_all, vel_all, keys
                )
            else:
                kicks = random.normal(
                    noise_key,
                    shape=(pos_all.shape[0], nsteps, natoms1, 3),
                    dtype=pos_all.dtype,
                )
                noise_acc = sigma[None, None, :, :] * kicks / jnp.sqrt(dt)
                loss_accs = vmap(
                    lambda p0, v0, n0: solve_fixed_noise_one(p0, v0, p, n0)
                )(pos_all, vel_all, noise_acc)
            loss_accs = _finite_guard(loss_accs, "solve_for_params.loss_accs")
            loss = jnp.mean(loss_accs)
            return _finite_guard(loss, "solve_for_params.loss")

        return value_and_grad(solve_for_params)(params)

    return fwd_bwd


# ── Solver registry ───────────────────────────────────────────────────────────

SOLVER_SPECS = [
    ("Euler", lambda: _PatchedUReversible(Euler())),
    ("Midpoint", lambda: _PatchedUReversible(Midpoint())),
    ("ReversibleHeun", ReversibleHeun),
    ("EES25", EES25),
]

if solver_filter:
    SOLVER_SPECS = [(name, factory) for name, factory in SOLVER_SPECS if name in solver_filter]
    if not SOLVER_SPECS:
        raise ValueError(
            f"EES_SOLVERS={sorted(solver_filter)} did not match any known solver."
        )

SOLVERS = {}
for solver_name, solver_factory in SOLVER_SPECS:
    solver_dt, solver_nsteps = _solver_dt_nsteps(solver_name)
    SOLVERS[solver_name] = {
        "fwd_bwd": make_langevin_fwd_bwd(
            solver_factory(),
            dt=solver_dt,
            nsteps=solver_nsteps,
            gamma=gamma_langevin,
            mode=rollout_mode,
        ),
        "dt": solver_dt,
        "nsteps": solver_nsteps,
        "nfe_per_step": SOLVER_NFE_PER_STEP[solver_name],
    }

# ── Training ──────────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("BENCHMARKING (Langevin NVT)")
print("=" * 60)
if any(
    [
        _env_flag("EES_DEBUG_NANS"),
        _env_flag("EES_DEBUG_INFS"),
        _env_flag("EES_DISABLE_JIT"),
        _env_flag("EES_FULL_TRACEBACK"),
        DEBUG_FINITE,
        bool(solver_filter),
        rollout_mode != "sde",
        grad_clip != 1.0,
    ]
):
    print("Diagnostic flags:")
    if _env_flag("EES_DEBUG_NANS"):
        print("  EES_DEBUG_NANS=1")
    if _env_flag("EES_DEBUG_INFS"):
        print("  EES_DEBUG_INFS=1")
    if DEBUG_FINITE:
        print("  EES_CHECK_FINITE=1")
    if _env_flag("EES_DISABLE_JIT"):
        print("  EES_DISABLE_JIT=1")
    if _env_flag("EES_FULL_TRACEBACK"):
        print("  EES_FULL_TRACEBACK=1")
    if solver_filter:
        print(f"  EES_SOLVERS={','.join(name for name, _ in SOLVER_SPECS)}")
    if rollout_mode != "sde":
        print(f"  EES_ROLLOUT_MODE={rollout_mode}")
    if grad_clip != 1.0:
        print(f"  EES_GRAD_CLIP={grad_clip}")
print(
    f"System: {natoms1} atoms, t_end={TOTAL_TIME} ps, "
    f"NFE budget={NFE_BUDGET}, batch_size={batch_size}, n_train_steps={n_train_steps}"
)
print(
    f"T={T} K, gamma={gamma_langevin} ps⁻¹, lr={lr}, "
    f"grad_clip={grad_clip}, rollout={rollout_mode}"
)
print("Per-solver discretization:")
COL = 18
for name, cfg in SOLVERS.items():
    print(
        f"  {name:<{COL}}  dt={cfg['dt']:.12f} ps"
        f"  nsteps={cfg['nsteps']}"
        f"  nfe/step={cfg['nfe_per_step']}"
    )

print(f"\n--- Training ({n_train_steps} steps) ---")
all_losses = {}
train_times = {}

for name, cfg in SOLVERS.items():
    fwd_bwd = cfg["fwd_bwd"]
    p = tree_util.tree_map(lambda x: x.copy(), params)
    transforms = []
    if grad_clip > 0:
        transforms.append(_safe_clip_by_global_norm(grad_clip))
    transforms.append(optax.adam(lr))
    opt = optax.chain(*transforms)
    opt_state = opt.init(p)
    losses = []
    step_times = []
    for step in range(n_train_steps):
        t0 = time.time()
        try:
            loss, g = fwd_bwd(state_init, p)
            jax.block_until_ready((loss, g))
        except Exception:
            print(f"  [{name:<{COL}}] failed at step {step}")
            raise
        step_times.append(time.time() - t0)
        loss_value = _assert_finite_scalar(loss, f"{name} loss at step {step}")
        _assert_tree_finite(g, f"{name} gradient at step {step}")
        grad_norm = _tree_global_norm(g)
        updates, opt_state = opt.update(g, opt_state, p)
        jax.block_until_ready((updates, opt_state))
        _assert_tree_finite(updates, f"{name} optimiser update at step {step}")
        p = optax.apply_updates(p, updates)
        jax.block_until_ready(p)
        _assert_tree_finite(p, f"{name} parameters after step {step}")
        losses.append(loss_value)
        tag = " (warmup)" if step == 0 else ""
        print(
            f"  [{name:<{COL}}] step {step:3d}  loss={loss_value:.6f}"
            f"  |grad|={grad_norm:.3e}  {step_times[-1]:.2f}s{tag}"
        )
    all_losses[name] = losses
    train_times[name] = sum(step_times)

# ── Plot ──────────────────────────────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(7, 5))
colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
for i, name in enumerate(SOLVERS):
    ax.plot(all_losses[name], label=name, linewidth=2, color=colors[i % len(colors)])
ax.set_xlabel("Training step")
ax.set_ylabel("Normalised loss")
ax.set_title(f"Loss during training (Langevin NVT, {rollout_mode})")
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
output_dir = os.path.dirname(output_path)
if output_dir:
    os.makedirs(output_dir, exist_ok=True)
plt.savefig(output_path, dpi=150)
print(f"\nPlot saved to {output_path}")
plt.close()

# ── Summary ───────────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
for name, cfg in SOLVERS.items():
    print(
        f"  {name:<{COL}}  final_loss={all_losses[name][-1]:.6f}"
        f"  dt={cfg['dt']:.6g}"
        f"  nsteps={cfg['nsteps']}"
        f"  train={train_times[name]:.2f}s"
    )
