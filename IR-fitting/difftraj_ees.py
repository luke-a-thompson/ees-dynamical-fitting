import jax.numpy as jnp
from diffrax import (
    ConstantStepSize,
    ODETerm,
    ReversibleAdjoint,
    SaveAt,
    diffeqsolve,
)
from diffrax_lowstorage import EES25
from jax import grad, jit, tree_util, value_and_grad, vjp, vmap
from nblist import NeighborList


class EES_Loss_Generator:
    """MD trajectory generator using EES25 + ReversibleAdjoint.

    Drop-in geometry/term container comparable to Loss_Generator (leapfrog).
    EES25 from diffrax_lowstorage is an AbstractReversibleSolver — no UReversible
    wrapper needed.
    """

    def __init__(
        self,
        f_nout,
        box,
        pos0,
        mass,
        dt,
        nsteps,
        nout,
        cov_map,
        rc,
        efunc,
    ):
        self.f_nout = f_nout
        self.box = box
        mass = jnp.tile(mass.reshape([len(mass), 1]), (1, 3))
        self.mass = mass
        self.dt = dt
        self.nsteps = nsteps
        self.nout = nout

        nbl = NeighborList(box, rc, cov_map)
        nbl.allocate(pos0)

        @jit
        def potential(pos, params):
            nblist = nbl.nblist.update(pos)
            pairs = nblist.idx.T
            nbond = cov_map[pairs[:, 0], pairs[:, 1]]
            pairs_b = jnp.concatenate([pairs, nbond[:, None]], axis=1)
            return efunc(pos, box, pairs_b, params)

        bonds = []
        for i in range(len(cov_map)):
            bonds.append(
                jnp.concatenate([jnp.array([i]), jnp.where(cov_map[i] > 0)[0]])
            )

        @jit
        def regularize_pos(pos):
            cpos = jnp.stack([jnp.sum(pos[bond], axis=0) / len(bond) for bond in bonds])
            box_inv = jnp.linalg.inv(box)
            spos = cpos.dot(box_inv)
            spos -= jnp.floor(spos)
            shift = spos.dot(box) - cpos
            return pos + shift

        self.regularize_pos = regularize_pos
        self.potential = potential

        def vector_field(t, y, args):
            pos, vel = y
            force = -grad(potential, argnums=0)(pos, args)
            return (vel, force / mass)

        self.term = ODETerm(vector_field)
        self.solver = EES25()

        n_saves = nsteps // nout + 1
        self.save_ts = jnp.linspace(0.0, nsteps * dt, n_saves)
        self.n_saves = n_saves

    def _solve_single(self, pos0, vel0, params):
        sol = diffeqsolve(
            self.term,
            self.solver,
            t0=0.0,
            t1=self.nsteps * self.dt,
            dt0=self.dt,
            y0=(pos0, vel0),
            args=params,
            saveat=SaveAt(ts=self.save_ts),
            stepsize_controller=ConstantStepSize(),
            adjoint=ReversibleAdjoint(),
            max_steps=self.nsteps + 16,
        )
        return sol.ys  # ((n_saves, natoms, 3), (n_saves, natoms, 3))

    def solve_batch(self, state, params):
        """Solve for a batch of initial conditions.

        Returns (pos_traj, vel_traj) each with shape (batch, n_saves, natoms, 3).
        """
        return vmap(lambda p, v: self._solve_single(p, v, params))(
            state["pos"], state["vel"]
        )

    def ode_fwd(self, state, params):
        """Forward trajectory with observable extraction.

        Same interface as Loss_Generator.ode_fwd.
        """
        pos_traj, vel_traj = self.solve_batch(state, params)

        traj = {}
        traj["time"] = self.save_ts
        obs = []
        for t in range(self.n_saves):
            state_t = {"pos": pos_traj[:, t], "vel": vel_traj[:, t]}
            obs.append(self.f_nout(state_t))
        traj["state"] = jnp.stack(obs)  # (n_saves, batch, ...)

        final_state = {"pos": pos_traj[:, -1], "vel": vel_traj[:, -1]}
        return final_state, traj


def nve_reweight_ees(
    state_init,
    params,
    f_nout,
    box,
    mass,
    dt,
    nsteps,
    nout,
    cov_map,
    rc,
    efunc,
    L,
    batch_size,
    solver_cls=EES25,
):
    """NVE reweight using EES + ReversibleAdjoint.

    Drop-in replacement for nve_reweight in IR.py. Runs bidirectional
    trajectories (forward + time-reversed) and accumulates gradients in
    batch_size chunks to control memory.
    """
    gen = EES_Loss_Generator(
        f_nout, box, state_init["pos"][0], mass, dt, nsteps, nout, cov_map, rc, efunc
    )
    # Override solver if caller passes a different EES variant.
    gen.solver = solver_cls()

    pos_reg = vmap(gen.regularize_pos)(state_init["pos"])
    state = {
        "pos": jnp.concatenate([pos_reg, pos_reg]),
        "vel": jnp.concatenate([-state_init["vel"], state_init["vel"]]),
    }

    n_chunks = len(state["vel"]) // batch_size
    chunks = [
        {k: state[k][i * batch_size : (i + 1) * batch_size] for k in state}
        for i in range(n_chunks)
    ]

    # Forward pass — store VJP handles and trajectories for each chunk.
    all_obs = []
    residuals = []  # (vjp_fn, pos_traj, vel_traj) per chunk

    for chunk in chunks:

        def solve(p, _chunk=chunk):
            return gen.solve_batch(_chunk, p)

        (pos_traj, vel_traj), vjp_fn = vjp(solve, params)
        obs = jnp.stack(
            [
                f_nout({"pos": pos_traj[:, t], "vel": vel_traj[:, t]})
                for t in range(gen.n_saves)
            ]
        )
        all_obs.append(obs)
        residuals.append((vjp_fn, pos_traj, vel_traj))

    traj = jnp.concatenate(all_obs, axis=1)  # (n_saves, total_batch, ...)

    (err, (wavenum, lineshape_ref, lineshape)), grad_traj = value_and_grad(
        L, has_aux=True
    )(traj)

    # Backward pass — VJP through f_nout then diffeqsolve for each chunk.
    grad_chunks = jnp.array_split(grad_traj, n_chunks, axis=1)
    gradient = tree_util.tree_map(jnp.zeros_like, params)

    for (vjp_fn, pos_traj, vel_traj), grad_obs in zip(residuals, grad_chunks):
        grad_pos = jnp.stack(
            [
                vjp(f_nout, {"pos": pos_traj[:, t], "vel": vel_traj[:, t]})[1](
                    grad_obs[t]
                )[0]["pos"]
                for t in range(gen.n_saves)
            ],
            axis=1,
        )  # (chunk_batch, n_saves, natoms, 3)
        (grad_p,) = vjp_fn((grad_pos, jnp.zeros_like(vel_traj)))
        gradient = tree_util.tree_map(lambda a, b: a + b, gradient, grad_p)

    return err, gradient, wavenum, lineshape_ref, lineshape
