#!/usr/bin/env bash
set -euo pipefail

JAX_PLATFORM_NAME="${JAX_PLATFORM_NAME:-gpu}" \
pixi run python IR-fitting/benchmark_ees.py
