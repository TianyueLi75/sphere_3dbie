"""Pytest fixtures/setup shared by the biop + suspension test modules.

Two things must happen before numpy/shtns/jax are imported by any test module:
  1. OMP_NUM_THREADS is capped -- the eager shtns spectral evaluators (SHqst_to_point,
     SH_to_point, the *_cplx transforms) segfault inside the MLIR lowering when OpenMP
     spins up more than a single thread. This is set here (conftest is imported before
     the test modules) rather than relying on the caller's environment.
  2. The repository root is put on sys.path so `from sphere import ...`, `from biop import
     Lap3d/Stk3d` and `import suspension` resolve when pytest is run from anywhere.

Run with the jax+shtns_jax CPU environment, e.g.
    OMP_NUM_THREADS=1 /mnt/home/tli10/jax_venv312_cpu/bin/python -m pytest test/
"""

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")

import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
