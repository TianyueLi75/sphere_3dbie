# sphere_stok3dbie
A fast solver of Laplace and Stokes equations on a sphere using Boundary Integral Equations (BIE) and the layer potential operators spectra. Fast transforms to and from the scalar and vector spherical harmonics uses the [SHTns library](https://nschaeff.bitbucket.io/shtns/) in [python](https://github.com/meliao/shtns/tree/adjoint) with [jax](https://github.com/meliao/shtns/tree/jax_dev) support.

## Running the tests

Manufactured-solution tests for the Laplace/Stokes operators and the coupled suspension solvers live in `test/`. Run them with the jax + shtns_jax environment:

```bash
python -m pytest test/
```

Notes:
- `test/conftest.py` puts the repo root on `sys.path` and caps `OMP_NUM_THREADS=1` (the eager shtns evaluators segfault otherwise), so no extra setup is needed.
- Single file / single test, e.g. `python -m pytest test/test_Stk3d.py -k interior -v`.