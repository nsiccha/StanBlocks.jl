These optional integration tests exercise the ReactiveKernels extension.
Set up their environment from the repository root:

```sh
julia --startup-file=no test/reactivekernels/setup.jl
julia --startup-file=no --project=test/reactivekernels test/reactivekernels/runtests.jl
```

Set `RK_JAX_PYTHON` to a Python executable with JAX, NumPy and SciPy installed to
also run the emitted modules and check their values and gradients. Without
it, those checks are reported as skipped. The environment pins published
ReactiveKernels and OutputSignatures commits, and develops this StanBlocks
checkout. Ordinary `Pkg.test()` excludes the `:reactivekernels` test items.

The integration runner accepts the same `--name`, `--file`, and other
selectors as `test/runtests.jl`.

With `RK_JAX_PYTHON` set, check downstream precompilation and fresh-process
source dispatch separately:

```sh
julia --startup-file=no test/reactivekernels/precompile.jl
```

This precompiles the synthetic fixture in a temporary environment using the
same dependency pins, then checks JAX values and gradients from a new Julia
process.
