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
