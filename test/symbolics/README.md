These optional tests exercise `jacobian(f)`, the ODE Jacobian UDF that the
Symbolics.jl extension (`StanBlocksSymbolicsExt`) derives. Set up their
environment from the repository root:

```sh
julia --startup-file=no test/symbolics/setup.jl
julia --startup-file=no --project=test/symbolics test/symbolics/runtests.jl
```

Ordinary `Pkg.test()` excludes the `:symbolics` test items.

The items that compile and run the generated programs need the Stan toolchain
whose solvers take a Jacobian UDF: `nsiccha/stanc3` and `nsiccha/math`, branch
`dde-solver`. Point `STANBLOCKS_ODE_JACOBIAN_BRIDGESTAN` at a BridgeStan 2.9.0
tree whose `bin/stanc` is built from that stanc3 and whose
`stan/lib/stan_math` is that Stan Math. Without it, those items are reported
as skipped.

The runner accepts the same `--name`, `--file`, and other selectors as
`test/runtests.jl`.
