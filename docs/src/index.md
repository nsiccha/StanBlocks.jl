# StanBlocks.jl

StanBlocks.jl is a (limited) Julia-to-Stan transpiler: write probabilistic models in Julia syntax and automatically generate compilable [Stan](https://mc-stan.org/) code.

::: warning Caveats
This transpiler has many caveats. See [Caveats](#caveats) below.
:::

## Quick Start

```julia
using StanBlocks
import StanLogDensityProblems, JSON

# Model definition (no data bound yet)
earn_height_model = @slic begin
    beta  ~ flat(;n=2)
    sigma ~ flat(;lower=0.)
    earn  ~ normal(beta[1] + beta[2] * to_vector(height), sigma)
end

# Bind data
earn_height_posterior = earn_height_model(; earn, height)

# Inspect the generated Stan code
println(stan_code(earn_height_posterior))

# Compile and instantiate (requires StanLogDensityProblems.jl + JSON.jl)
earn_height_problem = stan_instantiate(earn_height_posterior)
```

## Key Features

### Activity Analysis

StanBlocks automatically determines which Stan block each variable belongs to:

| Block | Description |
|---|---|
| `data` | Passed in from Julia (observed quantities) |
| `transformed_data` | Computed once from data (e.g. `log_y = log.(y)`) |
| `parameters` | Sampled by HMC; determine posterior dimension |
| `transformed_parameters` | Deterministic functions of parameters, evaluated each gradient step |
| `generated_quantities` | Computed per sample, do not enter the likelihood |

### Type Inference

Stan requires explicit types and shapes. StanBlocks infers them automatically from passed values and from the model structure. Shapes extracted from data are exposed as integer data in the generated Stan code.

### User-Defined Functions

The [`@deffun`](api#StanBlocks.@deffun) macro extends the Stan function library with variadic signatures and limited dynamic dispatch:

```julia
@deffun begin
    my_normal_lpdf(y, mu, sigma) = normal_lpdf(y, mu, sigma)
    my_normal_rng(mu, sigma)::real = normal_rng(mu, sigma)

    # Dispatch on function-argument type (higher-order functions)
    dispatch_lpdf(y, ::typeof(my_normal), args...) = my_normal_lpdf(y, args...)
end
```

### Constraints

Constraints are specified as keyword arguments to distributions. Many distributions (e.g. `beta`, `gamma`) infer their bounds automatically:

```julia
model = @slic (;y) begin
    sigma ~ std_normal(;lower=0.)   # half-normal (explicit lower bound)
    theta ~ beta(1., 1.)            # [0,1] bounds inferred automatically
    y ~ normal(0., sigma)
end
```

### Sub-Models and Composition

Models can be composed by referencing one model inside another:

```julia
prior = @slic begin
    mu  ~ std_normal()
    tau ~ std_normal(;lower=0.)
end

hierarchical = @slic (;y) begin
    theta ~ prior()
    y ~ normal(theta, 1.)
end
```

## Caveats

### Constant Terms in Log-Density

Stan's `~` statement [drops constant terms](https://mc-stan.org/docs/reference-manual/statements.html#log-probability-increment-vs.-distribution-statement) that do not depend on model parameters. StanBlocks does **not** drop constants—it always includes the full log-density. This means the absolute value of the log-density will generally differ from a hand-written Stan model, but the posterior geometry is identical and sampling is unaffected.

### Limited Coverage

Not all Julia constructs can be transpiled. In particular, arbitrary Julia functions (not defined via `@deffun`) cannot be transpiled automatically.

### Name Resolution

User-defined functions and sub-models currently need to be defined in `Main`.

## See Also

- [API Reference](api) – full list of exported functions and macros
- [Case Studies](https://nsiccha.github.io/StanBlocks.jl/slic/) – golf, radon, crowdsourcing, and more
- [Stan Documentation](https://mc-stan.org/docs/stan-users-guide/)
- [BridgeStan](https://github.com/roualdes/bridgestan) – Stan–Julia bridge used for compilation
