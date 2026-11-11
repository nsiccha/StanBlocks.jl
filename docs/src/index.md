# StanBlocks.jl

StanBlocks turns probabilistic models written in Julia syntax into Stan programs that run through BridgeStan.

```sh
julia -e 'using Pkg; Pkg.add(url="https://github.com/nsiccha/StanBlocks.jl", rev="devibe")'
```

```@example start
using StanBlocks
model = @slic (; y = [0.2, -0.1, 0.4]) begin
    mu ~ normal(0, 1)
    y ~ normal(mu, 1)
end
d = stan_descriptor(model)
for o in d.outputs
    println(o.name, " => ", o.generative)
end
```

The bound `y` values become observations; `mu` remains a parameter. StanBlocks
places the prior and likelihood in Stan's model block and generates predictive
replicates and pointwise log likelihoods for the observations.
Use `stan_code(model)` to inspect the complete generated program, as shown in
the [model guide](guides/models.md#Inspection).

## Models

[Model definitions](guides/models.md) explains binding, constraints and inspection.
[Composition](guides/composition.md) reuses a model within another model.
[Execution](guides/execution.md) compiles the program and evaluates its density.
The install command is a shell command; the documentation environment already
has the checked-out package installed, so the build executes the model above.

## Examples

[Golf putting](examples/golf.md) estimates a physical error scale and checks
predictions against observed success rates. [Partial pooling](examples/pooling.md)
shows how measurement uncertainty controls shrinkage.
[Epidemic dynamics](examples/epidemic.md) distinguishes prevalence from incidence.

## Reference

[Model APIs](reference/models.md) covers definitions and functions.
[Execution APIs](reference/execution.md) covers compilation, descriptors and errors.
