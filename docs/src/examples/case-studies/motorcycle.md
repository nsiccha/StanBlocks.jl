# Motorcycle data

This example follows the HSGP model sequence in [Aki Vehtari's motorcycle case
study](https://users.aalto.fi/~ave/casestudies/Motorcycle/motorcycle.html#GP_model_with_Hilbert_basis_functions).
The response is head acceleration after impact. Its mean changes nonlinearly
over time, and the size of the residual fluctuations changes as the impact is
absorbed, so a straight line with constant noise is a poor description.

The small random arrays below are build fixtures rather than the original
measurements. Every comparison evaluates the exact displayed Julia source at
documentation-build time and places the complete generated Stan program beside
it, with the same expandable side-by-side modal as the feature atlas.

## HSGP building block (`hsgp` below)

A Hilbert-space Gaussian-process (HSGP) approximation replaces a dense
Gaussian-process covariance matrix with a finite basis expansion. That makes a
smooth latent function a weighted matrix-vector product while preserving two
interpretable hyperparameters: a length scale and a marginal amplitude.

The source first rescales `x` to a fixed interval and constructs a sine basis
matrix `X` with 20 columns. `x_scale` controls how quickly the function can
vary, `y_scale` controls its marginal standard deviation, and
`unit_weight ~ std_normal(; n=n_functions)` introduces one standard-normal
coefficient per basis function. The deterministic `scale` vector turns those
coefficients into the GP approximation returned by the submodel.

Several StanBlocks features are doing work here:

- `@slic` traces the Julia linear algebra and elementwise expressions rather
  than requiring separate Stan declarations.
- The `n=n_functions` sampling keyword determines the length of
  `unit_weight`; the product `X * (scale .* unit_weight)` therefore has its
  observation-length result inferred automatically.
- `hsgp(; x)` binds the free input `x` as data. The returned vector can then be
  embedded in another `@slic` model with `lhs ~ hsgp(; x)`.
- Positive support is encoded through the chosen distributions
  (`uniform(0, 2)` and `lognormal(0, 1)`), and the Stan pane shows where the
  resulting parameters and transformed quantities are emitted.

```@raw html
<div class="atlas-comparison" data-atlas-comparison>
```

```@eval
Main.FeatureAtlasDocs.comparison(Main.FeatureAtlasDocs.example_module(:MotorcycleCaseStudy), raw"""
using StanBlocks, Markdown
x = randn(10)
obs = randn(10)

hsgp = @slic begin 
    "Transforms to [-2, 2]"
    xi = 2 * (x - min(x)) / (max(x) - min(x)) - 1.
    L = 1.5
    n_functions = 20
    X = sin(pi/(2L) * (xi+L) * range(1,n_functions)')/sqrt(L)
    "The GP lengthscale"
    x_scale ~ uniform(0, 2)
    "The GP marginal standard deviation"
    y_scale ~ lognormal(0, 1)
    "The scales for the basis functions weights"
    scale = y_scale * sqrt(sqrt(2pi) * x_scale) * exp(-0.25*(x_scale*pi/2L)^2 * range(1,n_functions)^2)
    "The basis functions weights"
    unit_weight ~ std_normal(;n=n_functions)
    "The final GP values"
    return (X * (scale .* unit_weight))
end
hsgp_posterior = hsgp(;x)
""", :hsgp_posterior)
```

```@raw html
</div>
```

## Homoskedastic model

The first observation model uses one HSGP for the mean acceleration,

```math
\mu(x) = \alpha + f(x), \qquad
y_i \sim \operatorname{Normal}(\mu(x_i), \sigma).
```

`y_intercept` sets the global level, `dy ~ hsgp(; x)` embeds the smooth
deviation, and the scalar `sigma` is shared by every observation. This is the
homoskedastic assumption: the mean may be highly nonlinear, but residual
spread is constant. Binding `homo(; x, obs)` supplies both free names as data;
the HSGP's local parameters are inlined into the parent without exposing them
as manual arguments.

```@raw html
<div class="atlas-comparison" data-atlas-comparison>
```

```@eval
Main.FeatureAtlasDocs.comparison(Main.FeatureAtlasDocs.example_module(:MotorcycleCaseStudy), raw"""
homo = @slic begin 
    y_intercept ~ std_normal()
    dy ~ hsgp(;x)
    sigma ~ lognormal(-2, 1)
    obs ~ normal(y_intercept + dy, sigma)
end
homo_posterior = homo(;x, obs)
""", :homo_posterior)
```

```@raw html
</div>
```

## Heteroskedastic model

The residual amplitude visibly changes over time, so the next model gives the
log standard deviation its own smooth function:

```math
\mu(x)=\alpha_\mu+f_\mu(x), \qquad
\log \sigma(x)=\alpha_\sigma+f_\sigma(x).
```

Calling `hsgp` twice creates two independent, hygienically renamed sets of GP
hyperparameters and basis weights: one for the mean and one for the log scale.
`exp(log_sigma_intercept + dlog_sigma)` maps the second process to positive
standard deviations. Relative to the homoskedastic model, the likelihood is
still normal and the mean component is unchanged; only the scalar `sigma` is
replaced by an observation-length vector.

```@raw html
<div class="atlas-comparison" data-atlas-comparison>
```

```@eval
Main.FeatureAtlasDocs.comparison(Main.FeatureAtlasDocs.example_module(:MotorcycleCaseStudy), raw"""
hetero = @slic begin
    y_intercept ~ std_normal()
    dy ~ hsgp(;x)
    log_sigma_intercept ~ std_normal()
    dlog_sigma ~ hsgp(;x)
    obs ~ normal(y_intercept + dy, exp(log_sigma_intercept + dlog_sigma))
end
hetero_posterior = hetero(;x,obs)
""", :hetero_posterior)
```

```@raw html
</div>
```

::: details Alternative heteroskedastic model using subsubmodels

The following formulation has the same statistical structure but factors the
repeated “intercept plus HSGP” pattern into another submodel. It demonstrates
nested SLIC composition rather than adding a new assumption.

### Submodel with submodel (`intercept_hsgp` below)

`intercept_hsgp` samples an intercept, embeds `hsgp`, and returns their sum.
The nested call keeps `x` as a keyword-bound data dependency and keeps all
fresh parameters local to this use. StanBlocks follows the composition through
both levels when it generates the final Stan program.

```@raw html
<div class="atlas-comparison" data-atlas-comparison>
```

```@eval
Main.FeatureAtlasDocs.comparison(Main.FeatureAtlasDocs.example_module(:MotorcycleCaseStudy), raw"""
intercept_hsgp = @slic begin 
    intercept ~ std_normal()
    "Submodel uses `hsgp` as a submodel"
    d ~ hsgp(;x)
    return intercept + d
end  
intercept_hsgp_posterior = intercept_hsgp(;x)
""", :intercept_hsgp_posterior)
```

```@raw html
</div>
```

### Final model

The outer model calls `intercept_hsgp` twice. The first returned vector is the
mean `y`; the second is `log_sigma`, exponentiated in the likelihood. Because
each submodel use is hygienic, the two processes do not accidentally share an
intercept, length scale, amplitude, or basis weights. Compare its Stan pane
with `hetero`: the model is statistically equivalent, but the Julia source
expresses the reusable structure directly.

```@raw html
<div class="atlas-comparison" data-atlas-comparison>
```

```@eval
Main.FeatureAtlasDocs.comparison(Main.FeatureAtlasDocs.example_module(:MotorcycleCaseStudy), raw"""
hetero2 = @slic begin 
    y ~ intercept_hsgp(;x)
    log_sigma ~ intercept_hsgp(;x)
    obs ~ normal(y, exp(log_sigma))
end
hetero2_posterior = hetero2(;x,obs)
""", :hetero2_posterior)
```

```@raw html
</div>
```
:::
