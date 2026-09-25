# Multilevel regression modeling (Radon data)

Indoor radon varies with both building characteristics and local geology.  The
[original case study](https://mc-stan.org/learn-stan/case-studies/radon_cmdstanpy_plotnine.html)
therefore relates log radon to a house-level predictor such as measurement
floor while allowing the baseline level to vary by county.  The main modelling
choice is how much information the counties should share.

These compact examples use random values only as build fixtures; they are not
the original radon observations.  Open any comparison to inspect the exact
generated Stan program beside the Julia source that produced it.  That Stan
pane is generated during the documentation build, so readers do not have to
execute the example themselves.

The arguments supplied when a model is called (`y`, `x`, and, where needed,
`county`) become Stan data.  Names first introduced on the left of `~` become
unknowns.  StanBlocks infers scalar and vector shapes from the distribution and
indexing expressions, while support such as `lower = 0.0` becomes a Stan
declaration constraint.

## Complete pooling (`radon_cp` below)

Complete pooling assumes every county has the same intercept `alpha`.  The
single slope `beta` describes the relationship with `x`, and positive `sigma`
is the residual standard deviation.  County labels are deliberately absent:
after conditioning on `x`, every observation is treated as coming from one
population.

```@raw html
<div class="atlas-comparison" data-atlas-comparison>
```

```@eval
Main.FeatureAtlasDocs.comparison(Main.FeatureAtlasDocs.example_module(:RadonCaseStudy), raw"""
using StanBlocks
y = x = randn(10)
county = rand(1:10, 10)
radon_cp = @slic begin 
    alpha ~ normal(0, 10)
    beta ~ normal(0, 10)
    sigma ~ normal(0, 10; lower=0.)
    y ~ normal(alpha + beta * x, sigma)
end
radon_cp_posterior = radon_cp(;y,x)
""", :radon_cp_posterior)
```

```@raw html
</div>
```

## No pooling

At the other extreme, no pooling gives every county an unrelated intercept.
`n = n_counties` makes `alpha` a vector, and `alpha[county]` selects the right
entry for each observation.  Its length is inferred from the largest county
index supplied as data.  The slope and residual scale remain shared, so “no
pooling” refers specifically to the county baselines.

The model is written out in full to keep the statistical contrast visible:
the substantive change from complete pooling is the vector-valued intercept
and its indexed use in the likelihood.

```@raw html
<div class="atlas-comparison" data-atlas-comparison>
```

```@eval
Main.FeatureAtlasDocs.comparison(Main.FeatureAtlasDocs.example_module(:RadonCaseStudy), raw"""
radon_np = @slic begin 
    n_counties = max(county)
    alpha ~ normal(0, 10; n=n_counties)
    beta ~ normal(0, 10)
    sigma ~ normal(0, 10; lower=0.)
    y ~ normal(alpha[county] + beta * x, sigma)
end
radon_np_posterior = radon_np(;y,x,county)
""", :radon_np_posterior)
```

```@raw html
</div>
```

## Partial pooling

Partial pooling replaces the independent, zero-centred intercept priors with a
population distribution.  `mu_alpha` learns the overall county level and
positive `sigma_alpha` learns the between-county variation.  Each county can
still move toward its own data, but a county with little information is shrunk
toward `mu_alpha` rather than estimated in isolation.  This is the multilevel
compromise between the first two models.

```@raw html
<div class="atlas-comparison" data-atlas-comparison>
```

```@eval
Main.FeatureAtlasDocs.comparison(Main.FeatureAtlasDocs.example_module(:RadonCaseStudy), raw"""
radon_pp = @slic begin 
    n_counties = max(county)
    mu_alpha ~ normal(0, 10)
    sigma_alpha ~ normal(0, 10; lower=0)
    alpha ~ normal(mu_alpha, sigma_alpha; n=n_counties)
    beta ~ normal(0, 10)
    sigma ~ normal(0, 10; lower=0.)
    y ~ normal(alpha[county] + beta * x, sigma)
end
radon_pp_posterior = radon_pp(;y,x,county)
""", :radon_pp_posterior)
```

```@raw html
</div>
```

### Cross-validation-aware county labels

The final model has exactly the same partial-pooling likelihood, but wraps the
county index with `StanBlocks.stan.maybecv`.  In an ordinary fit this behaves
like the original data.  In StanBlocks' leave-one-group-out workflow, the
wrapper marks the county dimension as cross-validation-aware: the held-out
contribution is omitted from the fitted likelihood, and quantities depending
on that county's intercept can be regenerated from the population model in
generated quantities.  Population-level parameters are still learned from the
retained counties.

The annotation belongs at the data boundary, so the statistical model need not
be duplicated.  The generated-Stan pane makes the resulting
parameter/generated-quantities split explicit.

```@raw html
<div class="atlas-comparison" data-atlas-comparison>
```

```@eval
Main.FeatureAtlasDocs.comparison(Main.FeatureAtlasDocs.example_module(:RadonCaseStudy), raw"""
radon_pp_cv_posterior = radon_pp(;y,x,county=StanBlocks.stan.maybecv(:county, county))
""", :radon_pp_cv_posterior)
```

```@raw html
</div>
```
