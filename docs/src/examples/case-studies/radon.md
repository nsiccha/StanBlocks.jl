# Multilevel regression modeling (Radon data)

Qualitatively reproduces [Mitzi Morris' radon case study](https://mc-stan.org/learn-stan/case-studies/radon_cmdstanpy_plotnine.html).

## Complete pooling (`radon_cp` below)

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
    # \"\"\"Would be nice to be able to reuse the previous model...
    # Maybe `y ~ radon_cp(;alpha)` or `y ~ radon_cp(;alpha ~ normal(0, 10; n=n_counties))` , but the syntax of `radon_cp` would have to reflect the special role of `y` somehow.\"\"\"
    y ~ normal(alpha[county] + beta * x, sigma)
end
radon_np_posterior = radon_np(;y,x,county)
""", :radon_np_posterior)
```

```@raw html
</div>
```

## Partial pooling

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

### Cross validation

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
