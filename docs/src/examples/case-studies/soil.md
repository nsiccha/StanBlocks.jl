# Soil carbon modeling

Qualitatively reproduces [Bob Carpenter's soil carbon modeling case study](https://mc-stan.org/learn-stan/case-studies/soil-knit.html).

```@raw html
<div class="atlas-comparison" data-atlas-comparison>
```

```@eval
Main.FeatureAtlasDocs.comparison(Main.FeatureAtlasDocs.example_module(:SoilCaseStudy), raw"""
using StanBlocks
data = (;
    eCO2mean=[1.],    
    eCO2_hat=[1.],
    eCO2sd=[1.]
)
soil_measurement_posterior = @slic data begin
    k1 ~ std_normal(;lower=0)
    k2 ~ std_normal(;lower=0)
    alpha1 ~ std_normal(;lower=0)
    alpha2 ~ std_normal(;lower=0)
    gamma ~ beta(10, 1)
    sigma ~ cauchy(0, 1; lower=0.)
    \"\"\"C_hat = integrate_ode(two_pool_feedback, ...)
    eCO2_hat = ...\"\"\"
    eCO2mean ~ normal(eCO2_hat, sigma)
end
""", :soil_measurement_posterior)
```

```@raw html
</div>
```

```@raw html
<div class="atlas-comparison" data-atlas-comparison>
```

```@eval
Main.FeatureAtlasDocs.comparison(Main.FeatureAtlasDocs.example_module(:SoilCaseStudy), raw"""
soil_latent_posterior = @slic data begin
    k1 ~ std_normal(;lower=0)
    k2 ~ std_normal(;lower=0)
    alpha1 ~ std_normal(;lower=0)
    alpha2 ~ std_normal(;lower=0)
    gamma ~ beta(10, 1)
    sigma ~ cauchy(0, 1; lower=0.)
    \"\"\"C_hat = integrate_ode(two_pool_feedback, ...)
    eCO2_hat = ...
    Would be nice to be able to reuse the previous model...\"\"\"
    eCO2 ~ normal(eCO2_hat, sigma)
    eCO2mean ~ normal(eCO2, eCO2sd)
end
""", :soil_latent_posterior)
```

```@raw html
</div>
```
