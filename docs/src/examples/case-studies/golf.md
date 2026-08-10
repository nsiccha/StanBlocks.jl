# Golf putting 

Qualitatively reproduces [Andrew Gelman's golf putting case study](https://mc-stan.org/learn-stan/case-studies/golf.html).

## Logistic regression
```@raw html
<div class="atlas-comparison" data-atlas-comparison>
```

```@eval
Main.FeatureAtlasDocs.comparison(Main.FeatureAtlasDocs.example_module(:GolfCaseStudy), raw"""
using StanBlocks
x = distance_tolerance = overshot = randn(10)
y = fill(1, 10)
n = fill(2, 10)
R = r = 1.

logistic = @slic begin 
    "Maybe alternatively `a, b .~ flat()` or a, b ~ flat(;n=2)?"
    a ~ flat()
    b ~ flat() 
    y ~ binomial_logit(n, a + b * x)
end
logistic_posterior = logistic(;y,n,x)
""", :logistic_posterior)
```

```@raw html
</div>
```

## Modelling based on first principles

### Submodels

#### Angle submodel (`angle_submodel` below)
```@eval
Main.FeatureAtlasDocs.source(Main.FeatureAtlasDocs.example_module(:GolfCaseStudy), raw"""
angle_submodel = @slic begin 
    threshold_angle = asin((R - r) ./ x) 
    sigma ~ flat(;lower=0.)
    sigma_degrees = sigma * 180 / pi
    return 2 * Phi(threshold_angle / sigma) - 1
end
""")
```

#### Distance submodel (`distance_submodel` below)
```@eval
Main.FeatureAtlasDocs.source(Main.FeatureAtlasDocs.example_module(:GolfCaseStudy), raw"""
distance_submodel = @slic begin 
    sigma_distance ~ std_normal(;lower=0.)
    return Phi(
        (distance_tolerance - overshot) ./ ((x + overshot) * sigma_distance)
    ) - Phi(
        (-overshot)./ ((x + overshot) * sigma_distance)
    )
end
""")
```
### Angle model
```@raw html
<div class="atlas-comparison" data-atlas-comparison>
```

```@eval
Main.FeatureAtlasDocs.comparison(Main.FeatureAtlasDocs.example_module(:GolfCaseStudy), raw"""
angle = @slic begin 
    p ~ angle_submodel(;R,r,x)
    y ~ binomial(n, p)
end
angle_posterior = angle(;R,r,x,y,n)
""", :angle_posterior)
```

```@raw html
</div>
```

### Angle + distance model

```@raw html
<div class="atlas-comparison" data-atlas-comparison>
```

```@eval
Main.FeatureAtlasDocs.comparison(Main.FeatureAtlasDocs.example_module(:GolfCaseStudy), raw"""
second_principles = @slic begin 
    p_angle ~ angle_submodel(;R,r,x)
    p_distance ~ distance_submodel(;distance_tolerance, overshot, x)
    p = p_angle .* p_distance
    y ~ binomial(n, p)
end
second_principles_posterior = second_principles(;R,r,x,distance_tolerance, overshot,y,n,)
""", :second_principles_posterior)
```

```@raw html
</div>
```

## Adding a fudge factor

```@raw html
<div class="atlas-comparison" data-atlas-comparison>
```

```@eval
Main.FeatureAtlasDocs.comparison(Main.FeatureAtlasDocs.example_module(:GolfCaseStudy), raw"""
third_principles = @slic begin 
    raw_proportions = to_vector(y) ./ to_vector(n)

    p_angle ~ angle_submodel(;R,r,x)
    p_distance ~ distance_submodel(;distance_tolerance, overshot, x)
    sigma_y ~ std_normal(;lower=0.)

    p = p_angle .* p_distance
    raw_proportions ~ normal(p, sqrt(p .* (1 - p) ./ to_vector(n) + sigma_y ^ 2))
end
third_principles_posterior = third_principles(;R,r,x,distance_tolerance,overshot,y,n)
""", :third_principles_posterior)
```

```@raw html
</div>
```
