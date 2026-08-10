# Golf putting

This guided example follows the model sequence in [Andrew Gelman's golf putting
case study](https://mc-stan.org/learn-stan/case-studies/golf.html). The question
is not merely whether longer putts are harder, but which physical mechanisms
make them harder. We therefore move from a descriptive regression to models
of aiming error and distance control.

The small arrays below are build fixtures, not the original golf data. Each
comparison evaluates the displayed Julia source while the documentation is
built and shows the complete generated Stan program beside it. The comparison
can also be opened as the same side-by-side modal used by the feature atlas;
readers do not have to execute anything to see the Stan code.

## Logistic regression

The baseline treats the success probability as a logistic curve of distance,

```math
\operatorname{logit}(p_i) = a + b x_i, \qquad
y_i \sim \operatorname{Binomial}(n_i, p_i).
```

It is useful as a descriptive benchmark, but `a` and `b` do not say *why* a
putt misses. In the SLIC source, the first use of the fresh names `a` and `b`
on the left of `~` introduces scalar parameters. The call
`logistic(; y, n, x)` binds the remaining free names as data. StanBlocks
infers the scalar and vector shapes from those values and emits the data,
parameter, and model declarations visible in the Stan pane.

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

A successful putt must satisfy two conditions: it must start within the
angular window subtended by the hole, and its speed must leave the ball within
a tolerable stopping-distance range. Splitting those mechanisms into SLIC
submodels makes the assumptions explicit and lets later models reuse them.

### Submodels

Both definitions below are anonymous `@slic` values. Their free names become
data when supplied as keyword arguments, their local sampled variables remain
local to each use, and their final `return` value is the quantity embedded by
the parent model. A parent writes `p ~ submodel(; inputs...)`; this is SLIC
composition, not a probability distribution named `submodel`. The compiler
inlines the submodel hygienically and carries its parameters and transformed
values into the generated Stan blocks.

#### Angle submodel (`angle_submodel` below)

For a hole of radius `R`, a ball of radius `r`, and putt distance `x`, the
allowable aiming angle is `asin((R-r)/x)`. The positive parameter `sigma`
represents angular error. Assuming centered Gaussian aiming error, the returned
vector is the probability of landing inside the angular window. The
elementwise `./` and the vector-valued `Phi` call are enough for StanBlocks to
infer a vector result; no Stan declaration is written by hand.

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

The second component models speed control. `overshot` locates the target
stopping point beyond the hole, while `distance_tolerance` defines the accepted
interval around it. `sigma_distance` is constrained positive by the
`lower=0.` distribution keyword. The difference of two normal CDF values is
the probability that the stopping distance lies inside that interval.

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

This is the first mechanistic replacement for logistic regression. The parent
model obtains the vector `p` by embedding `angle_submodel`, then uses that
probability in the same binomial observation model. Compare the generated Stan
with the baseline: the regression coefficient disappears, while the angular
error parameter and its deterministic probability calculation appear.

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

The next model assumes the aiming and distance-control events are independent,
so their probabilities multiply elementwise. The two submodel calls introduce
distinct, hygienically scoped parameters (`sigma` and `sigma_distance`) and
return vectors of the same inferred length. The parent only states how those
components combine and how successes are observed.

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

## Allowing extra variation

The final step changes the observation layer rather than the putting physics.
It models the observed proportions `y ./ n` with a normal approximation whose
variance contains the usual binomial term
`p .* (1-p) ./ n` plus an extra scale `sigma_y^2`. That additional scale can
absorb variation not explained by the two physical mechanisms.

`to_vector` converts the integer count arrays to Stan vectors before division;
the dotted arithmetic then remains elementwise. StanBlocks infers
`raw_proportions`, both component probabilities, and the combined `p` as
vector-valued transformed quantities. The generated Stan pane makes the
change from a binomial likelihood to this continuous approximation explicit.

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
