# Building up a model of golf putting

This example follows the model-building sequence from the Stan case study
[Model building and expansion for golf
putting](https://mc-stan.org/learn-stan/case-studies/golf.html). The data record
the distance `x`, number of attempts `n`, and number of successful putts
`y` at each distance. Rather than presenting five unrelated programs, the
StanBlocks implementation starts with a small model and reuses it as the
physical assumptions become richer.

The first two models use the shorter `dataset1`. The angle-and-distance
models use `dataset12`, the row-wise concatenation of both source datasets,
because the longer-distance observations are where the additional mechanism
matters.

## The five-model progression

| Display label | Statistical model and what changes |
|:--|:--|
| **Logistic regression** | A descriptive baseline: the success log-odds are `a + b * x`. The flat priors make `a` and `b` free coefficients, but the model does not encode the geometry of a ball and cup. |
| **Angle model** | Replaces the linear predictor with a geometric tolerance angle, `asin((R-r)/x)`. A putt succeeds when angular error falls inside that interval, with normally distributed error scale `sigma`. `2Phi(threshold/sigma)-1` is the corresponding success probability. `sigma_degrees` is a generated quantity for interpretation. |
| **Angle and distance** | Keeps the angle calculation and adds an independent distance-success term. The two positive components of `sigma` control angular and distance error. In this variant, overshoot and distance tolerance are supplied constants. The final probability is `p_angle .* p_distance`. |
| **Angle, distance, and residual variation** | Keeps the two-mechanism success probability but uses a normal approximation for the observed count. `sigma_y` adds extra-binomial variation to the count-scale standard deviation, allowing more dispersion than the binomial sampling term alone. |
| **Estimated overshoot and tolerance** | Returns to the binomial angle-and-distance likelihood, but promotes `overshot` and `distance_tolerance` from fixed constants to positive parameters with weak normal priors. |

The labels in the generated-code selector are therefore descriptions of the
increment added at each stage. In the Julia source, the longer historical
identifiers (`golf_angle_distance_2`,
`golf_angle_distance_3_with_resids`, and
`golf_angle_distance_4`) preserve the numbering used while the family was
developed.

## What StanBlocks is doing

Each `@slic` block is a reusable model fragment. Names such as `x`,
`n`, `y`, `r`, and `R` remain inputs until the fragment is
instantiated with keyword arguments. Splatting `dataset1...` or
`dataset12...` binds the data by name; the same mechanism supplies physical
constants and the fixed overshoot assumptions.

Several StanBlocks features make the progression concise:

- `a ~ flat()` and `b ~ flat()` declare unconstrained parameters
  without adding a prior density. Keyword constraints such as
  `lower=0` become Stan parameter constraints.
- Ordinary Julia scalar, vector, and element-wise expressions are traced into
  Stan. Deterministic values are placed in the earliest legal Stan block, while
  `sigma_degrees` is emitted after sampling as a generated quantity.
- `Base.merge` derives a model by replacing or adding statements in a
  quoted fragment. The angle calculation is written once and reused by all
  three descendants; only the assumptions that differ appear in each merge.
- The final named tuple is a family of independently instantiated models. The
  build evaluates this exact displayed source, calls `stan_code` on every
  member, and places each complete Stan program behind its descriptive label.

## Generated Stan models

Use the **StanBlocks** and **Generated Stan** tabs to switch views. **Compare
side by side** opens the same pair in a wide modal; the generated pane keeps
all five labelled programs together.

```@raw html
<div class="atlas-comparison" data-atlas-comparison data-stan-label="Generated Stan models">
```

```@eval
Main.FeatureAtlasDocs.comparisons(@__MODULE__, raw"""
using StanBlocks

"Downloads and preprocesses the golf datasets"
golf_data(url) = begin 
    x, n, y = eachcol(mapreduce(row->(parse.(Float64, split(row))), hcat, filter(startswith(r"[0-9]"), readlines(download(url))))')
    (;x, n=Int.(n), y=Int.(y))
end

# Fetch the first dataset
dataset1 = golf_data("https://gist.githubusercontent.com/nsiccha/553d6ce6a784142e87fbfd7aaf4c5e99/raw/668b86ddd84b45b707e594dd313f717bef1a553d/dataset1.txt")

# Fetch the second dataset
dataset2 = golf_data("https://gist.githubusercontent.com/nsiccha/13fd14e1cc5e1520aa688c1f4df2f9a7/raw/726764870b46c72911e89668eef48a22a6c13035/dataset2.txt")

# Combine the two datasets
dataset12 = map(vcat, dataset1, dataset2)


golf_logistic = @slic begin 
    a ~ flat()
    b ~ flat()
    y ~ binomial_logit(n, a + b * x)
end
golf_angle = @slic begin 
    threshold_angle = asin((R - r) ./ x)
    sigma ~ flat(;lower=0)
    p = 2 * Phi(threshold_angle / sigma) - 1
    y ~ binomial(n, p)
    sigma_degrees = sigma * 180 / pi
end
r = (1.68/2)/12
R = (4.25/2)/12
golf_angle_distance_2 = Base.merge(golf_angle, quote 
    sigma ~ normal(0, 1; n=2, lower=0.)
    sigma_angle = sigma[1]
    sigma_distance = sigma[2]
    p_angle = 2 * Phi(threshold_angle / sigma_angle) - 1
    p_distance = (
        Phi((distance_tolerance - overshot) ./ ((x + overshot) * sigma_distance))
         - Phi((-overshot) ./ ((x + overshot) * sigma_distance))
    )
    p = p_angle .* p_distance
    sigma_degrees = sigma_angle * 180 / pi
end)

overshot = 1.
distance_tolerance = 3.
golf_angle_distance_3_with_resids = Base.merge(golf_angle_distance_2, quote
    vec_n = to_vector(n)
    sigma_y ~ normal(0, 1)
    y ~ normal(vec_n .* p, vec_n .* sqrt(p .* (1 - p) ./ to_vector(n) + sigma_y ^ 2))
end)
golf_angle_distance_4 = Base.merge(golf_angle_distance_2, quote 
    overshot ~ normal(1, 5; lower=0)
    distance_tolerance ~ normal(3, 5; lower=0)
end)

golf_models = (;
    golf_logistic=golf_logistic(;dataset1...),
    golf_angle=golf_angle(;r, R, dataset1...),
    golf_angle_distance_2=golf_angle_distance_2(;r, R, overshot, distance_tolerance, dataset12...),
    golf_angle_distance_3_with_resids=golf_angle_distance_3_with_resids(;r, R, overshot, distance_tolerance, dataset12...),
    golf_angle_distance_4=golf_angle_distance_4(;r, R, overshot, distance_tolerance, dataset12...),
)
""", [
    "Logistic regression" => :(golf_models.golf_logistic),
    "Angle model" => :(golf_models.golf_angle),
    "Angle and distance" => :(golf_models.golf_angle_distance_2),
    "Angle, distance, and residual variation" => :(golf_models.golf_angle_distance_3_with_resids),
    "Estimated overshoot and tolerance" => :(golf_models.golf_angle_distance_4),
])
```

```@raw html
</div>
```
