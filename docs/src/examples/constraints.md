# Reusable user-defined constraints

These examples show how a StanBlocks model can separate an unconstrained
parameter, its density and Jacobian adjustment, and the deterministic transform
that returns the constrained value. The source is adapted from an exploratory
Quarto notebook, so this page distinguishes reusable compiler patterns from
unfinished statistical definitions instead of presenting every historical
fragment as a ready-made prior.

The documentation build evaluates each displayed `@eval` source block and
calls `stan_code` on every resulting model. The generated-code tabs
therefore contain complete programs; users do not have to execute the examples
to see them.

## A two-dimensional disk transform

`uniform_disk_constrain` maps two unconstrained real values to polar
coordinates:

1. `radius = inv_logit(xi[1])` maps to `(0, 1)`;
2. `angle = 2pi * inv_logit(xi[2])` maps to one full turn; and
3. `x = radius * cos(angle)` and
   `y = radius * sin(angle)` lie inside the unit disk.

The function returns `(; radius, angle, x, y)`. StanBlocks traces that
Julia named tuple to a Stan tuple, while Julia field syntax such as
`theta.x` remains available in the model definition.

`@lhs @lpxf uniform_disk_lpdf(xi::vector[n], n)` gives sampling syntax two
pieces of information: `@lhs` declares an unconstrained vector of symbolic
length `n`, and `@lpxf` registers the method as a target-density
contribution. The `disk` submodel samples that vector at length two and
returns the transformed tuple. A larger model can consequently write
`theta ~ disk()` and consume `theta.x` and `theta.y` without
repeating the transform.

### Why the preserved source is not a uniform disk prior

The historical method returns a constant zero:

```julia
@lhs @lpxf uniform_disk_lpdf(xi::vector[n], n) = 0.
```

That is a transform demonstration, not a proper uniform-area prior. Write
`r = inv_logit(xi[1])` and `q = inv_logit(xi[2])`. The absolute
Jacobian determinant from `xi` to `(x, y)` is

`2pi * r^2 * (1-r) * q * (1-q)`.

A density that is uniform with respect to disk area must include the log of
that determinant in unconstrained coordinates (plus the constant disk
density). Returning zero omits the adjustment and starts from an improper flat
density on `xi`. It should not be used as a uniform prior in an analysis.
A production version would put both the Jacobian and any desired density on
`(x, y)` in the registered `_lpdf`, keeping
`uniform_disk_constrain` as the single deterministic map.

### Why the generated example has a dummy likelihood

The disk value otherwise affects no observation, so StanBlocks can legally
place the prior-independent calculation in generated quantities. The
`dummy` likelihood makes `theta` part of the model target and keeps
the parameter-transform path visible in the emitted program. It is scaffolding
for this compiler example, not a recommended observation model.

The model also uses `Base.merge` to insert `theta ~ disk()` into that
scaffold. This is the same fragment-reuse mechanism used by the model-family
examples.

## Generated disk model

```@raw html
<div class="atlas-comparison" data-atlas-comparison data-stan-label="Generated Stan model">
```

```@eval
Main.FeatureAtlasDocs.comparisons(Main, raw"""
using StanBlocks

@deffun begin 
    \"\"\"
    Historical transform demonstration: this zero density omits the Jacobian of
    `uniform_disk_constrain` and therefore does not define a uniform-area
    prior on the disk. The surrounding documentation derives the missing term.
    \"\"\"
    @lhs @lpxf uniform_disk_lpdf(xi::vector[n], n) = 0.
    \"\"\"
    This function returns the constrained parameters `x` and `y` together with the intermediate quantities `radius` and `angle` as a a Named Tuple, 
    which for Stan will look like a regular Tuple. 
    Within StanBlocks.jl, its possible to access the fields of the return value via `.` syntax, 
    e.g. `rv.radius` or `rv.x` after `rv = uniform_disk_constrain(xi)`.
    \"\"\"
    uniform_disk_constrain(xi::vector[n]) = begin
        radius = inv_logit(xi[1])
        angle = 2pi * inv_logit(xi[2])
        (;radius, angle, x=cos(angle) * radius, y=sin(angle) * radius)
    end
    "Needed to prevent StanBlocks.jl from moving `theta` to generated quantities"
    @lpxf dummy_lpdf(args...) = 0.
    "Needed to prevent StanBlocks.jl from moving `theta` to generated quantities"
    dummy_lpdfs(args...) = 0.
    "Needed to prevent StanBlocks.jl from moving `theta` to generated quantities"
    dummy_rng(args...) = 0.
end


disk = @slic begin
    \"\"\"
    This initializes the two unconstrained coordinates. In this preserved
    demonstration, `uniform_disk_lpdf` contributes zero; a production
    distribution must add its Jacobian and constrained-space prior there.
    \"\"\"
    xi ~ uniform_disk(2)
    \"\"\"
    Apply the deterministic disk transform and return its named components.
    \"\"\"
    return uniform_disk_constrain(xi)
end

dummy_model = @slic begin 
    "Needed to prevent StanBlocks.jl from moving `theta` to generated quantities"
    dummy_obs = 1. 
    "Needed to prevent StanBlocks.jl from moving `theta` to generated quantities"
    dummy_obs ~ dummy(theta)
end

disk_posteriors = (;
    disk=Base.merge(dummy_model, quote 
        theta ~ disk()
    end)
)
""", :disk_posteriors)
```

```@raw html
</div>
```

## Centered and non-centered hierarchical parameters

The original notebook left adaptive centering as a one-line reminder. The
underlying issue is a useful application of the same transform-plus-density
split.

For a hierarchical effect with location `mu` and scale `tau`, a
centered parameterization samples the effect directly,

```julia
theta ~ normal(mu, tau)
```

whereas a non-centered parameterization samples a standard-normal coordinate
and transforms it:

```julia
z ~ std_normal()
theta = mu + tau * z
```

These define the same prior on `theta` but different posterior geometries.
A partial-centering control must preserve that equality. For example, if a
data-supplied `rho` changes the scale in
`theta = mu + tau^rho * z`, then the density of `z` must change
correspondingly (or the change-of-variables Jacobian must be included). Merely
changing the transform changes the statistical model.

This is why it can be attractive to compute a custom transform's prior term and
Jacobian together, as discussed in the
[offset/multiplier thread](https://discourse.mc-stan.org/t/offset-multiplier-initialization/20712/20)
and the article on
[compound declare-distribute statements](https://statmodeling.stat.columbia.edu/2018/02/01/stan-feature-declare-distribute/).
StanBlocks can express fixed centered and non-centered fragments today and
select between reusable fragments with data or model construction. The
preserved notebook does not implement or validate a general automatic adaptive
centering transform, so there is deliberately no generated model for this
design note.

## Ten simplex transforms

The second executable family reproduces the transformation experiments from
[`bob-carpenter/transforms`](https://github.com/bob-carpenter/transforms).
Each model returns a length-ten simplex but uses a different unconstrained
coordinate system.

The reusable `any_simplex` fragment has two function-valued inputs:

```julia
any_simplex = @slic begin
    xi ~ simplex_prior(constrain_f, prior_f, n)
    return legacy_simplex_constrain(xi, constrain_f, n)
end
```

`constrain_f` maps unconstrained coordinates to a named tuple with fields
`jac` and `x`. `prior_f` supplies a density on the constrained
simplex; this page passes `uniform_simplex_lpdf`, whose contribution is
zero because the transform's Jacobian already determines the intended base
measure. Dispatch on `typeof(constrain_f)` selects
`unconstrained_dim`, allowing some transforms to use `n-1`
coordinates and others `n`.

### What differs between the family members

| Selector label | Coordinates and transform |
|:--|:--|
| **constrain_alr** | Additive log-ratio coordinates. It appends a reference log weight to `n-1` free log ratios, normalizes with log-sum-exp, and returns the softmax. |
| **constrain_expanded_softmax** | `n` free log weights followed by softmax. Softmax has a translation redundancy, so the Jacobian term also gives the log normalizer a standard-normal density to make the expanded representation proper. |
| **constrain_ilr** | An isometric log-ratio construction from `n-1` sequential balance coordinates, followed by softmax. |
| **constrain_ilr_reflector** | A reflector-based ILR construction. It maps `n-1` coordinates into `n` log coordinates with a reflector basis before softmax. |
| **constrain_normalized_exponential** | Maps `n` normal coordinates through the normal CDF and exponential quantile, then normalizes the resulting positive values. |
| **constrain_stickbreaking_angular** | Uses `n-1` logistic coordinates as angles and squares spherical-coordinate components to obtain positive values summing to one. |
| **constrain_stickbreaking_logistic** | A sequential stick-breaking map whose break fractions use shifted logistic coordinates. |
| **constrain_stickbreaking_normal** | The corresponding sequential map using shifted normal-CDF break fractions. |
| **constrain_stickbreaking_power_logistic** | A power stick-breaking construction driven by logistic-CDF values, with the remaining-stick powers depending on the component index. |
| **constrain_stickbreaking_power_normal** | The same power construction driven by normal-CDF values. |

The selector labels intentionally retain the function names from the source
repository. “ILR” means isometric log-ratio; “reflector” identifies the
alternative ILR basis construction; “logistic” versus “normal” identifies the
CDF used for a stick break; and “power” distinguishes the powered
remaining-stick construction from the ordinary sequential version.

### StanBlocks features in the generic implementation

- The symbolic type
  `vector[unconstrained_dim(constrain_f, n)]` is resolved through dispatch
  on a function value. One model definition can therefore declare either
  `n-1` or `n` unconstrained parameters.
- `simplex_prior_lpdf` evaluates `constrain_f(xi)` once for its
  `jac` and `x` fields, then adds `prior_f(tmp.x)`.
  `@lhs @lpxf` registers both the parameter declaration and target
  contribution.
- `legacy_simplex_constrain` returns the constrained `vector[n]`.
  The “legacy” prefix avoids colliding with the maintained StanBlocks builtin
  that now owns the original helper name.
- `@deffun @stanonly` says these definitions are emitted for Stan but are
  not required to have executable Julia counterparts. This is appropriate for
  helpers built directly from Stan probability and special functions.
- Mapping over a named tuple of function values creates all ten models. The
  comparison helper preserves those names as the generated-code selectors.
- As in the disk example, `Base.merge` inserts the reusable simplex
  fragment into a dummy likelihood scaffold so the compiler must retain the
  parameter path.

### The remaining duplicated work

The density calls `constrain_f(xi)` to obtain the Jacobian and constrained
value, and `legacy_simplex_constrain` calls it again to return the value to
the model. The generated Stan program therefore expresses the transform twice.
Keeping both operations in one named helper prevents their formulas from
drifting, but it does not fuse their evaluations across the density and
transformed value.

A future compiler-level paired-transform abstraction could make that
relationship explicit and reuse the intermediate. Until such an abstraction
exists, the honest contract is the one shown here: one transform function,
called in both contexts, with the extra evaluation visible in the generated
program rather than hidden by notebook prose.

## Generated simplex models

```@raw html
<div class="atlas-comparison" data-atlas-comparison data-stan-label="Generated Stan models">
```

```@eval
Main.FeatureAtlasDocs.comparisons(Main, raw"""
using StanBlocks
@deffun @stanonly begin
    Base.reverse(x::vector[n])::vector[n]
    StanBlocks.stan.std_normal_lcdf(x)::real
    Base.log2()::real
    std_normal_lcdfs(x::real) = std_normal_lcdf(x)
    std_normal_lcdfs(x::vector[n]) = jbroadcasted(std_normal_lcdfs, x)

    unconstrained_dim(constrain_f, n) = reject(n)
    @lhs @lpxf simplex_prior_lpdf(xi::vector[unconstrained_dim(constrain_f, n)], constrain_f, prior_f, n) = begin
        tmp = constrain_f(xi)
        tmp.jac + prior_f(tmp.x)
    end
    legacy_simplex_constrain(xi::vector[unconstrained_dim(constrain_f, n)], constrain_f, n)::vector[n] = constrain_f(xi).x
    uniform_simplex_lpdf(xi) = 0.
    
    constrain_alr(xi::vector[n]) = begin
        r = log1p_exp(log_sum_exp(xi))
        (;
            jac=sum(xi)-(n+1)*r, 
            x=exp(append_row(xi - r, -r))
        )
    end
    unconstrained_dim(::typeof(constrain_alr), n) = n-1
    constrain_expanded_softmax(xi::vector[n]) = begin
        r = log_sum_exp(xi)
        (;
        jac=std_normal_lpdf(r - log(n)) + sum(xi) - n * r, x=exp(xi - r))
    end
    unconstrained_dim(::typeof(constrain_expanded_softmax), n) = n
    constrain_ilr(xi::vector[n]) = begin 
        ns = linspaced_vector(n, 1, n)
        w = xi ./ sqrt(ns .* (ns + 1))
        z = append_row(reverse(cumulative_sum(reverse(w))), 0) - append_row(0, ns .* w)
        r = log_sum_exp(z)
        (;
            jac=0.5 * log(n+1)+sum(z) - (n+1) * r, 
            x=exp(z - r)
        )
    end
    unconstrained_dim(::typeof(constrain_ilr), n) = n-1
    constrain_ilr_reflector(xi::vector[n]) = begin 
        sqrtN = sqrt((n+1))
        zN = sum(xi) / sqrtN
        z = append_row(xi - zN ./ (sqrtN - 1), zN)
        r = log_sum_exp(z)
        (;
            jac=0.5 * log(n+1)+sum(z) - (n+1) * r, 
            x=exp(z - r)
        )
    end
    unconstrained_dim(::typeof(constrain_ilr_reflector), n) = n-1
    exponential_log_qf(x) = -log1m_exp(x)
    constrain_normalized_exponential(xi::vector[n]) = begin 
        z = log(exponential_log_qf(std_normal_lcdfs(xi)))
        r = log_sum_exp(z)
        (;
            jac=std_normal_lpdf(xi) - lgamma(n), 
            x=exp(z - r)
        )
    end
    unconstrained_dim(::typeof(constrain_normalized_exponential), n) = n
    constrain_stickbreaking_angular(xi::vector[n]) = begin 
        log_u = log_inv_logit(xi)
        log_phi = log_u + (log(pi) - log2())
        phi = exp(log_phi)
        log_s = log(sin(phi))
        log_c = log(cos(phi))
        log_s2_prod = append_row(0, 2 * cumulative_sum(log_s))
        (;
            jac=n * log2() + sum(log1m_exp(log_u)) + sum(log_s) + sum(log_phi)+sum(log_s2_prod[2:n]) + sum(log_c), 
            x=exp(log_s2_prod + append_row(2 * log_c, 0))
        )
    end
    unconstrained_dim(::typeof(constrain_stickbreaking_angular), n) = n-1
    constrain_stickbreaking_logistic(xi::vector[n]) = begin 
        log_z = log_inv_logit(xi - log(reverse(linspaced_vector(n, 1, n))))
        log_cum_prod = append_row(0, cumulative_sum(log1m_exp(log_z)))
        (;
            jac=sum(log_cum_prod) + sum(log_z), 
            x=exp(append_row(log_z, 0) + log_cum_prod)
        )
    end
    unconstrained_dim(::typeof(constrain_stickbreaking_logistic), n) = n-1
    constrain_stickbreaking_normal(xi::vector[n]) = begin 
        w = xi - log(reverse(linspaced_vector(n, 1, n))) / 2
        log_z = std_normal_lcdfs(w)
        log_cum_prod = append_row(0, cumulative_sum(log1m_exp(log_z)))
        (;
            jac=std_normal_lpdf(w) + sum(log_cum_prod[2:n]), 
            x=exp(append_row(log_z, 0) + log_cum_prod)
        )
    end
    unconstrained_dim(::typeof(constrain_stickbreaking_normal), n) = n-1
    constrain_stickbreaking_power_logistic(xi::vector[n]) = begin 
        log_u = log_inv_logit(xi)
        log_w = log_u ./ reverse(linspaced_vector(n, 1, n))
        (;
            jac=2 * sum(log_u) - sum(xi) - lgamma(n+1), 
            x=exp(append_row(log1m_exp(log_w), 0) + append_row(0, cumulative_sum(log_w)))
        )
    end
    unconstrained_dim(::typeof(constrain_stickbreaking_power_logistic), n) = n-1
    constrain_stickbreaking_power_normal(xi::vector[n]) = begin 
        log_u = std_normal_lcdfs(xi)
        log_w = log_u ./ reverse(linspaced_vector(n, 1, n))
        (;
            jac=std_normal_lpdf(xi) - lgamma(n+1), 
            x=exp(append_row(log1m_exp(log_w), 0) + append_row(0, cumulative_sum(log_w)))
        )
    end
    unconstrained_dim(::typeof(constrain_stickbreaking_power_normal), n) = n-1
end

any_simplex = @slic begin 
    xi ~ simplex_prior(constrain_f, prior_f, n)
    return legacy_simplex_constrain(xi, constrain_f, n)
end
dummy_simplex_model = Base.merge(dummy_model, quote 
    theta ~ any_simplex(;constrain_f, prior_f, n=10)
end)

simplex_posteriors = map((;
        constrain_alr, constrain_expanded_softmax, constrain_ilr, constrain_ilr_reflector, 
        constrain_normalized_exponential, 
        constrain_stickbreaking_angular, constrain_stickbreaking_logistic, constrain_stickbreaking_normal, 
        constrain_stickbreaking_power_logistic, constrain_stickbreaking_power_normal
    )) do constrain_f 
    dummy_simplex_model(;constrain_f, prior_f=uniform_simplex_lpdf)
end
""", :simplex_posteriors)
```

```@raw html
</div>
```
