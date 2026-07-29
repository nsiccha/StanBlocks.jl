# `maybe_index(x, idx)` is a trace-time rewriting hook inserted by the
# partly-missing-vector body pre-pass.  After `forward!` resolves the head
# symbol to the bare function `builtin.maybe_index` (via canonical's
# StanExpr2{<:types.func} rewrite), this dispatch intercepts before
# `stan_expr` would need a `tracetype` for `maybe_index`:
#   ndim 0 (scalar) → return `x` unchanged (drop `idx`)
#   ndim 1 (vector) → return `x[idx]`
#   otherwise       → loud error (matrix / array not supported)
expand_inline_or_trace(x::CanonicalExpr{typeof(builtin.maybe_index)}; info) = begin
    nd = stan_ndim(x.args[1])
    nd == 0 && return x.args[1]
    nd == 1 && return stan_call(getindex, x.args[1], x.args[2])
    error(
        "maybe_index: only scalar (rank 0) and vector (rank 1) distribution args are " *
        "supported for partly-missing-vector imputation; got rank-$(nd) for " *
        "$(expr(x.args[1])). Matrix or inherently-joint distribution args cannot be " *
        "element-wise indexed. Use an explicit obs/mis split model instead."
    )
end
fold_shape_query(x::StanExpr{<:CanonicalExpr{typeof(getindex),<:Tuple{<:StanExpr{<:CanonicalExpr{typeof(builtin.dims)}},<:StanExpr{<:Integer}}}}) = begin
    inner = expr(expr(x).args[1])
    arg = inner.args[1]::StanExpr
    i = expr(expr(x).args[2])::Integer
    sz = stan_size(arg)
    if 1 <= i <= length(sz)
        candidate = sz[i]
        qual(candidate) == :data ? fold_shape_query(candidate) : x
    else
        x
    end
end
fetch_data!(x::StanType{<:types.tup}; info) = fetch_data!((stan_size(x), x.info.arg_types); info)

# A distribution HOF has one base-family token among its arguments and owns a
# continuous/discrete pair of aggregate + pointwise wrappers. Registration is
# deliberately generic: future combinators can opt in without adding new
# sampling, likelihood, or RNG lowering methods.
_distribution_hof(::Any) = nothing
_distribution_hof(::typeof(builtin.weighted)) = (
    name = :weighted,
    family_arg = 1,
    data_args = (2,),
    requires_data_lhs = true,
    lpdf = builtin.weighted_lpdf,
    lpmf = builtin.weighted_lpmf,
    lpdfs = builtin.weighted_lpdfs,
    lpmfs = builtin.weighted_lpmfs,
    rng = builtin.weighted_rng,
)
_distribution_hof(::typeof(builtin.conditioned)) = (
    name = :conditioned,
    family_arg = 1,
    data_args = (),
    requires_data_lhs = false,
    optional_kwargs = (:lower, :upper),
    variants = (
        lower = (
            head = builtin.lower_conditioning,
            lpdf = builtin.lower_conditioning_lpdf,
            lpmf = builtin.lower_conditioning_lpmf,
            lpdfs = builtin.lower_conditioning_lpdfs,
            lpmfs = builtin.lower_conditioning_lpmfs,
            rng = builtin.lower_conditioning_rng,
        ),
        upper = (
            head = builtin.upper_conditioning,
            lpdf = builtin.upper_conditioning_lpdf,
            lpmf = builtin.upper_conditioning_lpmf,
            lpdfs = builtin.upper_conditioning_lpdfs,
            lpmfs = builtin.upper_conditioning_lpmfs,
            rng = builtin.upper_conditioning_rng,
        ),
        both = (
            head = builtin.conditioning,
            lpdf = builtin.conditioning_lpdf,
            lpmf = builtin.conditioning_lpmf,
            lpdfs = builtin.conditioning_lpdfs,
            lpmfs = builtin.conditioning_lpmfs,
            rng = builtin.conditioning_rng,
        ),
    ),
)
_distribution_hof(::typeof(builtin.clamped)) = (
    name = :clamped,
    family_arg = 1,
    data_args = (),
    # A clamped law has threshold atoms and is therefore not a valid
    # continuously-parameterised HMC prior. It is an observation model.
    requires_data_lhs = true,
    optional_kwargs = (:lower, :upper),
    variants = (
        lower = (
            head = builtin.lower_clamping,
            lpdf = builtin.lower_clamping_lpdf,
            lpmf = builtin.lower_clamping_lpmf,
            lpdfs = builtin.lower_clamping_lpdfs,
            lpmfs = builtin.lower_clamping_lpmfs,
            rng = builtin.lower_clamping_rng,
        ),
        upper = (
            head = builtin.upper_clamping,
            lpdf = builtin.upper_clamping_lpdf,
            lpmf = builtin.upper_clamping_lpmf,
            lpdfs = builtin.upper_clamping_lpdfs,
            lpmfs = builtin.upper_clamping_lpmfs,
            rng = builtin.upper_clamping_rng,
        ),
        both = (
            head = builtin.clamping,
            lpdf = builtin.clamping_lpdf,
            lpmf = builtin.clamping_lpmf,
            lpdfs = builtin.clamping_lpdfs,
            lpmfs = builtin.clamping_lpmfs,
            rng = builtin.clamping_rng,
        ),
    ),
)
_distribution_hof(::typeof(builtin.interval_evidence)) = (
    name = :interval_evidence,
    family_arg = 1,
    data_args = (),
    requires_data_lhs = true,
    lpdf = builtin.interval_evidence_impl_lpdf,
    lpmf = builtin.interval_evidence_impl_lpmf,
    lpdfs = builtin.interval_evidence_impl_lpdfs,
    lpmfs = builtin.interval_evidence_impl_lpmfs,
    rng = builtin.interval_evidence_impl_rng,
)
_distribution_hof(::typeof(builtin.interval_evidence_impl)) =
    _distribution_hof(builtin.interval_evidence)
_fixed_distribution_hof(name, variant; requires_data_lhs = false) = (
    name,
    family_arg = 1,
    data_args = (),
    requires_data_lhs,
    lpdf = variant.lpdf,
    lpmf = variant.lpmf,
    lpdfs = variant.lpdfs,
    lpmfs = variant.lpmfs,
    rng = variant.rng,
)
_distribution_hof(::typeof(builtin.lower_conditioning)) = _fixed_distribution_hof(
    :conditioned, _distribution_hof(builtin.conditioned).variants.lower,
)
_distribution_hof(::typeof(builtin.upper_conditioning)) = _fixed_distribution_hof(
    :conditioned, _distribution_hof(builtin.conditioned).variants.upper,
)
_distribution_hof(::typeof(builtin.conditioning)) = _fixed_distribution_hof(
    :conditioned, _distribution_hof(builtin.conditioned).variants.both,
)
_distribution_hof(::typeof(builtin.lower_clamping)) = _fixed_distribution_hof(
    :clamped, _distribution_hof(builtin.clamped).variants.lower;
    requires_data_lhs = true,
)
_distribution_hof(::typeof(builtin.upper_clamping)) = _fixed_distribution_hof(
    :clamped, _distribution_hof(builtin.clamped).variants.upper;
    requires_data_lhs = true,
)
_distribution_hof(::typeof(builtin.clamping)) = _fixed_distribution_hof(
    :clamped, _distribution_hof(builtin.clamped).variants.both;
    requires_data_lhs = true,
)

_family_function(x::StanExpr2{<:types.func}) = type(x).info.value
_family_function(x) = error(
    "distribution HOF: expected a base distribution function token, got $(type(x))"
)
_is_compile_time_nothing(x) = x === nothing || x === :nothing ||
    (x isa QuoteNode && x.value === nothing)
_forward_call_kwargs(
    x::CanonicalExpr,
    resolved_head::StanExpr2{<:types.func};
    info,
) = begin
    spec = _distribution_hof(_family_function(resolved_head))
    optional = isnothing(spec) ? () : get(spec, :optional_kwargs, ())
    isempty(optional) && return forward!(x.kwargs; info)
    # A distribution HOF owns these kwargs at trace time. Explicit `nothing`
    # is equivalent to omission and must disappear before generic symbol/type
    # resolution; present bounds remain on the canonical call so `autotype`
    # can also use them as parameter constraints.
    kept = (;[
        key => value
        for (key, value) in pairs(x.kwargs)
        if !(key in optional && _is_compile_time_nothing(value))
    ]...)
    forward!(kept; info)
end
_distribution_hof_family(spec, rhs::CanonicalExpr) = begin
    length(rhs.args) >= spec.family_arg || error(
        "distribution HOF: missing family argument $(spec.family_arg)"
    )
    _family_function(rhs.args[spec.family_arg])
end
_validate_distribution_hof(spec, lhs, rhs::CanonicalExpr) = begin
    if spec.requires_data_lhs && qual(lhs) != :data
        error(
            "$(spec.name): observations must be data-qualified; $(spec.name) priors are not supported"
        )
    end
    for i in spec.data_args
        length(rhs.args) >= i || error("distribution HOF: missing data argument $i")
        qual(rhs.args[i]) == :data || error(
            "$(spec.name): likelihood control arguments must be data-qualified; " *
            "parameter-dependent likelihood weights are not supported"
        )
    end
    nothing
end
validate_sampling_rhs(lhs, rhs::StanExpr{<:CanonicalExpr}; info) = begin
    canonical = expr(rhs)
    spec = _distribution_hof(head(canonical))
    isnothing(spec) || _validate_distribution_hof(spec, lhs, canonical)
    nothing
end
_probability_kind(family) = begin
    probability = lpxf_expr(family)
    name = string(nameof(probability))
    endswith(name, "_lpdf") && return :lpdf
    endswith(name, "_lpmf") && return :lpmf
    error(
        "distribution HOF: family $(nameof(family)) resolves to $name; " *
        "expected an _lpdf or _lpmf family"
    )
end
_hof_variant(spec, rhs::CanonicalExpr) = begin
    optional = get(spec, :optional_kwargs, ())
    isempty(optional) && return spec
    present = Tuple(key for key in optional if key in keys(rhs.kwargs))
    mode = present == (:lower,) ? :lower :
        present == (:upper,) ? :upper :
        present == (:lower, :upper) ? :both : error(
            "$(spec.name): provide at least one of `lower=` or `upper=`"
        )
    getproperty(spec.variants, mode)
end
_hof_call_args(spec, rhs::CanonicalExpr) = begin
    optional = get(spec, :optional_kwargs, ())
    isempty(optional) && return rhs.args
    i = spec.family_arg
    bounds = Any[rhs.kwargs[key] for key in optional if key in keys(rhs.kwargs)]
    Tuple(vcat(collect(rhs.args[1:i]), bounds, collect(rhs.args[i+1:end])))
end
_hof_probability(spec, family, rhs) = begin
    variant = _hof_variant(spec, rhs)
    _probability_kind(family) === :lpdf ? variant.lpdf : variant.lpmf
end
_hof_pointwise(spec, family, rhs) = begin
    variant = _hof_variant(spec, rhs)
    _probability_kind(family) === :lpdf ? variant.lpdfs : variant.lpmfs
end
_hof_rng(spec, rhs) = _hof_variant(spec, rhs).rng

const CdfCapableBuiltinFamily = Union{typeof.((
    builtin.std_normal, builtin.normal, builtin.student_t, builtin.cauchy,
    builtin.beta, builtin.beta_proportion, builtin.lognormal, builtin.exponential,
    builtin.gamma, builtin.inv_gamma, builtin.weibull, builtin.uniform,
    builtin.chi_square, builtin.inv_chi_square, builtin.scaled_inv_chi_square,
    builtin.frechet, builtin.rayleigh, builtin.von_mises,
    builtin.double_exponential, builtin.logistic, builtin.gumbel,
    builtin.skew_normal, builtin.exp_mod_normal, builtin.skew_double_exponential,
    builtin.pareto, builtin.pareto_type_2,
    builtin.bernoulli, builtin.binomial, builtin.beta_binomial,
    builtin.neg_binomial, builtin.neg_binomial_2, builtin.poisson,
    builtin.discrete_range,
))...}

# Keep optional-bound mode in the call head/positional args, not only in
# `CanonicalExpr.kwargs`: later compiler passes intentionally rebuild call ASTs
# without kwargs after `autotype` has consumed declaration constraints. The
# rewrite therefore preserves the bounds twice for exactly one stage: as
# positional probability/RNG inputs, and as support constraints for autotype.
const OptionalBoundDistributionHOF = Union{
    typeof(builtin.conditioned), typeof(builtin.clamped)
}
expand_inline_or_trace(
    x::CanonicalExpr{<:OptionalBoundDistributionHOF};
    info,
) = begin
    spec = _distribution_hof(head(x))
    variant = _hof_variant(spec, x)
    forward!(CanonicalExpr(
        variant.head,
        _hof_call_args(spec, x)...;
        x.kwargs...,
    ); info)
end
expand_inline_or_trace(
    x::CanonicalExpr{typeof(builtin.interval_evidence)};
    info,
) = forward!(CanonicalExpr(
    builtin.interval_evidence_impl,
    x.args...;
    x.kwargs...,
); info)
_family_probability_companion(family, suffix::Symbol) = begin
    if parentmodule(family) === builtin && !(family isa CdfCapableBuiltinFamily)
        error(
            "distribution HOF: built-in family $(nameof(family)) has no Stan " *
            "`$(nameof(family))$(suffix)` companion; choose a CDF-capable family"
        )
    end
    probability = lpxf_expr(family)
    probability_name = string(nameof(probability))
    probability_suffix = _probability_kind(family) === :lpdf ? "_lpdf" : "_lpmf"
    base_name = probability_name[1:end-length(probability_suffix)]
    companion_name = Symbol(base_name, suffix)
    mod = parentmodule(probability)
    isdefined(mod, companion_name) || error(
        "distribution HOF: family $(nameof(family)) requires companion " *
        "`$companion_name`, but it is not defined in $mod"
    )
    getfield(mod, companion_name)
end
logcdf_expr(family) = _family_probability_companion(family, :_lcdf)
logccdf_expr(family) = _family_probability_companion(family, :_lccdf)

lpxf_expr(lhs, rhs::StanExpr) = lpxf_expr(lhs, expr(rhs))
lpxf_expr(lhs, rhs::CanonicalExpr) = begin
    spec = _distribution_hof(head(rhs))
    isnothing(spec) && return stan_call(lpxf_expr(head(rhs)), lhs, rhs.args...)
    _validate_distribution_hof(spec, lhs, rhs)
    family = _distribution_hof_family(spec, rhs)
    stan_call(_hof_probability(spec, family, rhs), lhs, _hof_call_args(spec, rhs)...)
end
for lpxf_rhs in (
    :dummy_lpdf,
    :truncated_normal_lpdf,
    :truncated_student_t_lpdf,
    :flat_lpdf, :std_normal_lpdf, :normal_lpdf, :student_t_lpdf, :cauchy_lpdf,
    :beta_lpdf, :beta_proportion_lpdf, :lognormal_lpdf, :exponential_lpdf, :gamma_lpdf,
    :inv_gamma_lpdf, :weibull_lpdf, :uniform_lpdf,
    :chi_square_lpdf, :inv_chi_square_lpdf, :scaled_inv_chi_square_lpdf,
    :frechet_lpdf, :rayleigh_lpdf, :loglogistic_lpdf, :von_mises_lpdf,
    :double_exponential_lpdf, :logistic_lpdf, :gumbel_lpdf,
    :skew_normal_lpdf, :exp_mod_normal_lpdf, :skew_double_exponential_lpdf,
    :pareto_lpdf, :pareto_type_2_lpdf, :wiener_lpdf,
    :dirichlet_lpdf, :multi_normal_lpdf, :multi_normal_prec_lpdf, :multi_normal_cholesky_lpdf,
    :multi_gp_lpdf, :multi_gp_cholesky_lpdf,
    :multi_student_t_lpdf, :multi_student_t_cholesky_lpdf,
    :gaussian_dlm_obs_lpdf,
    :lkj_corr_lpdf, :lkj_corr_cholesky_lpdf,
    :wishart_lpdf, :inv_wishart_lpdf, :inv_wishart_cholesky_lpdf, :wishart_cholesky_lpdf,
    :normal_id_glm_lpdf,
    :bernoulli_lpmf, :bernoulli_logit_lpmf, :bernoulli_logit_glm_lpmf,
    :binomial_lpmf, :binomial_logit_lpmf, :beta_binomial_lpmf,
    :neg_binomial_lpmf, :neg_binomial_2_lpmf, :neg_binomial_2_log_lpmf,
    :neg_binomial_2_log_glm_lpmf,
    :poisson_lpmf, :poisson_log_lpmf, :poisson_log_glm_lpmf,
    :discrete_range_lpmf, :hypergeometric_lpmf, :multinomial_lpmf,
    :categorical_lpmf, :categorical_logit_lpmf,
    :ordered_logistic_lpmf,
)
    base_rhs = Symbol(string(lpxf_rhs)[1:end-5])
    rng_rhs = Symbol(base_rhs, "_rng")
    lpxfs_rhs = Symbol(lpxf_rhs, "s")

    @eval lpxf_expr(::typeof(builtin.$base_rhs)) = builtin.$lpxf_rhs
    @eval rng_expr(::typeof(builtin.$base_rhs)) = builtin.$rng_rhs
    @eval likelihood_expr(::typeof(builtin.$base_rhs)) = builtin.$lpxfs_rhs
end

lpxf_expr(x) = error("$x is missing `lpxf_expr`")
likelihood_expr(lhs, rhs::StanExpr) = likelihood_expr(lhs, expr(rhs))
likelihood_expr(lhs, rhs::CanonicalExpr) = begin
    spec = _distribution_hof(head(rhs))
    isnothing(spec) && return stan_call(likelihood_expr(head(rhs)), lhs, rhs.args...)
    _validate_distribution_hof(spec, lhs, rhs)
    family = _distribution_hof_family(spec, rhs)
    stan_call(_hof_pointwise(spec, family, rhs), lhs, _hof_call_args(spec, rhs)...)
end
likelihood_expr(rhs) = error("$rhs is missing `likelihood_expr`")
# gq `~` synthesis: `rng_expr(token, rhs)` builds either `rng_fn(args...)` (for
# scalar tokens — matches Stan's native rng signatures) or `rng_fn(token, args...)`
# (for sized tokens — dispatched into per-shape `*_rng` @deffun overloads).
# The token is a `tokenof{T}` StanExpr carrying the wanted output shape.
rng_expr(token, rhs::StanExpr) = rng_expr(token, expr(rhs))
# Scalar token path: native Stan rng, no token forwarding.
rng_expr(token::StanExpr2{<:types.tokenof,0}, rhs::CanonicalExpr) = begin
    spec = _distribution_hof(head(rhs))
    isnothing(spec) && return stan_call(rng_expr(head(rhs)), rhs.args...)
    _distribution_hof_family(spec, rhs)
    stan_call(_hof_rng(spec, rhs), _hof_call_args(spec, rhs)...)
end
# Sized token path: prepend token so per-shape @deffun overloads dispatch.
rng_expr(token::StanExpr2{<:types.tokenof}, rhs::CanonicalExpr) = begin
    spec = _distribution_hof(head(rhs))
    isnothing(spec) && return stan_call(rng_expr(head(rhs)), token, rhs.args...)
    _distribution_hof_family(spec, rhs)
    stan_call(_hof_rng(spec, rhs), token, _hof_call_args(spec, rhs)...)
end
rng_expr(x) = begin
    spec = _distribution_hof(x)
    isnothing(spec) && error("$x is missing `rng_expr`")
    spec.rng
end

# Base-family companion selectors are compile-time calls. With no trailing
# arguments they return the selected function token; with trailing arguments
# they immediately trace a call to that function. In either form the selector
# itself is absent from emitted Stan.
const DistributionFamilySelector = Union{
    typeof(builtin.density), typeof(builtin.pointwise), typeof(builtin.predictive),
    typeof(builtin.logcdf), typeof(builtin.logccdf)
}
_family_selector_target(::typeof(builtin.density), family) = lpxf_expr(family)
_family_selector_target(::typeof(builtin.pointwise), family) = likelihood_expr(family)
_family_selector_target(::typeof(builtin.predictive), family) = rng_expr(family)
_family_selector_target(::typeof(builtin.logcdf), family) = logcdf_expr(family)
_family_selector_target(::typeof(builtin.logccdf), family) = logccdf_expr(family)
expand_inline_or_trace(
    x::CanonicalExpr{<:DistributionFamilySelector,<:Tuple{<:StanExpr2{<:types.func},Vararg{Any}}};
    info,
) = begin
    family = _family_function(x.args[1])
    selected = _family_selector_target(head(x), family)
    rest = x.args[2:end]
    isempty(rest) ? forward!(selected; info) :
        forward!(CanonicalExpr(selected, rest...); info)
end
