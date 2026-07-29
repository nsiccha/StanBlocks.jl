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
    family_arg = 1,
    data_args = (2,),
    requires_data_lhs = true,
    lpdf = builtin.weighted_lpdf,
    lpmf = builtin.weighted_lpmf,
    lpdfs = builtin.weighted_lpdfs,
    lpmfs = builtin.weighted_lpmfs,
    rng = builtin.weighted_rng,
)

_family_function(x::StanExpr2{<:types.func}) = type(x).info.value
_family_function(x) = error(
    "distribution HOF: expected a base distribution function token, got $(type(x))"
)
_distribution_hof_family(spec, rhs::CanonicalExpr) = begin
    length(rhs.args) >= spec.family_arg || error(
        "distribution HOF: missing family argument $(spec.family_arg)"
    )
    _family_function(rhs.args[spec.family_arg])
end
_validate_distribution_hof(spec, lhs, rhs::CanonicalExpr) = begin
    if spec.requires_data_lhs && qual(lhs) != :data
        error(
            "weighted: observations must be data-qualified; weighted priors are not supported"
        )
    end
    for i in spec.data_args
        length(rhs.args) >= i || error("distribution HOF: missing data argument $i")
        qual(rhs.args[i]) == :data || error(
            "weighted: likelihood weights must be data-qualified; " *
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
_hof_probability(spec, family) =
    _probability_kind(family) === :lpdf ? spec.lpdf : spec.lpmf
_hof_pointwise(spec, family) =
    _probability_kind(family) === :lpdf ? spec.lpdfs : spec.lpmfs

lpxf_expr(lhs, rhs::StanExpr) = lpxf_expr(lhs, expr(rhs))
lpxf_expr(lhs, rhs::CanonicalExpr) = begin
    spec = _distribution_hof(head(rhs))
    isnothing(spec) && return stan_call(lpxf_expr(head(rhs)), lhs, rhs.args...)
    _validate_distribution_hof(spec, lhs, rhs)
    family = _distribution_hof_family(spec, rhs)
    stan_call(_hof_probability(spec, family), lhs, rhs.args...)
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
    stan_call(_hof_pointwise(spec, family), lhs, rhs.args...)
end
likelihood_expr(rhs) = error("$rhs is missing `likelihood_expr`")
# gq `~` synthesis: `rng_expr(token, rhs)` builds either `rng_fn(args...)` (for
# scalar tokens — matches Stan's native rng signatures) or `rng_fn(token, args...)`
# (for sized tokens — dispatched into per-shape `*_rng` @deffun overloads).
# The token is a `tokenof{T}` StanExpr carrying the wanted output shape.
rng_expr(token, rhs::StanExpr) = rng_expr(token, expr(rhs))
# Scalar token path: native Stan rng, no token forwarding.
rng_expr(token::StanExpr2{<:types.tokenof,0}, rhs::CanonicalExpr) = stan_call(rng_expr(head(rhs)), rhs.args...)
# Sized token path: prepend token so per-shape @deffun overloads dispatch.
rng_expr(token::StanExpr2{<:types.tokenof}, rhs::CanonicalExpr) = stan_call(rng_expr(head(rhs)), token, rhs.args...)
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
    typeof(builtin.density), typeof(builtin.pointwise), typeof(builtin.predictive)
}
_family_selector_target(::typeof(builtin.density), family) = lpxf_expr(family)
_family_selector_target(::typeof(builtin.pointwise), family) = likelihood_expr(family)
_family_selector_target(::typeof(builtin.predictive), family) = rng_expr(family)
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
