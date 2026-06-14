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
lpxf_expr(lhs, rhs::StanExpr) = lpxf_expr(lhs, expr(rhs))
lpxf_expr(lhs, rhs::CanonicalExpr) = stan_call(lpxf_expr(head(rhs)), lhs, rhs.args...)
for lpxf_rhs in (
    :dummy_lpdf,
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
    :neg_binomial_lpdf, :neg_binomial_2_lpmf, :neg_binomial_2_log_lpdf,
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
likelihood_expr(lhs, rhs::CanonicalExpr) = stan_call(likelihood_expr(head(rhs)), lhs, rhs.args...)
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
rng_expr(x) = error("$x is missing `rng_expr`")

