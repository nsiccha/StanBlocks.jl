builtin_module_names(x::Symbol) = endswith(string(x), r"_lp[md]f") ? [
    x
    Symbol(x, "s")
    Symbol(string(x)[1:end-length("_lpdf")])
    Symbol(string(x)[1:end-length("_lpdf")], "_rng")
    Symbol(string(x)[1:end-length("_lpdf")], "_cdf")
    Symbol(string(x)[1:end-length("_lpdf")], "_lccdf")
    Symbol(string(x)[1:end-length("_lpdf")], "_lcdf")
] : x
builtin_module_names(x::Expr) = mapreduce(builtin_module_names, vcat, x.args; init=[])
_head_or_type(x::Expr) = x.head
_head_or_type(x) = typeof(x)
macro builtin_module(x)
    @assert Meta.isexpr(x, :vcat) "@builtin_module expects a `[names...]` vector literal, got `$x` (head `$(_head_or_type(x))`)."
    names = builtin_module_names(x)
    esc(Expr(:block,
        Expr(:toplevel,
            Expr(:module, true, :builtin, Expr(:block, [
                Expr(:function, name)
                for name in names
            ]...))
        ),
        [
            Expr(:const, Expr(:(=), name, Expr(:(.), :builtin, QuoteNode(name))))
            for name in names
        ]...
    ))
end

@builtin_module [
    flat_lpdf
    std_normal_lpdf
    normal_lpdf
    student_t_lpdf
    cauchy_lpdf
    beta_lpdf
    beta_proportion_lpdf
    beta_binomial_lpmf
    binomial_lpmf
    binomial_logit_lpmf
    lognormal_lpdf
    chi_square_lpdf
    inv_chi_square_lpdf
    scaled_inv_chi_square_lpdf
    exponential_lpdf
    gamma_lpdf
    inv_gamma_lpdf
    weibull_lpdf
    frechet_lpdf
    rayleigh_lpdf
    loglogistic_lpdf
    uniform_lpdf
    von_mises_lpdf
    neg_binomial_2_lpmf
    bernoulli_lpmf
    bernoulli_logit_lpmf
    bernoulli_logit_glm_lpmf
    ordered_logistic_lpmf
    multi_normal_lpdf
    multi_normal_prec_lpdf
    multi_normal_cholesky_lpdf
    multi_gp_lpdf
    multi_gp_cholesky_lpdf
    multi_student_t_lpdf
    multi_student_t_cholesky_lpdf
    gaussian_dlm_obs_lpdf
    dirichlet_lpdf
    lkj_corr_lpdf
    lkj_corr_cholesky_lpdf
    wishart_lpdf
    inv_wishart_lpdf
    inv_wishart_cholesky_lpdf
    wishart_cholesky_lpdf

    vector_std_normal_rng
    log1m
    to_vector
    to_row_vector
    to_matrix
    rep_array
    rep_vector
    rep_matrix
    linspaced_array
    linspaced_int_array
    robust_linspaced_int_array
    linspaced_vector
    to_array_1d
    to_array_2d
    cholesky_decompose
    diag_pre_multiply
    diag_post_multiply
    mdivide_right_tri_low
    add_diag 
    gp_exp_quad_cov 
    inv_logit logit
    log_inv_logit
    log1m_exp
    Phi
    ode_rk45 ode_rk45_tol
    ode_ckrk ode_ckrk_tol
    ode_adams ode_adams_tol
    ode_bdf ode_bdf_tol
    append_array
    append_row
    append_col
    hcat
    reshape
    ragged_n ragged_total ragged_start ragged_end ragged_length
    diag_matrix
    mdivide_left_tri_low
    one_hot_vector
    sd
    mean
    cumulative_sum
    log_sum_exp
    lgamma
    matrix_exp
    log1p_exp log1m_exp
    sort_asc sort_desc
    sort_indices_asc sort_indices_desc
    dot_product rows_dot_product
    dims rows cols
    reject
    positive_infinity negative_infinity

    reduce_sum reduce_sum_static reduce_sum_reconstruct simple_reduce_sum simple_reduce_sum_helper

    broadcasted_getindex
    jbroadcasted jmap jsum

    # Unary math (Stan-specific, not Julia builtins)
    square log_diff_exp log_mix atan2
    is_inf is_nan
    Phi_approx inv_Phi
    erf erfc tgamma digamma trigamma
    lbeta inc_beta gamma_p gamma_q
    bessel_first_kind bessel_second_kind
    modified_bessel_first_kind modified_bessel_second_kind
    owens_t binary_log_loss
    fma fmin fmax fdim fmod cbrt

    # Array/vector operations
    norm1 norm2 distance squared_distance
    quantile num_elements segment variance

    # Matrix operations
    col row
    sub_col sub_row
    trace determinant log_determinant log_determinant_spd
    inverse inverse_spd chol2inv generalized_inverse
    eigenvalues_sym eigenvectors_sym
    crossprod tcrossprod
    quad_form quad_form_diag quad_form_sym
    trace_quad_form trace_gen_quad_form
    multiply_lower_tri_self_transpose
    dot_self columns_dot_product columns_dot_self rows_dot_self
    softmax log_softmax diagonal
    rep_row_vector linspaced_row_vector
    identity_matrix symmetrize_from_lower_tri
    mdivide_left_spd mdivide_right_spd
    matrix_power matrix_exp_multiply scale_matrix_exp_multiply

    # Additional GP covariance functions
    gp_exponential_cov gp_matern32_cov gp_matern52_cov
    gp_periodic_cov gp_dot_prod_cov

    # Additional distributions
    double_exponential_lpdf logistic_lpdf gumbel_lpdf
    skew_normal_lpdf exp_mod_normal_lpdf
    pareto_lpdf pareto_type_2_lpdf
    wiener_lpdf
    discrete_range_lpmf
    hypergeometric_lpmf
    multinomial_lpmf
    categorical_lpmf categorical_logit_lpmf
    poisson_lpmf poisson_log_lpmf
    neg_binomial_lpdf neg_binomial_2_log_lpdf
    skew_double_exponential_lpdf

    # No-op "distribution": `y ~ dummy(args...)` marks `y` observed while
    # contributing +0 to the density. Lets a prediction-only model keep a
    # scalar observed marker (`y = 1`) without resizing it to the prediction
    # grid. `_lpdf` auto-expands to `dummy` / `dummy_lpdfs` / `dummy_rng` etc.
    dummy_lpdf

    # Partly-missing-vector imputation helpers (auto-inserted by the body
    # pre-pass; not normally called directly by users).
    # `maybe_index` is a trace-time rewrite hook (lpxf_builtin.jl); it is
    # never emitted as a Stan function.  `merge_missing` is a real @deffun
    # that assembles the full vector in transformed_parameters.
    maybe_index
    merge_missing
]

# GLM distributions — defined outside @builtin_module to avoid Revise conflicts
for name in builtin_module_names(:normal_id_glm_lpdf)
    @eval builtin function $name end
    @eval const $name = builtin.$name
end
for name in builtin_module_names(:poisson_log_glm_lpmf)
    @eval builtin function $name end
    @eval const $name = builtin.$name
end
for name in builtin_module_names(:neg_binomial_2_log_glm_lpmf)
    @eval builtin function $name end
    @eval const $name = builtin.$name
end

for name in (:real, :int, :vector, :row_vector, :matrix, :ordered, :simplex, :positive_ordered, :cov_matrix, :corr_matrix, :cholesky_factor_cov, :cholesky_factor_corr)
    T = getproperty(types, name)
    @eval builtin const $name = $T
end

autokwargs(::CanonicalExpr{<:Union{typeof.((beta, beta_proportion))...}}) = (;lower=0, upper=1)
autokwargs(::CanonicalExpr{typeof(von_mises)}) = (;lower=0, upper=2pi)
autokwargs(x::CanonicalExpr{typeof(uniform)}) = (;lower=x.args[1], upper=x.args[2])
autokwargs(::CanonicalExpr{<:Union{typeof.((lognormal,chi_square,inv_chi_square,scaled_inv_chi_square,exponential,gamma,inv_gamma,weibull,frechet,rayleigh,loglogistic,pareto,pareto_type_2))...}}) = (;lower=0.)
autokwargs(::CanonicalExpr{<:Union{typeof.((logistic,gumbel,skew_normal,exp_mod_normal,double_exponential,skew_double_exponential))...}}) = (;)

import Statistics

@deffun begin 
    reduce_sum(args...)::real
    reduce_sum_static(args...)::real
    simple_reduce_sum(f, x, args...)::real = reduce_sum(simple_reduce_sum_helper, x, 1, f, args...)
    # Stan's `reduce_sum` only accepts `T[]` arrays as its data argument, not
    # `vector` (or its `simplex`/`ordered`/`positive_ordered` subtypes, or
    # `row_vector`). Auto-convert via `to_array_1d` so users can call
    # `simple_reduce_sum(f, ::any_vector, ...)` directly. `any_vector` covers
    # all 1-d vector-like Stan types in the SLIC hierarchy.
    simple_reduce_sum(f, x::any_vector[n], args...)::real = reduce_sum(simple_reduce_sum_helper, to_array_1d(x), 1, f, args...)
    simple_reduce_sum_helper(x_slice::anything[n], slice_start, slice_end, f, args...)::real = begin 
        rv = 0.
        for i in 1:n
            rv += f(x_slice[i], args...)
        end
        rv
    end
    positive_infinity()::real
    negative_infinity()::real
    reject(args...)::anything
    Base.log1p(x::real)::real
    Base.inv(::vector[n])::vector[n]
    Base.print(args...)::anything
    Base.size(x)::int
    Base.range(start::int, stop::int)::vector[stop]
    Base.sum(x)::real
    Base.sum(x::int[m])::int
    Base.sum(x::int[m,n])::int
    Base.sum(x::int[m,n,o])::int
    Base.:\(A::matrix[m, m], b::vector[m])::vector[m]
    mean(x)::real
    dims(x::anything[_])::int[1]
    dims(x::anything[_, _])::int[2]
    dims(x::anything[_, _, _])::int[3]
    cols(::matrix[m,n])::int
    rows(::matrix[m,n])::int
    # cumulative_sum moved to @defsig (see below) to include row_vector form
    diag_matrix(x::anything[n])::matrix[n,n]
    sd(x)::real
    one_hot_vector(n, k)::vector[n]
    mdivide_left_tri_low(::matrix[m,m], ::vector[m])::vector[m]
    mdivide_left_tri_low(::matrix[m,m], ::matrix[m,n])::matrix[m,n]
    linspaced_array(n, x, y)::real[n]
    linspaced_int_array(n, args...)::int[n]
    robust_linspaced_int_array(n, args...)::int[n] = if n == 0
        rv::int[n]
        rv 
    else
        linspaced_int_array(n, args...)
    end
    linspaced_vector(n, x, y)::vector[n]
    to_matrix(v, m, n)::matrix[m,n]
    rep_array(x::int, n)::int[n]
    rep_array(x::int, m, n)::int[m, n]
    rep_array(x::real, n)::real[n]
    rep_array(x::real, m, n)::real[m, n]
    rep_vector(v, n)::vector[n]
    rep_matrix(v::vector[m], n)::matrix[m, n]
    rep_matrix(x::real, m, n)::matrix[m,n]
    to_array_2d(v, m, n)::real[m,n]
    dot_product(x::vector[n], y::vector[n])::real
    matrix_exp(x::matrix[m,m])::matrix[m,m]
    rows_dot_product(x::matrix[m,n], y::matrix[m,n])::vector[m]
    append_col(x::anything[n], y::anything[n])::matrix[n,2]
    append_col(x::matrix[m, n1], y::matrix[m, n2])::matrix[m, n1+n2]
    append_col(x::anything[m], y::matrix[m, n2])::matrix[m, 1+n2]
    append_col(x::matrix[m, n1], y::anything[m])::matrix[m, n1+1]
    append_array(lhs::anything[m],rhs::anything[n])::real[m+n]
    append_array(lhs::anything[m],rhs::real)::real[m+1]
    append_row(lhs::vector[m],rhs::real)::vector[m+1]
    append_row(lhs::vector[m],rhs::vector[n])::vector[m+n]
    append_row(::real, ::vector[n])::vector[n+1]
    append_row(::matrix[m1,n], ::matrix[m2,n])::matrix[m1+m2,n]
    append_row(x::row_vector[n], y::row_vector[n])::matrix[2, n]
    append_row(x::matrix[m, n], y::row_vector[n])::matrix[m+1, n]
    flat_lpdf(args...)
    # `dummy` no-op distribution: `y ~ dummy(args...)` contributes +0. Bodies
    # ignore their args; monomorphization emits a concrete Stan
    # `real dummy_lpdf(<y>, <args>) { return 0.; }` per call-site signature, so
    # the native `y ~ dummy(args...);` resolves. `dummy_lpdfs` is the gq
    # pointwise-likelihood term, `dummy_rng` the gq draw — both trivially 0.
    dummy_lpdf(y, args...) = 0.
    dummy_lpdfs(y, args...) = 0.
    dummy_rng(args...) = 0.
    std_normal_lpdf(args...)
    normal_lpdf(args...)
    student_t_lpdf(args...)
    cauchy_lpdf(args...)
    beta_lpdf(args...)
    beta_proportion_lpdf(args...)
    beta_binomial_lpmf(args...)
    binomial_lpmf(args...)
    binomial_logit_lpmf(args...)
    lognormal_lpdf(args...)
    chi_square_lpdf(args...)
    inv_chi_square_lpdf(args...)
    scaled_inv_chi_square_lpdf(args...)
    scaled_inv_chi_square_rng(::real, ::real)::real
    exponential_lpdf(args...)
    gamma_lpdf(args...)
    inv_gamma_lpdf(args...)
    weibull_lpdf(args...)
    frechet_lpdf(args...)
    rayleigh_lpdf(args...)
    loglogistic_lpdf(args...)
    uniform_lpdf(args...)
    von_mises_lpdf(args...)
    neg_binomial_2_lpmf(args...)
    bernoulli_lpmf(args...)
    bernoulli_logit_lpmf(args...)
    bernoulli_logit_glm_lpmf(args...)
    normal_id_glm_lpdf(args...)
    poisson_log_glm_lpmf(args...)
    neg_binomial_2_log_glm_lpmf(args...)
    @lhs multi_normal_lpdf(obs::vector[n], loc::vector[n], cov)
    @lhs dirichlet_lpdf(w::simplex[n], alpha::vector[n])
    @lhs lkj_corr_lpdf(L::corr_matrix, x::real)
    @lhs lkj_corr_cholesky_lpdf(L::cholesky_factor_corr, x::real)
    @lhs wishart_lpdf(L::cov_matrix[m], x::real, sigma::matrix[m,m])
    @lhs wishart_cholesky_lpdf(L::cholesky_factor_cov[m], x::real, sigma::matrix[m,m])

    lognormal_rng(loc::real, scale::real)::real
    student_t_rng(nu::real, loc::real, scale::real)::real
    multi_normal_rng(loc::vector[n], args...)::vector[n]
    multi_normal_cholesky_rng(loc::vector[n], scale)::vector[n]
    bernoulli_rng(::vector[n])::int[n]
    bernoulli_logit_rng(::real)::int
    bernoulli_logit_rng(::vector[n])::int[n]
    bernoulli_logit_glm_rng(X::matrix[m,n], alpha, beta)::int[m]
    bernoulli_logit_glm_rng(X::matrix[m,n], alpha::real, beta) = bernoulli_logit_glm_rng(X, rep_vector(alpha, m), beta)
    normal_id_glm_rng(X::matrix[m,n], alpha, beta, sigma)::vector[m]
    normal_id_glm_rng(X::matrix[m,n], alpha::real, beta, sigma) = normal_id_glm_rng(X, rep_vector(alpha, m), beta, sigma)
    poisson_log_glm_rng(X::matrix[m,n], alpha, beta)::int[m]
    poisson_log_glm_rng(X::matrix[m,n], alpha::real, beta) = poisson_log_glm_rng(X, rep_vector(alpha, m), beta)
    neg_binomial_2_log_glm_rng(X::matrix[m,n], alpha, beta, phi)::int[m]
    neg_binomial_2_log_glm_rng(X::matrix[m,n], alpha::real, beta, phi) = neg_binomial_2_log_glm_rng(X, rep_vector(alpha, m), beta, phi)
    beta_rng(args...)::real
    binomial_rng(args...)::int
    binomial_logit_rng(n::int[m], p::vector[m])::int[m]
    binomial_logit_rng(n::vector[m], p::vector[m])::int[m]
    broadcasted_getindex(x, i) = x
    broadcasted_getindex(x::anything[m], i) = x[i]
    jbroadcasted(f, x1::anything[n]) = begin 
        rv::vector[n]
        for i in 1:n
            rv[i] = f(broadcasted_getindex(x1, i))
        end
        rv
    end
    jbroadcasted(f, x1::anything[n], x2) = begin 
        rv::vector[n]
        for i in 1:n
            rv[i] = f(broadcasted_getindex(x1, i), broadcasted_getindex(x2, i))
        end
        rv
    end
    jbroadcasted(f, x1::anything[n], x2, x3) = begin
        rv::vector[n]
        for i in 1:n
            rv[i] = f(broadcasted_getindex(x1, i), broadcasted_getindex(x2, i), broadcasted_getindex(x3, i))
        end
        rv
    end
    jbroadcasted(f, x1::anything[n], x2, x3, x4) = begin
        rv::vector[n]
        for i in 1:n
            rv[i] = f(broadcasted_getindex(x1, i), broadcasted_getindex(x2, i), broadcasted_getindex(x3, i), broadcasted_getindex(x4, i))
        end
        rv
    end
    vector_std_normal_rng(n::int)::vector[n] = to_vector(normal_rng(rep_vector(0, n), 1))
    # Sized-token rng overloads are generated via `@eval @deffun` loops below
    # (after the block closes). See comment at the @eval block.
    bernoulli_lpmfs(args...) = bernoulli_lpmf(args...)
    bernoulli_lpmfs(obs::anything[n], args...) = jbroadcasted(bernoulli_lpmfs, obs, args...)
    bernoulli_logit_lpmfs(args...) = bernoulli_logit_lpmf(args...)
    bernoulli_logit_lpmfs(obs::anything[n], args...) = jbroadcasted(bernoulli_logit_lpmfs, obs, args...)
    bernoulli_logit_glm_lpmfs(y::int[n], X::matrix[n,k], alpha, beta) = begin
        rv::vector[n]
        for i in 1:n
            rv[i] = bernoulli_logit_lpmf(y[i], alpha + X[i, :] * beta)
        end
        rv
    end
    normal_id_glm_lpdfs(y::anything[n], X::matrix[n,k], alpha, beta, sigma) = begin
        rv::vector[n]
        for i in 1:n
            rv[i] = normal_lpdf(y[i], alpha + X[i, :] * beta, sigma)
        end
        rv
    end
    poisson_log_glm_lpmfs(y::int[n], X::matrix[n,k], alpha, beta) = begin
        rv::vector[n]
        for i in 1:n
            rv[i] = poisson_log_lpmf(y[i], alpha + X[i, :] * beta)
        end
        rv
    end
    neg_binomial_2_log_glm_lpmfs(y::int[n], X::matrix[n,k], alpha, beta, phi) = begin
        rv::vector[n]
        for i in 1:n
            rv[i] = neg_binomial_2_lpmf(y[i], exp(alpha + X[i, :] * beta), phi)
        end
        rv
    end
    binomial_lpmfs(args...) = binomial_lpmf(args...)
    binomial_lpmfs(y::int[n], args...) = jbroadcasted(binomial_lpmfs, y, args...)
    binomial_logit_lpmfs(args...) = binomial_logit_lpmf(args...)
    binomial_logit_lpmfs(y::int[n], args...) = jbroadcasted(binomial_logit_lpmfs, y, args...)
    normal_lpdfs(args...) = normal_lpdf(args...)
    normal_lpdfs(obs::anything[n], loc, scale) = jbroadcasted(normal_lpdfs, obs, loc, scale)
    multi_normal_lpdfs(args...) = multi_normal_lpdf(args...)
    dirichlet_lpdfs(args...) = dirichlet_lpdf(args...)
    @lhs lkj_corr_cholesky_lpdf(L::cholesky_factor_corr[m,n], x::real, m::int, n::int)::real = begin
        rv = 0.0
        for i in 1:m
            rv += lkj_corr_cholesky_lpdf(L[i, :, :], x)
        end
        rv
    end
    lkj_corr_cholesky_lpdfs(args...) = lkj_corr_cholesky_lpdf(args...)
    lkj_corr_cholesky_lpdfs(L::anything[n], x) = jbroadcasted(lkj_corr_cholesky_lpdfs, L, x)
    # Scalar-fallback _lpdfs for distributions missing vectorized forms.
    # The args... variant handles scalar obs; array broadcasting is left for future work.
    cauchy_lpdfs(args...) = cauchy_lpdf(args...)
    cauchy_lpdfs(obs::anything[n], loc, scale) = jbroadcasted(cauchy_lpdfs, obs, loc, scale)
    student_t_lpdfs(args...) = student_t_lpdf(args...)
    lognormal_lpdfs(args...) = lognormal_lpdf(args...)
    lognormal_lpdfs(obs::anything[n], loc, scale) = jbroadcasted(lognormal_lpdfs, obs, loc, scale)
    gamma_lpdfs(args...) = gamma_lpdf(args...)
    gamma_lpdfs(obs::anything[n], alpha, beta) = jbroadcasted(gamma_lpdfs, obs, alpha, beta)
    beta_lpdfs(args...) = beta_lpdf(args...)
    beta_lpdfs(obs::anything[n], alpha, beta) = jbroadcasted(beta_lpdfs, obs, alpha, beta)
    exponential_lpdfs(args...) = exponential_lpdf(args...)
    exponential_lpdfs(obs::anything[n], rate) = jbroadcasted(exponential_lpdfs, obs, rate)
    uniform_lpdfs(args...) = uniform_lpdf(args...)
    uniform_lpdfs(obs::anything[n], lo, hi) = jbroadcasted(uniform_lpdfs, obs, lo, hi)
    neg_binomial_2_lpmfs(args...) = neg_binomial_2_lpmf(args...)
    neg_binomial_2_lpmfs(obs::anything[n], mu, phi) = jbroadcasted(neg_binomial_2_lpmfs, obs, mu, phi)
    poisson_lpmfs(args...) = poisson_lpmf(args...)
    poisson_lpmfs(obs::anything[n], lambda) = jbroadcasted(poisson_lpmfs, obs, lambda)
    poisson_log_lpmfs(args...) = poisson_log_lpmf(args...)
    poisson_log_lpmfs(obs::anything[n], alpha) = jbroadcasted(poisson_log_lpmfs, obs, alpha)
    ordered_logistic_lpmfs(args...) = ordered_logistic_lpmf(args...)
    ordered_logistic_lpmfs(y::int[n], eta::vector[n], c::vector[m]) = begin
        rv::vector[n]
        for i in 1:n
            rv[i] = ordered_logistic_lpmf(y[i], eta[i], c)
        end
        rv
    end
    ordered_logistic_rng(eta::vector[n], c::vector[m])::int[n] = begin
        rv::int[n]
        for i in 1:n
            rv[i] = ordered_logistic_rng(eta[i], c)
        end
        rv
    end
    ordered_logistic_rng(int[n], eta::vector[n], c::vector[m])::int[n] = ordered_logistic_rng(eta, c)
    inv_gamma_lpdfs(args...) = inv_gamma_lpdf(args...)
    inv_gamma_lpdfs(obs::anything[n], alpha, beta) = jbroadcasted(inv_gamma_lpdfs, obs, alpha, beta)
    double_exponential_lpdfs(args...) = double_exponential_lpdf(args...)
    double_exponential_lpdfs(obs::anything[n], mu, sigma) = jbroadcasted(double_exponential_lpdfs, obs, mu, sigma)
    logistic_lpdfs(args...) = logistic_lpdf(args...)
    logistic_lpdfs(obs::anything[n], mu, sigma) = jbroadcasted(logistic_lpdfs, obs, mu, sigma)
    weibull_lpdfs(args...) = weibull_lpdf(args...)
    weibull_lpdfs(obs::anything[n], alpha, sigma) = jbroadcasted(weibull_lpdfs, obs, alpha, sigma)
    gumbel_lpdfs(args...) = gumbel_lpdf(args...)
    gumbel_lpdfs(obs::anything[n], mu, beta) = jbroadcasted(gumbel_lpdfs, obs, mu, beta)
    chi_square_lpdfs(args...) = chi_square_lpdf(args...)
    chi_square_lpdfs(obs::anything[n], nu) = jbroadcasted(chi_square_lpdfs, obs, nu)
    skew_normal_lpdfs(args...) = skew_normal_lpdf(args...)
    skew_normal_lpdfs(obs::anything[n], mu, sigma, alpha) = jbroadcasted(skew_normal_lpdfs, obs, mu, sigma, alpha)
    frechet_lpdfs(args...) = frechet_lpdf(args...)
    frechet_lpdfs(obs::anything[n], alpha, sigma) = jbroadcasted(frechet_lpdfs, obs, alpha, sigma)
    rayleigh_lpdfs(args...) = rayleigh_lpdf(args...)
    rayleigh_lpdfs(obs::anything[n], sigma) = jbroadcasted(rayleigh_lpdfs, obs, sigma)
    loglogistic_lpdfs(args...) = loglogistic_lpdf(args...)
    loglogistic_lpdfs(obs::anything[n], alpha, beta) = jbroadcasted(loglogistic_lpdfs, obs, alpha, beta)
    von_mises_lpdfs(args...) = von_mises_lpdf(args...)
    von_mises_lpdfs(obs::anything[n], mu, kappa) = jbroadcasted(von_mises_lpdfs, obs, mu, kappa)
    inv_chi_square_lpdfs(args...) = inv_chi_square_lpdf(args...)
    inv_chi_square_lpdfs(obs::anything[n], nu) = jbroadcasted(inv_chi_square_lpdfs, obs, nu)
    scaled_inv_chi_square_lpdfs(args...) = scaled_inv_chi_square_lpdf(args...)
    scaled_inv_chi_square_lpdfs(obs::anything[n], nu, sigma) = jbroadcasted(scaled_inv_chi_square_lpdfs, obs, nu, sigma)
    exp_mod_normal_lpdfs(args...) = exp_mod_normal_lpdf(args...)
    exp_mod_normal_lpdfs(obs::anything[n], mu, sigma, lambda) = jbroadcasted(exp_mod_normal_lpdfs, obs, mu, sigma, lambda)
    pareto_lpdfs(args...) = pareto_lpdf(args...)
    pareto_lpdfs(obs::anything[n], y_min, alpha) = jbroadcasted(pareto_lpdfs, obs, y_min, alpha)
    pareto_type_2_lpdfs(args...) = pareto_type_2_lpdf(args...)
    pareto_type_2_lpdfs(obs::anything[n], mu, lambda, alpha) = jbroadcasted(pareto_type_2_lpdfs, obs, mu, lambda, alpha)
    beta_binomial_lpmfs(args...) = beta_binomial_lpmf(args...)
    beta_binomial_lpmfs(obs::anything[n], trials, alpha, beta) = jbroadcasted(beta_binomial_lpmfs, obs, trials, alpha, beta)
    student_t_lpdfs(obs::anything[n], nu, loc, scale) = jbroadcasted(student_t_lpdfs, obs, nu, loc, scale)
    vector_exponential_rng(rate::real, n::int)::vector[n] = exponential_rng(rep_vector(rate, n))
    lkj_corr_cholesky_rng(n::int, eta::real)
    
    normal_cdf(args...)
    normal_lcdf(args...)
    normal_lccdf(args...)

    append_row(x, y, z, args...) = append_row(append_row(x, y), z, args...)
    append_col(x, y, z, args...) = append_col(append_col(x, y), z, args...)

    # hcat for building a design matrix from column vectors. Narrow on purpose:
    # Stan's matrix / array-of-vectors / array-of-ints are distinct types, so
    # add more signatures only as sbimpl (or other callers) actually need them.
    hcat(x::vector[n])::matrix[n,1] = to_matrix(x, n, 1)
    hcat(x::vector[n], y::vector[n])::matrix[n,2] = append_col(x, y)
    hcat(x::matrix[m, n], y::vector[m])::matrix[m, n+1] = append_col(x, y)
    hcat(x, y, z, args...) = hcat(hcat(x, y), z, args...)

    # reshape: vector -> matrix only, matching Base.reshape(v, m, k) with fully
    # specified dims. Grow as needed.
    reshape(v::vector[n], m::int, k::int)::matrix[m, k] = to_matrix(v, m, k)

    # Legacy ragged-vector accessors operating on bare `ntup` values. Kept
    # for back-compat with existing models; new code should prefer the
    # `RaggedVector` usertype + Julia-dispatched `Base.length` /
    # `Base.getindex` methods declared below.
    ragged_n(x::ntup)::int = size(x.ends)
    ragged_total(x::ntup)::int = num_elements(x.mem)
    ragged_start(x::ntup, i::int)::int = if i == 1
        1
    else
        1 + x.ends[i-1]
    end
    ragged_end(x::ntup, i::int)::int = x.ends[i]
    ragged_length(x::ntup, i::int)::int = ragged_end(x, i) - ragged_start(x, i) + 1

    # New distribution lpdf/lpmf stubs
    double_exponential_lpdf(args...)
    logistic_lpdf(args...)
    gumbel_lpdf(args...)
    skew_normal_lpdf(args...)
    exp_mod_normal_lpdf(args...)
    pareto_lpdf(args...)
    pareto_type_2_lpdf(args...)
    wiener_lpdf(args...)
    discrete_range_lpmf(args...)
    hypergeometric_lpmf(args...)
    multinomial_lpmf(args...)
    categorical_lpmf(args...)
    categorical_logit_lpmf(args...)
    ordered_logistic_lpmf(args...)
    poisson_lpmf(args...)
    poisson_log_lpmf(args...)
    neg_binomial_lpdf(args...)
    neg_binomial_2_log_lpdf(args...)
    skew_double_exponential_lpdf(args...)

    # RNG signatures for distributions already in @builtin_module
    gamma_rng(::real, ::real)::real
    inv_gamma_rng(::real, ::real)::real
    chi_square_rng(::real)::real
    inv_chi_square_rng(::real)::real
    weibull_rng(::real, ::real)::real
    frechet_rng(::real, ::real)::real
    rayleigh_rng(::real)::real
    loglogistic_rng(::real, ::real)::real
    uniform_rng(::real, ::real)::real
    von_mises_rng(::real, ::real)::real
    neg_binomial_2_rng(::real, ::real)::int
    neg_binomial_2_rng(::vector[n], ::real)::int[n]
    neg_binomial_2_rng(::real, ::vector[n])::int[n]
    neg_binomial_2_rng(::vector[n], ::vector[n])::int[n]
    beta_binomial_rng(::int, ::real, ::real)::int
    binomial_rng(::int, ::real)::int
    binomial_rng(::int[n], ::real)::int[n]
    binomial_rng(::int[n], ::vector[n])::int[n]
    bernoulli_rng(::real)::int

    # New distribution RNG signatures
    double_exponential_rng(::real, ::real)::real
    logistic_rng(::real, ::real)::real
    gumbel_rng(::real, ::real)::real
    skew_normal_rng(::real, ::real, ::real)::real
    exp_mod_normal_rng(::real, ::real, ::real)::real
    pareto_rng(::real, ::real)::real
    pareto_type_2_rng(::real, ::real, ::real)::real
    categorical_rng(::vector[n])::int
    categorical_logit_rng(::vector[n])::int
    ordered_logistic_rng(::real, ::vector[m])::int
    poisson_rng(::real)::int
    poisson_rng(::vector[n])::int[n]
    poisson_log_rng(::real)::int
    poisson_log_rng(::vector[n])::int[n]
    neg_binomial_rng(::real, ::real)::int
    neg_binomial_rng(::vector[n], ::real)::int[n]
    neg_binomial_rng(::real, ::vector[n])::int[n]
    neg_binomial_rng(::vector[n], ::vector[n])::int[n]
    discrete_range_rng(::int, ::int)::int
    dirichlet_rng(::vector[n])::vector[n]
    multinomial_rng(::vector[n], ::int)::int[n]

    # Matrix construction
    rep_row_vector(x, n)::row_vector[n]
    linspaced_row_vector(n, x, y)::row_vector[n]
    identity_matrix(n::int)::matrix[n,n]
    symmetrize_from_lower_tri(::matrix[n,n])::matrix[n,n]

    # Matrix slicing
    col(A::matrix[m,n], j)::vector[m]
    row(A::matrix[m,n], i)::row_vector[n]
    sub_col(A::matrix[m,n], i, j, nrow)::vector[nrow]
    sub_row(A::matrix[m,n], i, j, ncol)::row_vector[ncol]
    segment(v::vector[m], i::int, n)::vector[n]
    segment(v::row_vector[m], i::int, n)::row_vector[n]
    segment(v::anything[m], i::int, n)::real[n]

    # num_elements
    num_elements(::vector[n])::int
    num_elements(::row_vector[n])::int
    num_elements(::matrix[m,n])::int
    num_elements(::anything[n])::int

    # rows/cols for vectors
    rows(::vector[n])::int
    rows(::row_vector[n])::int
    cols(::vector[n])::int
    cols(::row_vector[n])::int

    # Softmax (in @builtin_module, so @deffun can reference builtin.softmax)
    softmax(v::vector[n])::vector[n]
    log_softmax(v::vector[n])::vector[n]

    # Scalar math
    square(x::real)::real
    log_diff_exp(x::real, y::real)::real
    log_mix(theta::real, lp1::real, lp2::real)::real
    atan2(y::real, x::real)::real
    fma(x::real, y::real, z::real)::real
    fmin(x::real, y::real)::real
    fmax(x::real, y::real)::real
    fdim(x::real, y::real)::real
    fmod(x::real, y::real)::real
    binary_log_loss(y::int, y_hat::real)::real
    lbeta(a::real, b::real)::real
    inc_beta(a::real, b::real, x::real)::real
    gamma_p(a::real, x::real)::real
    gamma_q(a::real, x::real)::real
    owens_t(h::real, a::real)::real
    bessel_first_kind(v::int, x::real)::real
    bessel_second_kind(v::int, x::real)::real
    modified_bessel_first_kind(v::int, x::real)::real
    modified_bessel_second_kind(v::int, x::real)::real
    is_inf(x::real)::int
    is_nan(x::real)::int

    # GP covariance functions
    gp_exponential_cov(x::real[n], sigma::real, l::real)::matrix[n,n]
    gp_exponential_cov(x1::real[m], x2::real[n], sigma::real, l::real)::matrix[m,n]
    gp_matern32_cov(x::real[n], sigma::real, l::real)::matrix[n,n]
    gp_matern32_cov(x1::real[m], x2::real[n], sigma::real, l::real)::matrix[m,n]
    gp_matern52_cov(x::real[n], sigma::real, l::real)::matrix[n,n]
    gp_matern52_cov(x1::real[m], x2::real[n], sigma::real, l::real)::matrix[m,n]
    gp_periodic_cov(x::real[n], sigma::real, l::real, p::real)::matrix[n,n]
    gp_periodic_cov(x1::real[m], x2::real[n], sigma::real, l::real, p::real)::matrix[m,n]
    gp_dot_prod_cov(x::real[n], sigma::real)::matrix[n,n]
    gp_dot_prod_cov(x1::real[m], x2::real[n], sigma::real)::matrix[m,n]

    Base.invperm(x::int[n])::int[n] = begin
        rv = rep_array(0, n)
        for i in 1:n
            rv[x[i]] = i
        end
        rv
    end

    # Assemble a full vector from observed and imputed-missing parts.
    # Emitted into `transformed_parameters` by the body pre-pass when a
    # partly-missing data vector is detected (auto-detect Union{Missing}).
    merge_missing(y_obs::vector[n_obs], y_mis::vector[n_mis], ii_obs::int[n_obs], ii_mis::int[n_mis])::vector[n_obs+n_mis] = begin
        y::vector[n_obs+n_mis]
        for i in 1:n_obs
            y[ii_obs[i]] = y_obs[i]
        end
        for i in 1:n_mis
            y[ii_mis[i]] = y_mis[i]
        end
        y
    end
end

# --- RaggedVector usertype + Julia-dispatched accessors ---------------------
# A ragged vector is conceptually `Vector{<:AbstractVector{<:Real}}` — a
# variable-length collection of real subvectors. SLIC encodes one as a
# tagged ntup with fields:
#   mem  :: vector[total]   — concatenation of all subvectors
#   ends :: int[n_groups]   — inclusive 1-based end index of each subvector
# Standard Julia dispatch on the `RaggedVector` tag drives `length(rv)`
# / `rv[i]` so users don't need bespoke `ragged_*` accessors.
@usertype struct RaggedVector
    mem  :: vector
    ends :: int[]
end
# Built-in usertypes get aliased into the `builtin` submodule so SLIC's
# `_is_builtin_name` lookup picks them up at any user `@slic` site —
# same convention used for `vector`/`real`/etc. above.
@eval builtin const RaggedVector = $RaggedVector

@deffun begin
    Base.length(rv::RaggedVector)::int = size(rv.ends)
    Base.lastindex(rv::RaggedVector)::int = size(rv.ends)
    # The slice's runtime length is named *exactly* via `ragged_length`,
    # so the return-type annotation is `vector[ragged_length(rv, i)]`
    # (matching Stan's call-site declarations). Legacy `ragged_start` /
    # `ragged_end` / `ragged_length` are `::ntup`-typed and dispatch
    # on `RaggedVector <: ntup` automatically.
    Base.getindex(rv::RaggedVector, i::int)::vector[ragged_length(rv, i)] =
        rv.mem[ragged_start(rv, i):ragged_end(rv, i)]
end

# --- Sized-token rng overloads (generated via @eval @deffun) -----------------
# gq `x::T[n] ~ dist(args...)` synthesizes `dist_rng(T[n], args...)` which
# dispatches here on the token's shape. Stan's native `*_rng` broadcasts scalars
# against vector args natively, so we only need `rep_vector` in the all-scalar
# case (to give Stan a shape to work with). Our convention: output is always
# `vector[n]` / `int[n]` (not Stan's native `real[n]`), so the vector-token
# path wraps continuous rngs in `to_vector`.

# std_normal token (0-arg)
@deffun std_normal_rng(real[n])::real[n] = normal_rng(rep_vector(0, n), 1)
@deffun std_normal_rng(vector[n])::vector[n] = to_vector(normal_rng(rep_vector(0, n), 1))

# 2-arg continuous families. Two semantic cases:
#   - all scalar: need `rep_vector` so Stan produces a shaped output
#   - catch-all (at least one shape-[n] container): Stan native broadcasts
# Julia dispatch picks the more-specific `(a::real, b::real)` when both scalar.
for dist in (:normal, :cauchy, :lognormal, :gamma, :inv_gamma, :beta, :uniform,
             :weibull, :frechet, :double_exponential, :logistic, :gumbel,
             :pareto, :scaled_inv_chi_square, :von_mises, :loglogistic)
    drng = Symbol(dist, :_rng)
    @eval @deffun $drng(real[n],   a::real, b::real)::real[n]   = $drng(rep_vector(a, n), b)
    @eval @deffun $drng(vector[n], a::real, b::real)::vector[n] = to_vector($drng(rep_vector(a, n), b))
    @eval @deffun $drng(real[n],   a, b)::real[n]   = $drng(a, b)
    @eval @deffun $drng(vector[n], a, b)::vector[n] = to_vector($drng(a, b))
end

# 1-arg continuous families
for dist in (:exponential, :chi_square, :inv_chi_square, :rayleigh)
    drng = Symbol(dist, :_rng)
    @eval @deffun $drng(real[n],   a::real)::real[n]   = $drng(rep_vector(a, n))
    @eval @deffun $drng(vector[n], a::real)::vector[n] = to_vector($drng(rep_vector(a, n)))
    @eval @deffun $drng(real[n],   a)::real[n]   = $drng(a)
    @eval @deffun $drng(vector[n], a)::vector[n] = to_vector($drng(a))
end

# 3-arg continuous (leading arg — nu / alpha — is always scalar in practice)
for dist in (:student_t, :skew_normal, :exp_mod_normal, :pareto_type_2)
    drng = Symbol(dist, :_rng)
    @eval @deffun $drng(real[n],   nu::real, a::real, b::real)::real[n]   = $drng(nu, rep_vector(a, n), b)
    @eval @deffun $drng(vector[n], nu::real, a::real, b::real)::vector[n] = to_vector($drng(nu, rep_vector(a, n), b))
    @eval @deffun $drng(real[n],   nu::real, a, b)::real[n]   = $drng(nu, a, b)
    @eval @deffun $drng(vector[n], nu::real, a, b)::vector[n] = to_vector($drng(nu, a, b))
end

# 1-arg discrete families (output is int[n]; no to_vector wrap)
for dist in (:bernoulli, :bernoulli_logit, :poisson, :poisson_log)
    drng = Symbol(dist, :_rng)
    @eval @deffun $drng(int[n], p::real)::int[n] = $drng(rep_vector(p, n))
    @eval @deffun $drng(int[n], p)::int[n]       = $drng(p)
end

# 2-arg discrete families
for dist in (:neg_binomial, :neg_binomial_2)
    drng = Symbol(dist, :_rng)
    @eval @deffun $drng(int[n], a::real, b::real)::int[n] = $drng(rep_vector(a, n), b)
    @eval @deffun $drng(int[n], a, b)::int[n]             = $drng(a, b)
end

# binomial: N::int[n] is already a container, so Stan broadcasts scalar p natively
@deffun binomial_rng(int[n], N::int[n], p)::int[n] = binomial_rng(N, p)

# binomial_logit: Stan ships `binomial_logit_lpmf` but NOT a matching
# `binomial_logit_rng` (only the GLM-flavoured variant exists). Lower
# the token-path call to `binomial_rng(N, inv_logit(eta))` so SBBRMI's
# generated_quantities for a `BinomialLogit(N, eta)` likelihood compile
# under stanc.
@deffun binomial_logit_rng(int[n], N::int[n], eta)::int[n] = binomial_rng(N, inv_logit(eta))

# beta_binomial: (trials, alpha, beta); trials always int[n]
@deffun beta_binomial_rng(int[n], N::int[n], a::real, b::real)::int[n] = beta_binomial_rng(N, a, b)
@deffun beta_binomial_rng(int[n], N::int[n], a, b)::int[n] = beta_binomial_rng(N, a, b)

# dirichlet: native already returns vector[n]; token path just unwraps
@deffun dirichlet_rng(vector[n], alpha::vector[n])::vector[n] = dirichlet_rng(alpha)

# categorical / categorical_logit: per-row loop (same prob vector reused each row)
@deffun categorical_rng(int[n], p::vector[k])::int[n] = begin
    rv::int[n]
    for i in 1:n
        rv[i] = categorical_rng(p)
    end
    rv
end
@deffun categorical_logit_rng(int[n], eta::vector[k])::int[n] = begin
    rv::int[n]
    for i in 1:n
        rv[i] = categorical_logit_rng(eta)
    end
    rv
end

# multi_normal / multi_normal_cholesky: native already returns vector[n]
@deffun multi_normal_rng(vector[n], loc::vector[n], cov)::vector[n]          = multi_normal_rng(loc, cov)
@deffun multi_normal_cholesky_rng(vector[n], loc::vector[n], scale)::vector[n] = multi_normal_cholesky_rng(loc, scale)

@defsig begin
    Union{typeof.((sqrt, exp, log, log10, sin, cos, asin, acos, tan, atan,
        cosh, sinh, tanh, acosh, asinh, atanh,
        log1m, inv_logit, logit, log_inv_logit, log1m_exp, expm1, Phi, lgamma, abs,
        log1p_exp, log1m_exp, Base.inv, Base.log1p,
        exp2, log2, cbrt, ceil, floor, round, trunc,
        square, erf, erfc, tgamma, digamma, trigamma,
        Phi_approx, inv_Phi))...} => begin
        (real,)=>real
        (vector[n],)=>vector[n]
        (row_vector[n],)=>row_vector[n]
        (real[n],)=>real[n]
        (matrix[m,n],)=>matrix[m,n]
    end
    Union{typeof.((log_sum_exp, ))...} => begin
        (real, real) => real
        (vector[n], vector[n]) => vector[n]
    end
    # Matrix reductions returning real
    Union{typeof.((trace, determinant, log_determinant, log_determinant_spd))...} => begin
        (matrix[m,n],) => real
    end
    # Square-matrix-in, square-matrix-out
    Union{typeof.((inverse, inverse_spd, chol2inv, symmetrize_from_lower_tri))...} => begin
        (matrix[n,n],) => matrix[n,n]
    end
    typeof(generalized_inverse) => begin
        (matrix[m,n],) => matrix[n,m]
    end
    typeof(eigenvalues_sym) => begin
        (matrix[n,n],) => vector[n]
    end
    typeof(eigenvectors_sym) => begin
        (matrix[n,n],) => matrix[n,n]
    end
    typeof(crossprod) => begin
        (matrix[m,n],) => matrix[n,n]
    end
    typeof(tcrossprod) => begin
        (matrix[m,n],) => matrix[m,m]
    end
    typeof(multiply_lower_tri_self_transpose) => begin
        (matrix[m,n],) => matrix[m,m]
    end
    typeof(matrix_power) => begin
        (matrix[n,n], int[]) => matrix[n,n]
    end
    typeof(matrix_exp_multiply) => begin
        (matrix[m,m], matrix[m,n]) => matrix[m,n]
    end
    typeof(scale_matrix_exp_multiply) => begin
        (real[], matrix[m,m], matrix[m,n]) => matrix[m,n]
    end
    typeof(mdivide_left_spd) => begin
        (matrix[m,m], vector[m]) => vector[m]
        (matrix[m,m], matrix[m,n]) => matrix[m,n]
    end
    typeof(mdivide_right_spd) => begin
        (row_vector[m], matrix[m,m]) => row_vector[m]
        (matrix[m,n], matrix[n,n]) => matrix[m,n]
    end
    Union{typeof.((dot_self,))...} => begin
        (vector[n],) => real
        (row_vector[n],) => real
    end
    typeof(columns_dot_product) => begin
        (matrix[m,n], matrix[m,n]) => row_vector[n]
    end
    typeof(columns_dot_self) => begin
        (matrix[m,n],) => row_vector[n]
    end
    typeof(rows_dot_self) => begin
        (matrix[m,n],) => vector[m]
    end
    Union{typeof.((sort_asc, sort_desc))...} => begin
        (int[n],)=>int[n]
        (real[n],)=>real[n]
        (vector[n],)=>vector[n]
        (row_vector[n],)=>row_vector[n]
    end
    Union{typeof.((sort_indices_asc, sort_indices_desc))...} => begin
        (anything[n],)=>int[n]
        (vector[n],)=>int[n]
        (row_vector[n],)=>int[n]
    end
    # Quad forms: B'*A*B
    typeof(quad_form) => begin
        (matrix[m,m], vector[m]) => real
        (matrix[m,m], matrix[m,n]) => matrix[n,n]
    end
    typeof(quad_form_diag) => begin
        (matrix[m,m], vector[m]) => matrix[m,m]
        (matrix[m,m], row_vector[m]) => matrix[m,m]
    end
    typeof(quad_form_sym) => begin
        (matrix[m,m], vector[m]) => real
        (matrix[m,m], matrix[m,n]) => matrix[n,n]
    end
    Union{typeof.((trace_quad_form,))...} => begin
        (matrix[m,m], matrix[m,n]) => real
        (matrix[m,m], vector[m]) => real
    end
    typeof(trace_gen_quad_form) => begin
        (matrix[n,n], matrix[m,m], matrix[m,n]) => real
    end
    # diagonal
    typeof(diagonal) => begin
        (matrix[n,n],) => vector[n]
    end
    # row/col
    typeof(col) => begin
        (matrix[m,n], int[]) => vector[m]
    end
    typeof(row) => begin
        (matrix[m,n], int[]) => row_vector[n]
    end
    # norm / distance
    Union{typeof.((norm1, norm2))...} => begin
        (vector[n],) => real
        (row_vector[n],) => real
        (real[n],) => real
    end
    Union{typeof.((distance, squared_distance))...} => begin
        (vector[n], vector[n]) => real
        (row_vector[n], row_vector[n]) => real
    end
    # cumulative_sum for row_vector (already has vector, add row_vector)
    typeof(cumulative_sum) => begin
        (int[m],)=>int[m]
        (real[m],)=>real[m]
        (vector[m],)=>vector[m]
        (row_vector[m],)=>row_vector[m]
    end
    typeof(÷) => begin 
        (int, int) => int
    end
    Union{typeof.((+, -, ^, *, /))...} => begin 
        (real,) => real
        (vector[n],) => vector[n]
        (int, real) => real
        (int, int) => int
        (real, int) => real
        (real, real) => real
        (real[n], real[n]) => real[n]
        (int[n], int[n]) => int[n]
        (int[n], int) => int[n]
        (int, vector[n]) => vector[n]
        (int, int[n]) => int[n]
        (real, real[n]) => real[n]
        (real, vector[n]) => vector[n]
        (real, matrix[m,n]) => matrix[m,n]
        (vector[n], real) => vector[n]
        (vector[n], int) => vector[n]
        (vector[n], vector[m]) => vector[n]
        (row_vector[n], real) => row_vector[n]
        (row_vector[n], int) => row_vector[n]
        (matrix[m,n], real) => matrix[m,n]
    end
    Union{typeof.((+, -))...} => begin 
        (matrix[m,n],matrix[m,n]) => matrix[m,n]
    end
    Union{typeof.((*, ))...} => begin
        (vector[m], row_vector[n]) => matrix[m,n]
        (row_vector[n], vector[n]) => real
        (matrix[m,n], vector[n]) => vector[m]
        (matrix[m,n], matrix[n,o]) => matrix[m,o]
        (cholesky_factor_corr[m],matrix[m,n]) => matrix[m,n]
    end
    typeof(adjoint) => begin
        (vector[n],) => row_vector[n]
        (row_vector[n],) => vector[n]
        (matrix[m,n],) => matrix[n,m]
        (cholesky_factor_corr[m],) => matrix[m,m]
    end
    typeof(transpose) => begin
        (vector[n],) => row_vector[n]
        (row_vector[n],) => vector[n]
        (matrix[m,n],) => matrix[n,m]
        (cholesky_factor_corr[m],) => matrix[m,m]
    end
    typeof(getindex) => begin 
        (int[m], int) => int
        (int[m], int[n]) => int[n]
        (int[m,n], int) => int[n] 
        (int[m,n], int[o], int) => int[o] 
        (int[m,n], int, int) => int 
        (real[m], int) => real
        (real[m], int[n]) => real[n]
        (real[m,n], int) => real[n] 
        (real[m,n], int[o], int) => real[o] 
        (vector[m], int[n]) => vector[n]
        (any_vector[m], int) => real
        (vector[m,n], int) => vector[n]
        (any_vector[m,n], int, int) => real
        (vector[m,n], int[o], int) => real[o]
        (vector[m,n], int, int[o]) => vector[o]
        (vector[m,n], int[p], int[q]) => vector[p, q]
        (matrix[m,n], int, int) => real
        (matrix[m,n], int[o], int) => vector[o]
        (matrix[m,n], int, int[p]) => row_vector[p]
        (matrix[m,n], int[o], int[p]) => matrix[o, p]
        (matrix[m,n,k], int, int, int) => real
        (matrix[m,n,k], int, int[o], int) => vector[o]
        (matrix[m,n,k], int, int, int[p]) => row_vector[p]
        (matrix[m,n,k], int, int[o], int[p]) => matrix[o,p]
        (cholesky_factor_corr[m,n], int, int, int) => real
        (cholesky_factor_corr[m,n], int, int[o], int) => vector[o]
        (cholesky_factor_corr[m,n], int, int, int[p]) => row_vector[p]
        (cholesky_factor_corr[m,n], int, int[o], int[p]) => cholesky_factor_corr[o]
    end
    typeof(std_normal_rng) => begin 
        () => real
    end
    Union{typeof.((normal_rng, cauchy_rng))...} => begin 
        (int, int) => real
        (int, real) => real
        (real, int) => real
        (real, real) => real
        (int, int[n]) => real[n]
        (int, vector[n]) => real[n]
        (int, row_vector[n]) => real[n]
        (real, int[n]) => real[n]
        (real, vector[n]) => real[n]
        (real, row_vector[n]) => real[n]
        (int[n], real) => real[n]
        (vector[n], real) => real[n]
        (row_vector[n], real) => real[n]
        (vector[n], vector[n]) => real[n]
    end
    typeof(exponential_rng) => begin
        (real,)=>real
        (vector[n],)=>real[n]
    end
    Union{typeof.((lognormal_rng, gamma_rng, inv_gamma_rng, beta_rng, uniform_rng,
                    weibull_rng, frechet_rng, loglogistic_rng, von_mises_rng,
                    double_exponential_rng, logistic_rng, gumbel_rng,
                    pareto_rng, scaled_inv_chi_square_rng, neg_binomial_2_rng))...} => begin
        (real, real) => real
        (real, vector[n]) => real[n]
        (real, row_vector[n]) => real[n]
        (vector[n], real) => real[n]
        (vector[n], vector[n]) => real[n]
    end
    Union{typeof.((chi_square_rng, inv_chi_square_rng, rayleigh_rng))...} => begin
        (real,) => real
        (vector[n],) => real[n]
    end
    Union{typeof.((student_t_rng, skew_normal_rng, exp_mod_normal_rng, pareto_type_2_rng))...} => begin
        (real, real, real) => real
        (real, vector[n], real) => real[n]
        (real, vector[n], vector[n]) => real[n]
        (vector[n], vector[n], real) => real[n]
        (vector[n], vector[n], vector[n]) => real[n]
    end
    typeof(beta_binomial_rng) => begin
        (int[n], real, real) => int[n]
        (int[n], vector[n], real) => int[n]
        (int[n], real, vector[n]) => int[n]
        (int[n], vector[n], vector[n]) => int[n]
    end
    Base.BroadcastFunction => begin 
        (real, real) => real
        (real[n], real[n]) => real[n]
        (real[n], real) => real[n]
        (real, real[n]) => real[n]
        (int[n], int) => int[n]
        (int[n], int[n]) => int[n]
        (vector[n], real) => vector[n]
        (real, vector[n]) => vector[n]
        (vector[n], vector[n]) => vector[n]
        (row_vector[n], row_vector[n]) => row_vector[n]
    end
    typeof(cholesky_decompose) => begin
        (matrix[n,n],) => matrix[n,n]
    end
    typeof(mdivide_right_tri_low) => begin
        (row_vector[n], matrix[n,n]) => row_vector[n] 
    end
    typeof(diag_pre_multiply) => begin
        (vector[m], matrix[m,n]) => matrix[m,n]
        (vector[m], cholesky_factor_corr[m]) => matrix[m,m] 
    end
    typeof(diag_post_multiply) => begin
        (matrix[m,n], vector[n]) => matrix[m,n] 
        (matrix[m,n], row_vector[n]) => matrix[m,n] 
    end
    typeof(gp_exp_quad_cov) => begin 
        (real[n], real, real) => matrix[n,n]
        (real[m], real[n], real, real) => matrix[m,n]
    end
    typeof(size) => begin
        (vector[n],)=>int
        (real[n],)=>int
    end
    typeof(log_sum_exp) => begin 
        (real[n], ) => real
        (matrix[m,n], ) => real
        (row_vector[n], ) => real
        (vector[n], ) => real
    end
    Union{typeof.((|, &, ==, !=, <, <=, >, >=))...} => begin
        (anything, anything) => int
    end
    # step: Base.step is exported; Stan's step(real)->int dispatches on it
    typeof(step) => begin
        (real,) => int
        (int,) => int
    end
    # Julia functions with different Stan names (func_name overrides in functions.jl):
    #   length → num_elements,  minimum/maximum → min/max,
    #   LinearAlgebra.dot → dot_product,  abs2 → square
    typeof(length) => begin
        (vector[n],) => int
        (row_vector[n],) => int
        (matrix[m,n],) => int
        (real[n],) => int
        (int[n],) => int
    end
    Union{typeof.((minimum, maximum))...} => begin
        (real[n],) => real
        (int[n],) => int
        (vector[n],) => real
        (row_vector[n],) => real
    end
    typeof(abs2) => begin
        (real,) => real
        (int,) => real
        (vector[n],) => vector[n]
        (row_vector[n],) => row_vector[n]
        (matrix[m,n],) => matrix[m,n]
    end
    # prod/variance: Base.prod and Statistics.var are exported; must use @defsig not @deffun
    typeof(prod) => begin
        (real[n],) => real
        (int[n],) => real
        (vector[n],) => real
        (row_vector[n],) => real
        (matrix[m,n],) => real
    end
    typeof(variance) => begin
        (real[n],) => real
        (vector[n],) => real
        (row_vector[n],) => real
        (matrix[m,n],) => real
    end
    # quantile: Base function
    typeof(quantile) => begin
        (real[n], real) => real
        (vector[n], real) => real
        (row_vector[n], real) => real
    end
    # add_diag with vector/row_vector diagonal
    typeof(add_diag) => begin
        (matrix[n,n], real) => matrix[n,n]
        (matrix[n,n], vector[n]) => matrix[n,n]
        (matrix[n,n], row_vector[n]) => matrix[n,n]
    end
    # to_vector for row_vector and int array
    typeof(to_vector) => begin
        (vector[n],) => vector[n]
        (row_vector[n],) => vector[n]
        (real[n],) => vector[n]
        (int[n],) => vector[n]
        (matrix[m,n],) => vector[m*n]
    end
    typeof(to_row_vector) => begin
        (vector[n],) => row_vector[n]
        (row_vector[n],) => row_vector[n]
        (real[n],) => row_vector[n]
        (int[n],) => row_vector[n]
        (matrix[m,n],) => row_vector[m*n]
    end
    typeof(to_array_1d) => begin
        (vector[n],) => real[n]
        (row_vector[n],) => real[n]
        (int[n],) => int[n]
        (real[m,n],) => real[m*n]
        (int[m,n],) => int[m*n]
    end
    # min/max for more types
    Union{typeof.((min, max))...} => begin
        (int, int) => int
        (real, real) => real
        (int, real) => real
        (real, int) => real
        (int[n],) => int
        (real[n],) => real
        (vector[n],) => real
        (row_vector[n],) => real
        (matrix[m,n],) => real
    end
end

const TolODESolver = Union{typeof.((ode_rk45_tol, ode_ckrk_tol, ode_adams_tol, ode_bdf_tol))...}
const NoTolODESolver = Union{typeof.((ode_rk45, ode_ckrk, ode_adams, ode_bdf))...}
const ODESolver = Union{TolODESolver, NoTolODESolver}
const ReduceSumFunction = Union{typeof.((reduce_sum, reduce_sum_static))...}

tracetype(x::CanonicalExpr{<:ODESolver}) = StanType(
    types.vector, (stan_size(x.args[4], 1), stan_size(x.args[2], 1))
)

fetch_functions!(x::CanonicalExpr{<:TolODESolver}; info) = fetch_functions!(
    CanonicalExpr(x.args[1], x.args[3], x.args[2], x.args[8:end]..., _closure_captures(x.args[1])...); info
)

fetch_functions!(x::CanonicalExpr{<:NoTolODESolver}; info) = fetch_functions!(
    CanonicalExpr(x.args[1], x.args[3], x.args[2], x.args[5:end]..., _closure_captures(x.args[1])...); info
)

function reduce_sum_reconstruct end
function reduce_sum_deconstruct end
_is_tup_arg(arg::StanExpr2{<:types.tup}) = true
_is_tup_arg(_) = false
fetch_functions!(x::CanonicalExpr{<:ReduceSumFunction}; info) = begin
    fetch_functions!(
        CanonicalExpr(x.args[1], x.args[2], stan_expr(1), stan_expr(1), x.args[4:end]...); info
    )
    if any(_is_tup_arg, x.args) 
        fetch_functions!(
            CanonicalExpr(reduce_sum_reconstruct, x.args[1], x.args[2], stan_expr(1), stan_expr(1), x.args[4:end]...); info
        )
        # Work around https://github.com/stan-dev/math/issues/3041
        fetch_functions!(
            CanonicalExpr(reduce_sum_deconstruct, stan_expr(reduce_sum_reconstruct), x.args[2], x.args[3], x.args[1], x.args[4:end]...); info
        )
    end
end

reduce_sum_args!(x::StanExpr2{<:types.tup}; d) = StanExpr(CanonicalExpr(
    :tuple, [
        reduce_sum_args!(StanExpr(:_, arg_type); d)
        for arg_type in x.type.info.arg_types
    ]...
), type(x))
reduce_sum_args!(x::StanExpr; d) = begin
    name = Symbol("arg", 1+length(d))
    push!(d, name=>anon_expr(name, x))
    d[end][2]
end
reduce_sum_args!(x::Tuple; d) = reduce_sum_args!.(x; d)
fundef(x::CanonicalExpr{typeof(reduce_sum_reconstruct)}) = if any(_is_tup_arg, x.args) 
    deconstructed_args = []
    reconstructed_args = reduce_sum_args!(x.args; d=deconstructed_args)
    StanFunction3(
        "// Work around https://github.com/stan-dev/math/issues/3041\n", 
        StanType(types.real),
        reduce_sum_reconstruct,
        (;deconstructed_args...),
        [
            CanonicalExpr(:return, stan_call(reconstructed_args...))
        ]
    )
end

reduce_sum_args2!(x::StanExpr2{<:types.tup}; d) = for (i, arg_type) in enumerate(x.type.info.arg_types)
    reduce_sum_args2!(StanExpr(Symbol(expr(x), ".", i), arg_type); d)
end
reduce_sum_args2!(x::StanExpr; d) = push!(d, x)
fundef(x::CanonicalExpr{typeof(reduce_sum_deconstruct)}) = begin 
    deconstructed_args = []
    original_args = [
        Symbol("arg", i)=>anon_expr(Symbol("arg", i), x.args[i]) for i in eachindex(x.args)
    ]
    reduce_sum_args2!.(last.(original_args); d=deconstructed_args)
    StanFunction3(
        "// Work around https://github.com/stan-dev/math/issues/3041\n", 
        StanType(types.real),
        reduce_sum_deconstruct,
        (;original_args...),
        [
            CanonicalExpr(:return, stan_call(reduce_sum, deconstructed_args...))
        ]
    )
end
Base.show(io::IO, x::CanonicalExpr{<:ReduceSumFunction}) = if any(_is_tup_arg, x.args)
    # Work around https://github.com/stan-dev/math/issues/3041
    print(io, CanonicalExpr(reduce_sum_deconstruct, stan_expr(reduce_sum_reconstruct), x.args[2], x.args[3], x.args[1], x.args[4:end]...))
else
    # Phase-2 closure lifting: closures in args (typically the trailing
    # `f, args...` forwarded to the helper) expand to their capture values
    # so Stan's `reduce_sum` receives them as the trailing s1,s2,... args.
    autoprint(io, head(x), "(", Join(
        (func_name(x.args[1], x.args[2:end]), stan_call_args(x.args[2:end])...), ", "
    ), ")")
end