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
    # A higher-order distribution can provide both `_lpdf` and `_lpmf`
    # wrappers under one base name. Their generated base/RNG/CDF companion
    # names overlap, so keep the manifest declarative and deduplicate here.
    names = unique(builtin_module_names(x))
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
    weighted_lpdf
    weighted_lpmf
    # Public distribution-HOF tokens match Julia's distribution vocabulary;
    # collision-free conditioning/clamping names remain implementation details.
    truncated censored interval_censored restricted
    lower_conditioning_lpdf lower_conditioning_lpmf
    upper_conditioning_lpdf upper_conditioning_lpmf
    conditioning_lpdf conditioning_lpmf
    lower_clamping_lpdf lower_clamping_lpmf
    upper_clamping_lpdf upper_clamping_lpmf
    clamping_lpdf clamping_lpmf
    interval_evidence_impl_lpdf interval_evidence_impl_lpmf
    lower_conditioning_cell_rng upper_conditioning_cell_rng conditioning_cell_rng
    conditioning_outside
    lower_conditioning_inverse_rng upper_conditioning_inverse_rng conditioning_inverse_rng
    conditioning_edge conditioning_ccdf_tail conditioning_cdf_tail conditioning_midpoint
    conditioning_bisect conditioning_bracket_up conditioning_bracket_down
    neg_binomial_2_lccdf_precise neg_binomial_lccdf_precise binomial_lccdf_precise
    ordered_rejection_rng ordered_rejection_unsorted
    lower_clamping_cell_rng upper_clamping_cell_rng clamping_cell_rng
    jbroadcasted_rng
    # Sequential-marginalization families (forward filter): the general
    # `sequential_marginalize` HOF and the matrix-parameterized `kalman` instance.
    sequential_marginalize_lpdf sequential_marginalize_lpdfs sequential_marginalize_rng
    kalman_lpdf kalman_lpdfs kalman_rng
    hmm_forward_lpdf hmm_forward_lpdfs hmm_forward_rng
    hmm_forward_lpmf
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
    vector_exponential_rng
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
    to_int
    csr_matrix_times_vector
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
    # Stan's embedded Laplace approximation (Stan >= 2.37), lowered from
    # `z ~ laplace_marginal(prior) do z … end` (laplace.jl).
    laplace_marginal laplace_marginal_tol laplace_latent_rng laplace_latent_tol_rng
    # Torsten (metrumresearchgroup/Torsten) PK event-schedule solvers.
    pmx_solve_onecpt pmx_solve_twocpt
    pmx_solve_onecpt_effcpt pmx_solve_twocpt_effcpt pmx_solve_linode
    pmx_solve_rk45 pmx_solve_bdf pmx_solve_adams
    pmx_solve_onecpt_rk45 pmx_solve_onecpt_bdf pmx_solve_twocpt_rk45 pmx_solve_twocpt_bdf
    pmx_append_each
    append_array
    append_row
    append_col
    hcat
    reshape
    ragged_n ragged_total ragged_start ragged_end ragged_length as_matrix
    ragged_common_ends ragged_matched_group
    ragged_select_length ragged_select_total ragged_select_ends ragged_select_index
    ragged_sum ragged_mean
    # Stan 2.37 exposed constraint transforms (Feature 1) — @deffun sigs below.
    sum_to_zero_constrain sum_to_zero_unconstrain sum_to_zero_jacobian
    simplex_constrain simplex_unconstrain simplex_jacobian
    ordered_constrain ordered_unconstrain ordered_jacobian
    positive_ordered_constrain positive_ordered_unconstrain positive_ordered_jacobian
    cholesky_factor_corr_jacobian cholesky_factor_cov_jacobian
    lower_bound_jacobian upper_bound_jacobian lower_upper_bound_jacobian offset_multiplier_jacobian
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
    positive_infinity negative_infinity not_a_number
    # Stan also exposes a zero-argument `log2()` constant; registering the
    # shared name here keeps both it and the unary overload on one callable.
    log2

    reduce_sum reduce_sum_static reduce_sum_reconstruct simple_reduce_sum simple_reduce_sum_helper

    broadcasted_getindex
    jbroadcasted jmap jsum

    # Compile-time distribution-family selectors. These consume a base family
    # token (e.g. `normal`) and disappear before Stan emission.
    density pointwise predictive logcdf logccdf
    # The truncated draws' inverse-CDF fallback (lpxf_builtin.jl): the most
    # precise upper-tail log CCDF a family has, and its implied support.
    conditioning_logccdf conditioning_support_lower conditioning_support_upper

    # Unary math (Stan-specific, not Julia builtins)
    square inv_sqrt log_diff_exp log_mix atan2
    is_inf is_nan
    Phi_approx inv_Phi inv_cloglog
    erf erfc tgamma digamma trigamma
    lambert_w0
    lbeta lchoose inc_beta gamma_p gamma_q
    bessel_first_kind bessel_second_kind
    modified_bessel_first_kind modified_bessel_second_kind
    # Log-space first-kind modified Bessel (Stan >= 2.24; real order `v`).
    # The form a periodic Hilbert-space GP basis needs so a small length
    # scale cannot overflow `exp(1/rho^2)`. Shapes are in the `@defsig`
    # block below, beside `log_sum_exp`.
    log_modified_bessel_first_kind
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
    neg_binomial_lpmf neg_binomial_2_log_lpmf
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
    # that assembles the full vector in generated quantities when prior-only,
    # or transformed parameters when the completed vector feeds a likelihood.
    maybe_index
    merge_missing

    # erfc-stable normal log-CDF/CCDF companions (`@deffun` bodies + full
    # rationale beside the native CDF companions below). Selected for the
    # `normal` family by `logcdf_expr`/`logccdf_expr` (lpxf_builtin.jl) so
    # censored/truncated/interval-censored normal tail masses autodiff exactly.
    # Must be registered here too: `@deffun` builtins are asserted against this
    # manifest (`_assert_builtin_registered`). snag censored-normal-410c4e35.
    # `std_normal_lcdf_tail` is their shared deep-tail branch past 35σ (snag
    # fit-a-left-censo-942540f9).
    normal_lcdf_stable
    normal_lccdf_stable
    std_normal_lcdf_tail

    # Longitudinal dose/time-response building blocks. The obs model is a
    # censored normal w/ limits of quantification (`truncated_normal` — the
    # name says truncated, but the semantics are censoring);
    # `truncated_normal_lpdf` auto-expands to the
    # truncated_normal / _lpdfs / _rng / _cdf / _lccdf / _lcdf family. The mean
    # kernels (`biomarker_time_response` single-peak bump, `biomarker_dose_response`
    # log-sigmoid) + index/broadcast helpers are composed downstream (e.g. by a
    # BRM grouped term: kernels here, the consumer composes log_y).
    # `truncated_student_t_lpdf` = the heavy-tailed obs variant (same censoring
    # contract, + a leading `dof` arg, branches on the LOQ limits).
    truncated_normal_lpdf
    truncated_student_t_lpdf
    biomarker_time_response
    biomarker_dose_response
    linear_idxs
    broadcasted_max
    broadcasted_gt
]

# GLM distributions — defined outside @builtin_module to avoid Revise conflicts
for glm in (:normal_id_glm_lpdf, :poisson_log_glm_lpmf, :neg_binomial_2_log_glm_lpmf)
    for name in builtin_module_names(glm)
        @eval builtin function $name end
        @eval const $name = builtin.$name
    end
end

# Stan's `head`, `tail` and `reverse` share their names with StanBlocks' own
# `head` (a call node's function) and Base's `reverse`, so they cannot have the
# package-level alias the manifest adds (`const head = builtin.head`): it would
# replace those functions inside this package. They are defined in `builtin`
# alone, where a body's names resolve (`_is_builtin_name`), and this package
# spells them `builtin.head`, `builtin.tail` and `builtin.reverse`.
for name in (:head, :tail, :reverse)
    @eval builtin function $name end
end
# `head(v, n)` / `tail(v, n)` are the first / last `n` elements of a vector, row
# vector or array, and `reverse(v)` all of them in reverse order. Stan's array
# overloads are generic in the element type (`array[] T head(array[] T, int)`),
# so the result is typed like `getindex`'s peel (`_peeled_type`, functions.jl):
# the argument's type with its leading size replaced, at any array depth. Taken
# from a vector itself, the elements lose a `simplex`/`ordered` constraint and
# the result is a plain `vector` (`row_vector`). A matrix or a scalar has no
# such overload and stays untyped.
_sequence_core(::Type{<:types.vector}) = types.vector
_sequence_core(::Type{<:types.row_vector}) = types.row_vector
_sequence_core(_) = nothing
_sequence_type(t::StanType, sizes) = if l_ndim(t) > 0
    StanType(center_type(t), sizes, info(t); value=missing)
else
    core = _sequence_core(center_type(t))
    isnothing(core) ? StanType(types.anything) : StanType(core, sizes)
end
tracetype(x::CanonicalExpr{<:Union{typeof(builtin.head),typeof(builtin.tail)},<:Tuple{<:StanExpr,<:StanExpr2{<:types.int,0}}}) =
    _sequence_type(type(x.args[1]), (x.args[2], stan_size(type(x.args[1]))[2:end]...))
tracetype(x::CanonicalExpr{typeof(builtin.reverse),<:Tuple{<:StanExpr}}) =
    _sequence_type(type(x.args[1]), stan_size(type(x.args[1])))

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
    @stanonly simple_reduce_sum(f, x, args...)::real = reduce_sum(simple_reduce_sum_helper, x, 1, f, args...)
    # Stan's `reduce_sum` only accepts `T[]` arrays as its data argument, not
    # `vector` (or its `simplex`/`ordered`/`positive_ordered` subtypes, or
    # `row_vector`). Auto-convert via `to_array_1d` so users can call
    # `simple_reduce_sum(f, ::any_vector, ...)` directly. `any_vector` covers
    # all 1-d vector-like Stan types in the SLIC hierarchy.
    @stanonly simple_reduce_sum(f, x::any_vector[n], args...)::real = reduce_sum(simple_reduce_sum_helper, to_array_1d(x), 1, f, args...)
    @juliacompat simple_reduce_sum_helper(x_slice::anything[n], slice_start, slice_end, f, args...)::real = begin
        rv = 0.
        for i in 1:n
            rv += f(x_slice[i], args...)
        end
        rv
    end
    positive_infinity()::real
    negative_infinity()::real
    not_a_number()::real
    log2()::real
    reject(args...)::anything
    # --- Stan 2.37 exposed constraint-transform functions (Feature 1: ragged
    # non-trivial constrained parameters). Bodyless — these are Stan built-ins
    # (2.37+, wrapped by BridgeStan 2.7.0); SB only needs their signatures to
    # emit `<family>_jacobian(free_slice)` calls per ragged slice. `_jacobian`
    # constrains AND increments the target Jacobian implicitly (no `jacobian +=`
    # block needed). simplex: free vector[n] (=N-1) <-> simplex vector[n+1] (=N).
    # sum_to_zero is linear (isometric), so its `_jacobian` adds nothing; the
    # matrix form has zero row and column sums, one larger in each dimension.
    # The `@juliacompat` methods are `jcall` shims (functions.jl).
    sum_to_zero_constrain(y::vector[n])::vector[n+1]
    sum_to_zero_jacobian(y::vector[n])::vector[n+1]
    sum_to_zero_unconstrain(x::vector[n])::vector[n-1]
    sum_to_zero_constrain(y::matrix[m,n])::matrix[m+1,n+1]
    sum_to_zero_jacobian(y::matrix[m,n])::matrix[m+1,n+1]
    sum_to_zero_unconstrain(x::matrix[m,n])::matrix[m-1,n-1]
    simplex_constrain(y::vector[n])::vector[n+1]
    simplex_jacobian(y::vector[n])::vector[n+1]
    simplex_unconstrain(x::vector[n])::vector[n-1]
    ordered_constrain(y::vector[n])::vector[n]
    ordered_jacobian(y::vector[n])::vector[n]
    ordered_unconstrain(x::vector[n])::vector[n]
    positive_ordered_constrain(y::vector[n])::vector[n]
    positive_ordered_jacobian(y::vector[n])::vector[n]
    positive_ordered_unconstrain(x::vector[n])::vector[n]
    cholesky_factor_corr_jacobian(y::vector[n], K::int)::matrix[K,K]
    cholesky_factor_cov_jacobian(y::vector[n], M::int, N::int)::matrix[M,N]
    # Elementwise bound / affine transforms. A constrained `ordered` parameter is
    # one of these applied to an `ordered` free vector (`_ordered_transform`).
    lower_bound_jacobian(y::vector[n], lb::real)::vector[n]
    upper_bound_jacobian(y::vector[n], ub::real)::vector[n]
    lower_upper_bound_jacobian(y::vector[n], lb::real, ub::real)::vector[n]
    offset_multiplier_jacobian(y::vector[n], offset::real, multiplier::real)::vector[n]
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
    @stanonly robust_linspaced_int_array(n, args...)::int[n] = if n == 0
        rv::int[n]
        rv 
    else
        linspaced_int_array(n, args...)
    end
    linspaced_vector(n, x, y)::vector[n]
    to_matrix(v, m, n)::matrix[m,n]
    # Sparse CSR matrix-vector product: A*b for an m x n matrix A supplied in
    # compressed-row form (w = nonzero values, v = column indices, u = row start
    # pointers). Result size is the int arg `m` (size-from-int-arg, cf. range).
    csr_matrix_times_vector(m::int, n::int, w::vector[nw], v::int[nv], u::int[nu], b::vector[nb])::vector[m]
    rep_array(x::int, n)::int[n]
    rep_array(x::int, m, n)::int[m, n]
    rep_array(x::real, n)::real[n]
    rep_array(x::real, m, n)::real[m, n]
    rep_array(x::int[k], n)::int[n, k]
    rep_array(x::real[k], n)::real[n, k]
    rep_vector(v, n)::vector[n]
    rep_matrix(v::vector[m], n)::matrix[m, n]
    # A row vector repeats DOWN the rows: `m` copies of `rv` stacked.
    rep_matrix(rv::row_vector[n], m)::matrix[m, n]
    rep_matrix(x::real, m, n)::matrix[m,n]
    to_array_2d(v, m, n)::real[m,n]
    # Stan's native matrix → 2-D real array (row-major). Matrix data ingests as
    # `matrix`, so this is how data reaches an `array[,] real` slot such as
    # Torsten's per-event `x_r`.
    to_array_2d(x::matrix[m, n])::real[m, n]
    # Stan's dot product takes either orientation on either side, or two real
    # arrays (an int array promotes).
    dot_product(x::vector[n], y::vector[n])::real
    dot_product(x::vector[n], y::row_vector[n])::real
    dot_product(x::row_vector[n], y::vector[n])::real
    dot_product(x::row_vector[n], y::row_vector[n])::real
    dot_product(x::real[n], y::real[n])::real
    matrix_exp(x::matrix[m,m])::matrix[m,m]
    rows_dot_product(x::matrix[m,n], y::matrix[m,n])::vector[m]
    # A vector has n one-element rows, a row vector one n-element row.
    rows_dot_product(x::vector[n], y::vector[n])::vector[n]
    rows_dot_product(x::row_vector[n], y::row_vector[n])::vector[1]
    append_col(x::anything[n], y::anything[n])::matrix[n,2]
    append_col(x::matrix[m, n1], y::matrix[m, n2])::matrix[m, n1+n2]
    append_col(x::anything[m], y::matrix[m, n2])::matrix[m, 1+n2]
    append_col(x::matrix[m, n1], y::anything[m])::matrix[m, n1+1]
    # Two row_vectors concatenate COLUMN-wise into a longer row_vector (NOT a
    # matrix — that is `append_row`'s row-stacking). These forms are strictly
    # more specific than the `anything[n]` rule above, so they dispatch first.
    append_col(x::row_vector[m], y::row_vector[n])::row_vector[m+n]
    append_col(x::row_vector[n], y::real)::row_vector[n+1]
    append_col(x::real, y::row_vector[n])::row_vector[n+1]
    append_array(lhs::anything[m],rhs::anything[n])::real[m+n]
    append_array(lhs::int[m],rhs::int[n])::int[m+n]
    append_array(lhs::anything[m],rhs::real)::real[m+1]
    append_row(lhs::vector[m],rhs::real)::vector[m+1]
    append_row(lhs::vector[m],rhs::vector[n])::vector[m+n]
    append_row(::real, ::vector[n])::vector[n+1]
    append_row(::matrix[m1,n], ::matrix[m2,n])::matrix[m1+m2,n]
    append_row(x::row_vector[n], y::row_vector[n])::matrix[2, n]
    append_row(x::matrix[m, n], y::row_vector[n])::matrix[m+1, n]
    append_row(x::row_vector[n], y::matrix[m, n])::matrix[m+1, n]
    flat_lpdf(args...)
    # Observation weighting is an ordinary higher-order distribution family.
    # `family` is an always-inline base token, encoded into the emitted wrapper
    # name (`weighted_normal_lpdf`, `weighted_poisson_lpmf`, ...). The selector
    # calls below resolve through the same @lpxf registry as sampling/GQ.
    @stanonly begin
    weighted_lpdf(y, family, weight::real, args...) =
        weight * density(family, y, args...)
    weighted_lpmf(y, family, weight::real, args...) =
        weight * density(family, y, args...)
    # `lp` is deliberately UNannotated: `pointwise` carries its own size. A
    # builtin family sizes it by `y` (= `n`), but a custom `@lpxf` family whose
    # `_lpdfs`/`_lpmfs` twin is sized by one of ITS OWN arg tokens infers a
    # different-but-runtime-equal size (e.g. `vector[dims(args3)[1]]`). Forcing
    # `lp::vector[n]` here rejects that custom case at trace time; the
    # `weight .* lp` runtime dim-guard still enforces the equality.
    weighted_lpdf(y::anything[n], family, weight::vector[n], args...) = begin
        lp = pointwise(family, y, args...)
        sum(weight .* lp)
    end
    weighted_lpmf(y::anything[n], family, weight::vector[n], args...) = begin
        lp = pointwise(family, y, args...)
        sum(weight .* lp)
    end
    weighted_lpdfs(y, family, weight::real, args...) =
        weight * pointwise(family, y, args...)
    weighted_lpmfs(y, family, weight::real, args...) =
        weight * pointwise(family, y, args...)
    weighted_lpdfs(y::anything[n], family, weight::vector[n], args...)::vector[n] = begin
        lp = pointwise(family, y, args...)
        weight .* lp
    end
    weighted_lpmfs(y::anything[n], family, weight::vector[n], args...)::vector[n] = begin
        lp = pointwise(family, y, args...)
        weight .* lp
    end
    weighted_rng(family, weight, args...) =
        predictive(family, args...)
    # Sized forms forward the token to the base family's own companion, so they
    # inherit that family's guard — no guard here (audit todo 19n8abc).
    weighted_rng(real[n], family, weight, args...) =
        predictive(family, real[n], args...)
    weighted_rng(vector[n], family, weight, args...) =
        predictive(family, vector[n], args...)
    weighted_rng(int[n], family, weight, args...) =
        predictive(family, int[n], args...)

    # Mathematical truncation (`truncated`) ---------------------------------
    # Public optional kwargs select one of these lower/upper/two-sided
    # implementation families at trace time. Only concrete bounds reach Stan.
    lower_conditioning_lpdf(y::real, family, lo, args...)::real = begin
        rv::real[1]
        rv[1] = negative_infinity()
        if y >= lo
            rv[1] = density(family, y, args...) - logccdf(family, lo, args...)
        end
        rv[1]
    end
    lower_conditioning_lpmf(y::int, family, lo, args...)::real = begin
        rv::real[1]
        rv[1] = negative_infinity()
        if y >= lo
            rv[1] = density(family, y, args...) - logccdf(family, lo - 1, args...)
        end
        rv[1]
    end
    lower_conditioning_lpdf(y::anything[n], family, lo, args...)::real =
        sum(lower_conditioning_lpdfs(y, family, lo, args...))
    lower_conditioning_lpmf(y::anything[n], family, lo, args...)::real =
        sum(lower_conditioning_lpmfs(y, family, lo, args...))
    lower_conditioning_lpdfs(y::real, family, lo, args...) =
        lower_conditioning_lpdf(y, family, lo, args...)
    lower_conditioning_lpmfs(y::int, family, lo, args...) =
        lower_conditioning_lpmf(y, family, lo, args...)
    lower_conditioning_lpdfs(y::anything[n], family, lo, args...)::vector[n] =
        jbroadcasted(lower_conditioning_lpdf, y, family, lo, args...)
    lower_conditioning_lpmfs(y::anything[n], family, lo, args...)::vector[n] =
        jbroadcasted(lower_conditioning_lpmf, y, family, lo, args...)
    # A truncated draw redraws the base family until it lands in range, then,
    # after 100 misses, inverts the base CDF on the range (`*_inverse_rng`
    # below). The mixture is exact: a redraw that lands is a draw from the
    # truncated law, and so is the inversion.
    lower_conditioning_rng(family, lo, args...) = begin
        draw::typeof(predictive(family, args...))[1]
        attempts::int[1]
        draw[1] = predictive(family, args...)
        attempts[1] = 1
        while draw[1] < lo && attempts[1] < 100
            draw[1] = predictive(family, args...)
            attempts[1] = attempts[1] + 1
        end
        if draw[1] < lo
            draw[1] = lower_conditioning_inverse_rng(draw[1], family, lo, args...)
        end
        draw[1]
    end
    lower_conditioning_cell_rng(dummy, family, lo, args...) =
        lower_conditioning_rng(family, lo, args...)
    # `jbroadcasted` emits an explicit `for i in 1:n` loop of scalar cell draws,
    # so an empty token runs zero iterations — no guard, here or in the five
    # sibling HOF families below, which share this shape (todo 19n8abc).
    lower_conditioning_rng(vector[n], family, lo, args...)::vector[n] =
        jbroadcasted(lower_conditioning_cell_rng, rep_vector(0., n), family, lo, args...)
    lower_conditioning_rng(real[n], family, lo, args...)::real[n] =
        to_array_1d(jbroadcasted(lower_conditioning_cell_rng, rep_vector(0., n), family, lo, args...))
    lower_conditioning_rng(int[n], family, lo, args...) =
        jbroadcasted(lower_conditioning_cell_rng, rep_array(0, n), family, lo, args...)

    upper_conditioning_lpdf(y::real, family, hi, args...)::real = begin
        rv::real[1]
        rv[1] = negative_infinity()
        if y <= hi
            rv[1] = density(family, y, args...) - logcdf(family, hi, args...)
        end
        rv[1]
    end
    upper_conditioning_lpmf(y::int, family, hi, args...)::real = begin
        rv::real[1]
        rv[1] = negative_infinity()
        if y <= hi
            rv[1] = density(family, y, args...) - logcdf(family, hi, args...)
        end
        rv[1]
    end
    upper_conditioning_lpdf(y::anything[n], family, hi, args...)::real =
        sum(upper_conditioning_lpdfs(y, family, hi, args...))
    upper_conditioning_lpmf(y::anything[n], family, hi, args...)::real =
        sum(upper_conditioning_lpmfs(y, family, hi, args...))
    upper_conditioning_lpdfs(y::real, family, hi, args...) =
        upper_conditioning_lpdf(y, family, hi, args...)
    upper_conditioning_lpmfs(y::int, family, hi, args...) =
        upper_conditioning_lpmf(y, family, hi, args...)
    upper_conditioning_lpdfs(y::anything[n], family, hi, args...)::vector[n] =
        jbroadcasted(upper_conditioning_lpdf, y, family, hi, args...)
    upper_conditioning_lpmfs(y::anything[n], family, hi, args...)::vector[n] =
        jbroadcasted(upper_conditioning_lpmf, y, family, hi, args...)
    upper_conditioning_rng(family, hi, args...) = begin
        draw::typeof(predictive(family, args...))[1]
        attempts::int[1]
        draw[1] = predictive(family, args...)
        attempts[1] = 1
        while draw[1] > hi && attempts[1] < 100
            draw[1] = predictive(family, args...)
            attempts[1] = attempts[1] + 1
        end
        if draw[1] > hi
            draw[1] = upper_conditioning_inverse_rng(draw[1], family, hi, args...)
        end
        draw[1]
    end
    upper_conditioning_cell_rng(dummy, family, hi, args...) =
        upper_conditioning_rng(family, hi, args...)
    upper_conditioning_rng(vector[n], family, hi, args...)::vector[n] =
        jbroadcasted(upper_conditioning_cell_rng, rep_vector(0., n), family, hi, args...)
    upper_conditioning_rng(real[n], family, hi, args...)::real[n] =
        to_array_1d(jbroadcasted(upper_conditioning_cell_rng, rep_vector(0., n), family, hi, args...))
    upper_conditioning_rng(int[n], family, hi, args...) =
        jbroadcasted(upper_conditioning_cell_rng, rep_array(0, n), family, hi, args...)

    # Two-sided inclusive truncation.
    conditioning_lpdf(y::real, family, lo, hi, args...)::real = begin
        rv::real[1]
        rv[1] = negative_infinity()
        if lo >= hi
            reject("truncated: lower bound must be less than upper bound")
        else
            if y >= lo
                if y <= hi
                    rv[1] = density(family, y, args...) -
                        log_diff_exp(logcdf(family, hi, args...), logcdf(family, lo, args...))
                end
            end
        end
        rv[1]
    end
    conditioning_lpmf(y::int, family, lo, hi, args...)::real = begin
        rv::real[1]
        rv[1] = negative_infinity()
        if lo > hi
            reject("truncated: lower bound must not exceed upper bound")
        else
            if y >= lo
                if y <= hi
                    rv[1] = density(family, y, args...) -
                        log_diff_exp(logcdf(family, hi, args...), logcdf(family, lo - 1, args...))
                end
            end
        end
        rv[1]
    end
    conditioning_lpdf(y::anything[n], family, lo, hi, args...)::real =
        sum(conditioning_lpdfs(y, family, lo, hi, args...))
    conditioning_lpmf(y::anything[n], family, lo, hi, args...)::real =
        sum(conditioning_lpmfs(y, family, lo, hi, args...))
    conditioning_lpdfs(y::real, family, lo, hi, args...) =
        conditioning_lpdf(y, family, lo, hi, args...)
    conditioning_lpmfs(y::int, family, lo, hi, args...) =
        conditioning_lpmf(y, family, lo, hi, args...)
    conditioning_lpdfs(y::anything[n], family, lo, hi, args...)::vector[n] =
        jbroadcasted(conditioning_lpdf, y, family, lo, hi, args...)
    conditioning_lpmfs(y::anything[n], family, lo, hi, args...)::vector[n] =
        jbroadcasted(conditioning_lpmf, y, family, lo, hi, args...)
    conditioning_outside(x, lo, hi)::int = begin
        rv::int[1]
        rv[1] = 0
        if x < lo
            rv[1] = 1
        else
            if x > hi
                rv[1] = 1
            end
        end
        rv[1]
    end
    conditioning_rng(family, lo, hi, args...) = begin
        draw::typeof(predictive(family, args...))[1]
        attempts::int[1]
        draw[1] = predictive(family, args...)
        attempts[1] = 1
        while conditioning_outside(draw[1], lo, hi) == 1 && attempts[1] < 100
            draw[1] = predictive(family, args...)
            attempts[1] = attempts[1] + 1
        end
        if conditioning_outside(draw[1], lo, hi) == 1
            draw[1] = conditioning_inverse_rng(draw[1], family, lo, hi, args...)
        end
        draw[1]
    end

    # Inverse-CDF draws (snag truncated-rng-re-bd555cae). Redrawing alone needs
    # about 1/mass attempts, so a range holding 1e-7 of the base law's mass
    # exhausted any budget. The inversion takes the smallest x in the range with
    # tail(x) <= t, for a decreasing `tail` of the base CDF and a t that places a
    # uniform draw between the range's ends on that scale: an integer draw is the
    # exact quantile, a real one is bisected until no float lies between the
    # brackets. Each scale keeps its precision in its own tail, so a lower bound
    # inverts the log CCDF, an upper bound the log CDF, and a two-sided range the
    # CCDF when it lies above the base median. The first argument is the missed
    # draw, which only selects the integer or real method. A range whose mass is
    # not representable in floating point on its scale rejects, saying so.
    conditioning_ccdf_tail(family, x, args...)::real = conditioning_logccdf(family, x, args...)
    conditioning_cdf_tail(family, x, args...)::real = -logcdf(family, x, args...)
    # The last point below the range: `lo - 1` for a count, `lo` for a real.
    conditioning_edge(x::int, lo) = lo - 1
    conditioning_edge(x::real, lo) = lo
    # (a + b) ÷ 2, without forming a + b or b - a, either of which can leave
    # Stan's 32-bit integers.
    conditioning_midpoint(a::int, b::int)::int =
        a ÷ 2 + b ÷ 2 + ((a - 2 * (a ÷ 2)) + (b - 2 * (b ÷ 2))) ÷ 2
    # The smallest x in (a, b] with tail(x) <= t, given tail(a) > t >= tail(b).
    conditioning_bisect(x::int, tail, family, t::real, a::int, b::int, args...)::int = begin
        bracket::int[2]
        bracket[1] = a
        bracket[2] = b
        while bracket[1] + 1 < bracket[2]
            mid = conditioning_midpoint(bracket[1], bracket[2])
            if tail(family, mid, args...) <= t
                bracket[2] = mid
            else
                bracket[1] = mid
            end
        end
        bracket[2]
    end
    conditioning_bisect(x::real, tail, family, t::real, a, b, args...)::real = begin
        bracket::real[2]
        mid::real[1]
        bracket[1] = a
        bracket[2] = b
        mid[1] = 0.5 * bracket[1] + 0.5 * bracket[2]
        while bracket[1] < mid[1] && mid[1] < bracket[2]
            if tail(family, mid[1], args...) <= t
                bracket[2] = mid[1]
            else
                bracket[1] = mid[1]
            end
            mid[1] = 0.5 * bracket[1] + 0.5 * bracket[2]
        end
        bracket[2]
    end
    # The open end of a one-sided range: from a known end, double the step until
    # tail crosses t. A count stops at Stan's largest integer, a real at the
    # family's support, where its CDF is still defined.
    conditioning_bracket_up(x::int, tail, family, t::real, a::int, args...)::int = begin
        b::int[1]
        step::int[1]
        b[1] = a + 1
        step[1] = 1
        while tail(family, b[1], args...) > t
            if b[1] == 2147483647
                reject("truncated: the draw lies above Stan's largest integer")
            end
            if b[1] > 2147483647 - step[1]
                b[1] = 2147483647
            else
                b[1] = b[1] + step[1]
            end
            if step[1] < 1073741824
                step[1] = 2 * step[1]
            end
        end
        b[1]
    end
    conditioning_bracket_up(x::real, tail, family, t::real, a, args...)::real = begin
        support = conditioning_support_upper(family, args...)
        b::real[1]
        step::real[1]
        step[1] = fmax(1.0, fmax(a, -a))
        b[1] = fmin(a + step[1], support)
        while tail(family, b[1], args...) > t
            if b[1] >= support
                reject("truncated: the draw lies above the family's support")
            end
            step[1] = 2 * step[1]
            b[1] = fmin(a + step[1], support)
        end
        if is_inf(b[1]) == 1
            reject("truncated: the draw lies above the largest finite real")
        end
        b[1]
    end
    conditioning_bracket_down(x::int, tail, family, t::real, b::int, args...)::int = begin
        a::int[1]
        step::int[1]
        a[1] = b - 1
        step[1] = 1
        while tail(family, a[1], args...) <= t
            if a[1] == -2147483647
                reject("truncated: the draw lies below Stan's smallest integer")
            end
            if a[1] < -2147483647 + step[1]
                a[1] = -2147483647
            else
                a[1] = a[1] - step[1]
            end
            if step[1] < 1073741824
                step[1] = 2 * step[1]
            end
        end
        a[1]
    end
    conditioning_bracket_down(x::real, tail, family, t::real, b, args...)::real = begin
        support = conditioning_support_lower(family, args...)
        a::real[1]
        step::real[1]
        step[1] = fmax(1.0, fmax(b, -b))
        a[1] = fmax(b - step[1], support)
        while tail(family, a[1], args...) <= t
            if a[1] <= support
                reject("truncated: the draw lies below the family's support")
            end
            step[1] = 2 * step[1]
            a[1] = fmax(b - step[1], support)
        end
        if is_inf(a[1]) == 1
            reject("truncated: the draw lies below the smallest finite real")
        end
        a[1]
    end
    lower_conditioning_inverse_rng(x, family, lo, args...) = begin
        edge = conditioning_edge(x, lo)
        log_mass = conditioning_logccdf(family, edge, args...)
        if is_nan(log_mass) == 1 || log_mass == negative_infinity()
            reject("truncated: no probability above the lower bound is representable in floating point")
        end
        t = log1m(uniform_rng(0, 1)) + log_mass
        conditioning_bisect(x, conditioning_ccdf_tail, family, t, edge,
            conditioning_bracket_up(x, conditioning_ccdf_tail, family, t, edge, args...), args...)
    end
    upper_conditioning_inverse_rng(x, family, hi, args...) = begin
        log_mass = logcdf(family, hi, args...)
        if is_nan(log_mass) == 1 || log_mass == negative_infinity()
            reject("truncated: no probability below the upper bound is representable in floating point")
        end
        t = -(log1m(uniform_rng(0, 1)) + log_mass)
        conditioning_bisect(x, conditioning_cdf_tail, family, t,
            conditioning_bracket_down(x, conditioning_cdf_tail, family, t, hi, args...), hi, args...)
    end
    conditioning_inverse_rng(x, family, lo, hi, args...) = begin
        if lo > hi
            reject("truncated: lower bound must not exceed upper bound")
        end
        draw::typeof(x)[1]
        edge = conditioning_edge(x, lo)
        log_cdf_edge = logcdf(family, edge, args...)
        log_v = log1m(uniform_rng(0, 1))
        if log_cdf_edge > log(0.5)
            log_ccdf_edge = conditioning_logccdf(family, edge, args...)
            log_ccdf_hi = conditioning_logccdf(family, hi, args...)
            log_mass = log_diff_exp(log_ccdf_edge, log_ccdf_hi)
            if is_nan(log_mass) == 1 || log_mass == negative_infinity()
                reject("truncated: no probability between the bounds is representable in floating point")
            end
            draw[1] = conditioning_bisect(x, conditioning_ccdf_tail, family,
                log_sum_exp(log_ccdf_hi, log_v + log_mass), edge, hi, args...)
        else
            log_cdf_hi = logcdf(family, hi, args...)
            log_mass = log_diff_exp(log_cdf_hi, log_cdf_edge)
            if is_nan(log_mass) == 1 || log_mass == negative_infinity()
                reject("truncated: no probability between the bounds is representable in floating point")
            end
            draw[1] = conditioning_bisect(x, conditioning_cdf_tail, family,
                -log_sum_exp(log_cdf_edge, log_v + log_mass), edge, hi, args...)
        end
        draw[1]
    end
    conditioning_cell_rng(dummy, family, lo, hi, args...) =
        conditioning_rng(family, lo, hi, args...)
    conditioning_rng(vector[n], family, lo, hi, args...)::vector[n] =
        jbroadcasted(conditioning_cell_rng, rep_vector(0., n), family, lo, hi, args...)
    conditioning_rng(real[n], family, lo, hi, args...)::real[n] =
        to_array_1d(jbroadcasted(conditioning_cell_rng, rep_vector(0., n), family, lo, hi, args...))
    conditioning_rng(int[n], family, lo, hi, args...)::int[n] =
        jbroadcasted(conditioning_cell_rng, rep_array(0, n), family, lo, hi, args...)

    # Ordered re-draw by rejection (`_ordered_rejection`, passes.jl) ---------
    # `rng` is a family's resolved vector draw. Stan's `ordered` check is strict,
    # so a tie is a rejection too. An ordered vector has no CDF to invert, so
    # the redraws stop with a `reject` after 100000 attempts.
    ordered_rejection_unsorted(x::vector[n])::int = begin
        rv::int[1]
        rv[1] = 0
        for i in 2:n
            if x[i] <= x[i - 1]
                rv[1] = 1
            end
        end
        rv[1]
    end
    ordered_rejection_rng(vector[n], rng, args...)::vector[n] = begin
        draw::vector[n]
        attempts::int[1]
        draw = rng(vector[n], args...)
        attempts[1] = 1
        while ordered_rejection_unsorted(draw) == 1
            if attempts[1] >= 100000
                reject("ordered: rejection sampler found no increasing draw in 100000 attempts; the prior rarely draws its elements in order")
            end
            draw = rng(vector[n], args...)
            attempts[1] = attempts[1] + 1
        end
        draw
    end

    # Threshold censoring (`censored`) --------------------------------------
    lower_clamping_lpdf(y::real, family, lo, args...)::real = begin
        rv::real[1]
        rv[1] = negative_infinity()
        if y == lo
            rv[1] = logcdf(family, lo, args...)
        else
            if y > lo
                rv[1] = density(family, y, args...)
            end
        end
        rv[1]
    end
    lower_clamping_lpmf(y::int, family, lo, args...)::real = begin
        rv::real[1]
        rv[1] = negative_infinity()
        if y == lo
            rv[1] = logcdf(family, lo, args...)
        else
            if y > lo
                rv[1] = density(family, y, args...)
            end
        end
        rv[1]
    end
    lower_clamping_lpdf(y::anything[n], family, lo, args...)::real =
        sum(lower_clamping_lpdfs(y, family, lo, args...))
    lower_clamping_lpmf(y::anything[n], family, lo, args...)::real =
        sum(lower_clamping_lpmfs(y, family, lo, args...))
    lower_clamping_lpdfs(y::real, family, lo, args...) =
        lower_clamping_lpdf(y, family, lo, args...)
    lower_clamping_lpmfs(y::int, family, lo, args...) =
        lower_clamping_lpmf(y, family, lo, args...)
    lower_clamping_lpdfs(y::anything[n], family, lo, args...)::vector[n] =
        jbroadcasted(lower_clamping_lpdf, y, family, lo, args...)
    lower_clamping_lpmfs(y::anything[n], family, lo, args...)::vector[n] =
        jbroadcasted(lower_clamping_lpmf, y, family, lo, args...)
    lower_clamping_rng(family, lo, args...) = begin
        draw::typeof(predictive(family, args...))[1]
        draw[1] = predictive(family, args...)
        if draw[1] < lo
            draw[1] = lo
        end
        draw[1]
    end
    lower_clamping_cell_rng(dummy, family, lo, args...) =
        lower_clamping_rng(family, lo, args...)
    lower_clamping_rng(vector[n], family, lo, args...)::vector[n] =
        jbroadcasted(lower_clamping_cell_rng, rep_vector(0., n), family, lo, args...)
    lower_clamping_rng(real[n], family, lo, args...)::real[n] =
        to_array_1d(jbroadcasted(lower_clamping_cell_rng, rep_vector(0., n), family, lo, args...))
    lower_clamping_rng(int[n], family, lo, args...) =
        jbroadcasted(lower_clamping_cell_rng, rep_array(0, n), family, lo, args...)

    upper_clamping_lpdf(y::real, family, hi, args...)::real = begin
        rv::real[1]
        rv[1] = negative_infinity()
        if y == hi
            rv[1] = logccdf(family, hi, args...)
        else
            if y < hi
                rv[1] = density(family, y, args...)
            end
        end
        rv[1]
    end
    upper_clamping_lpmf(y::int, family, hi, args...)::real = begin
        rv::real[1]
        rv[1] = negative_infinity()
        if y == hi
            rv[1] = logccdf(family, hi - 1, args...)
        else
            if y < hi
                rv[1] = density(family, y, args...)
            end
        end
        rv[1]
    end
    upper_clamping_lpdf(y::anything[n], family, hi, args...)::real =
        sum(upper_clamping_lpdfs(y, family, hi, args...))
    upper_clamping_lpmf(y::anything[n], family, hi, args...)::real =
        sum(upper_clamping_lpmfs(y, family, hi, args...))
    upper_clamping_lpdfs(y::real, family, hi, args...) =
        upper_clamping_lpdf(y, family, hi, args...)
    upper_clamping_lpmfs(y::int, family, hi, args...) =
        upper_clamping_lpmf(y, family, hi, args...)
    upper_clamping_lpdfs(y::anything[n], family, hi, args...)::vector[n] =
        jbroadcasted(upper_clamping_lpdf, y, family, hi, args...)
    upper_clamping_lpmfs(y::anything[n], family, hi, args...)::vector[n] =
        jbroadcasted(upper_clamping_lpmf, y, family, hi, args...)
    upper_clamping_rng(family, hi, args...) = begin
        draw::typeof(predictive(family, args...))[1]
        draw[1] = predictive(family, args...)
        if draw[1] > hi
            draw[1] = hi
        end
        draw[1]
    end
    upper_clamping_cell_rng(dummy, family, hi, args...) =
        upper_clamping_rng(family, hi, args...)
    upper_clamping_rng(vector[n], family, hi, args...)::vector[n] =
        jbroadcasted(upper_clamping_cell_rng, rep_vector(0., n), family, hi, args...)
    upper_clamping_rng(real[n], family, hi, args...)::real[n] =
        to_array_1d(jbroadcasted(upper_clamping_cell_rng, rep_vector(0., n), family, hi, args...))
    upper_clamping_rng(int[n], family, hi, args...) =
        jbroadcasted(upper_clamping_cell_rng, rep_array(0, n), family, hi, args...)

    # Two-sided censoring.
    clamping_lpdf(y::real, family, lo, hi, args...)::real = begin
        rv::real[1]
        if lo >= hi
            reject("censored: lower bound must be less than upper bound")
        end
        rv[1] = negative_infinity()
        if y == lo
            rv[1] = logcdf(family, lo, args...)
        else
            if y == hi
                rv[1] = logccdf(family, hi, args...)
            else
                if y > lo
                    if y < hi
                        rv[1] = density(family, y, args...)
                    end
                end
            end
        end
        rv[1]
    end
    clamping_lpmf(y::int, family, lo, hi, args...)::real = begin
        rv::real[1]
        if lo >= hi
            reject("censored: lower bound must be less than upper bound")
        end
        rv[1] = negative_infinity()
        if y == lo
            rv[1] = logcdf(family, lo, args...)
        else
            if y == hi
                rv[1] = logccdf(family, hi - 1, args...)
            else
                if y > lo
                    if y < hi
                        rv[1] = density(family, y, args...)
                    end
                end
            end
        end
        rv[1]
    end
    clamping_lpdf(y::anything[n], family, lo, hi, args...)::real =
        sum(clamping_lpdfs(y, family, lo, hi, args...))
    clamping_lpmf(y::anything[n], family, lo, hi, args...)::real =
        sum(clamping_lpmfs(y, family, lo, hi, args...))
    clamping_lpdfs(y::real, family, lo, hi, args...) =
        clamping_lpdf(y, family, lo, hi, args...)
    clamping_lpmfs(y::int, family, lo, hi, args...) =
        clamping_lpmf(y, family, lo, hi, args...)
    clamping_lpdfs(y::anything[n], family, lo, hi, args...)::vector[n] =
        jbroadcasted(clamping_lpdf, y, family, lo, hi, args...)
    clamping_lpmfs(y::anything[n], family, lo, hi, args...)::vector[n] =
        jbroadcasted(clamping_lpmf, y, family, lo, hi, args...)
    clamping_rng(family, lo, hi, args...) = begin
        draw::typeof(predictive(family, args...))[1]
        draw[1] = predictive(family, args...)
        if draw[1] < lo
            draw[1] = lo
        else
            if draw[1] > hi
                draw[1] = hi
            end
        end
        draw[1]
    end
    clamping_cell_rng(dummy, family, lo, hi, args...) =
        clamping_rng(family, lo, hi, args...)
    clamping_rng(vector[n], family, lo, hi, args...)::vector[n] =
        jbroadcasted(clamping_cell_rng, rep_vector(0., n), family, lo, hi, args...)
    clamping_rng(real[n], family, lo, hi, args...)::real[n] =
        to_array_1d(jbroadcasted(clamping_cell_rng, rep_vector(0., n), family, lo, hi, args...))
    clamping_rng(int[n], family, lo, hi, args...)::int[n] =
        jbroadcasted(clamping_cell_rng, rep_array(0, n), family, lo, hi, args...)

    # Genuine interval evidence: latent value is known only to lie in
    # `(lower, upper]`. The observed marker drives shape; predictive draws stay
    # on the uncoarsened base-family scale.
    interval_evidence_impl_lpdf(y, family, lo, hi, args...)::real = begin
        rv::real[1]
        if lo >= hi
            reject("interval_censored: lower bound must be less than upper bound")
        end
        rv[1] = log_diff_exp(
            logcdf(family, hi, args...), logcdf(family, lo, args...)
        )
        rv[1]
    end
    interval_evidence_impl_lpmf(y, family, lo, hi, args...)::real = begin
        rv::real[1]
        if lo >= hi
            reject("interval_censored: lower bound must be less than upper bound")
        end
        rv[1] = log_diff_exp(
            logcdf(family, hi, args...), logcdf(family, lo, args...)
        )
        rv[1]
    end
    interval_evidence_impl_lpdf(y::anything[n], family, lo, hi, args...)::real =
        sum(interval_evidence_impl_lpdfs(y, family, lo, hi, args...))
    interval_evidence_impl_lpmf(y::anything[n], family, lo, hi, args...)::real =
        sum(interval_evidence_impl_lpmfs(y, family, lo, hi, args...))
    interval_evidence_impl_lpdfs(y::anything[n], family, lo, hi, args...)::vector[n] =
        jbroadcasted(interval_evidence_impl_lpdf, y, family, lo, hi, args...)
    interval_evidence_impl_lpmfs(y::anything[n], family, lo, hi, args...)::vector[n] =
        jbroadcasted(interval_evidence_impl_lpmf, y, family, lo, hi, args...)
    interval_evidence_impl_lpdfs(y, family, lo, hi, args...) =
        interval_evidence_impl_lpdf(y, family, lo, hi, args...)
    interval_evidence_impl_lpmfs(y, family, lo, hi, args...) =
        interval_evidence_impl_lpmf(y, family, lo, hi, args...)
    interval_evidence_impl_rng(family, lo, hi, args...) =
        predictive(family, args...)
    # Sized forms forward the token to the base family's own companion — same
    # inherited-guard posture as `weighted_rng` above (todo 19n8abc).
    interval_evidence_impl_rng(real[n], family, lo, hi, args...) =
        predictive(family, real[n], args...)
    interval_evidence_impl_rng(vector[n], family, lo, hi, args...) =
        predictive(family, vector[n], args...)
    interval_evidence_impl_rng(int[n], family, lo, hi, args...) =
        predictive(family, int[n], args...)
    end
    # `dummy` no-op distribution: `y ~ dummy(args...)` contributes +0. Bodies
    # ignore their args; monomorphization emits a concrete Stan
    # `real dummy_lpdf(<y>, <args>) { return 0.; }` per call-site signature, so
    # the native `y ~ dummy(args...);` resolves. `dummy_lpdfs` is the gq
    # pointwise-likelihood term, `dummy_rng` the gq draw — both trivially 0.
    @stanonly dummy_lpdf(y, args...) = 0.
    @stanonly dummy_lpdfs(y, args...) = 0.
    @stanonly dummy_rng(args...) = 0.
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
    @lhs multi_normal_cholesky_lpdf(obs::vector[n], loc::vector[n], scale)
    @lhs dirichlet_lpdf(w::simplex[n], alpha::vector[n])
    @lhs lkj_corr_lpdf(L::corr_matrix, x::real)
    @lhs lkj_corr_cholesky_lpdf(L::cholesky_factor_corr, x::real)
    @lhs wishart_lpdf(L::cov_matrix[m], x::real, sigma::matrix[m,m])
    @lhs wishart_cholesky_lpdf(L::cholesky_factor_cov[m], x::real, sigma::matrix[m,m])
    # Sized ELEMENT densities over any natively-constrained square matrix (an
    # array element `L[i]`, a plate cell). The unsized `@lhs` rows above only
    # name a family's support type; a value sized by its one dim
    # (`r_ndim(square_matrix) == 1`) needs these to type the density call.
    lkj_corr_lpdf(L::square_matrix[n], eta::real)::real
    lkj_corr_cholesky_lpdf(L::square_matrix[n], eta::real)::real
    wishart_lpdf(W::square_matrix[n], nu::real, Sigma::matrix[n,n])::real
    inv_wishart_lpdf(W::square_matrix[n], nu::real, Sigma::matrix[n,n])::real
    wishart_cholesky_lpdf(L::square_matrix[n], nu::real, LSigma::matrix[n,n])::real
    inv_wishart_cholesky_lpdf(L::square_matrix[n], nu::real, LSigma::matrix[n,n])::real

    lognormal_rng(loc::real, scale::real)::real
    student_t_rng(nu::real, loc::real, scale::real)::real
    multi_normal_rng(loc::vector[n], args...)::vector[n]
    multi_normal_cholesky_rng(loc::vector[n], scale)::vector[n]
    multi_student_t_rng(nu::real, loc::vector[n], scale)::vector[n]
    multi_student_t_cholesky_rng(nu::real, loc::vector[n], scale)::vector[n]
    wishart_rng(nu::real, Sigma::matrix[n,n])::matrix[n,n]
    inv_wishart_rng(nu::real, Sigma::matrix[n,n])::matrix[n,n]
    wishart_cholesky_rng(nu::real, L::matrix[n,n])::matrix[n,n]
    inv_wishart_cholesky_rng(nu::real, L::matrix[n,n])::matrix[n,n]
    bernoulli_rng(::vector[n])::int[n]
    bernoulli_logit_rng(::real)::int
    bernoulli_logit_rng(::vector[n])::int[n]
    bernoulli_logit_glm_rng(X::matrix[m,n], alpha, beta)::int[m]
    @stanonly bernoulli_logit_glm_rng(X::matrix[m,n], alpha::real, beta) = bernoulli_logit_glm_rng(X, rep_vector(alpha, m), beta)
    # `bernoulli_logit_glm_rng` above is the ONLY `*_glm_rng` Stan Math ships
    # (checked against `stanc --dump-stan-math-signatures`, 2.39.0). Declaring
    # the other three GLM families' rngs as native would emit calls stanc
    # rejects, so define them as SB functions over the linear predictor
    # `alpha + X * beta` — the same lowering `binomial_logit_rng` gets below.
    # Empty-segment guard (todo 19n8abc): a 0-row `X` with a scalar `sigma` /
    # `phi` replays the 2-arg-loop segfault, so the lowered plain forms
    # early-return exactly like their sized-token overloads below.
    @stanonly normal_id_glm_rng(X::matrix[m,n], alpha::real, beta, sigma)::vector[m] = if m == 0
        rv::vector[m]
        rv
    else
        to_vector(normal_rng(rep_vector(alpha, m) + X * beta, sigma))
    end
    @stanonly normal_id_glm_rng(X::matrix[m,n], alpha::vector[m], beta, sigma)::vector[m] = if m == 0
        rv::vector[m]
        rv
    else
        to_vector(normal_rng(alpha + X * beta, sigma))
    end
    @stanonly poisson_log_glm_rng(X::matrix[m,n], alpha::real, beta)::int[m] = if m == 0
        rv::int[m]
        rv
    else
        poisson_log_rng(rep_vector(alpha, m) + X * beta)
    end
    @stanonly poisson_log_glm_rng(X::matrix[m,n], alpha::vector[m], beta)::int[m] = if m == 0
        rv::int[m]
        rv
    else
        poisson_log_rng(alpha + X * beta)
    end
    @stanonly neg_binomial_2_log_glm_rng(X::matrix[m,n], alpha::real, beta, phi)::int[m] = if m == 0
        rv::int[m]
        rv
    else
        neg_binomial_2_log_rng(rep_vector(alpha, m) + X * beta, phi)
    end
    @stanonly neg_binomial_2_log_glm_rng(X::matrix[m,n], alpha::vector[m], beta, phi)::int[m] = if m == 0
        rv::int[m]
        rv
    else
        neg_binomial_2_log_rng(alpha + X * beta, phi)
    end
    beta_rng(args...)::real
    binomial_rng(args...)::int
    binomial_logit_rng(n::int[m], p::vector[m])::int[m]
    binomial_logit_rng(n::vector[m], p::vector[m])::int[m]
    @juliacompat broadcasted_getindex(x, i) = x
    @juliacompat broadcasted_getindex(x::anything[m], i) = x[i]
    # `jbroadcasted(f, args...)` is a TRACE-LEVEL construct (custom tracetype +
    # fundef below, not a fixed-arity @deffun): arbitrary arity, any array-arg
    # positions, and an INFERRED output container (int element → array[] int,
    # real element → vector[n]). Its Stan function is generated per call shape.
    @stanonly begin
    # Empty-segment guard (todo 19n8abc): same scalar-`1` mix as the
    # `std_normal_rng` sized overloads below — `n == 0` returns the empty draw.
    vector_std_normal_rng(n::int)::vector[n] = if n == 0
        rv::vector[n]
        rv
    else
        to_vector(normal_rng(rep_vector(0, n), 1))
    end
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
    # Stan's native multinomial density omits the total count N because it is
    # recoverable as sum(obs), while multinomial_rng requires N. SLIC's auto-GQ
    # contract forwards exactly the density-family arguments, so expose an
    # explicit-N overload that keeps density and predictive calls symmetric.
    # The batched form is one composition per row and produces one pointwise
    # likelihood value / predictive draw per row.
    @lhs multinomial_lpmf(obs::int[K], probs::vector[K], N::int)::real = begin
        if sum(obs) != N
            reject("multinomial: explicit N must equal sum(obs)")
        end
        multinomial_lpmf(obs, probs)
    end
    multinomial_lpmfs(obs::int[K], probs::vector[K], N::int)::real =
        multinomial_lpmf(obs, probs, N)
    @lhs multinomial_lpmf(obs::int[n,K], probs::vector[K], row_N::int[n])::real =
        sum(multinomial_lpmfs(obs, probs, row_N))
    multinomial_lpmfs(obs::int[n,K], probs::vector[K], row_N::int[n])::vector[n] = begin
        rv::vector[n]
        for i in 1:n
            rv[i] = multinomial_lpmf(obs[i, :], probs, row_N[i])
        end
        rv
    end
    # Single draw sized by the one container (`N` is a count, not a vectorized
    # arg); the row form is an explicit loop — both outside the guarded class
    # (audit todo 19n8abc).
    multinomial_rng(int[K], probs::vector[K], N::int)::int[K] =
        multinomial_rng(probs, N)
    multinomial_rng(int[n,K], probs::vector[K], row_N::int[n])::int[n,K] = begin
        rv::int[n,K]
        for i in 1:n
            rv[i, :] = multinomial_rng(probs, row_N[i])
        end
        rv
    end
    # A vector-valued plate cell is collected as matrix[k, n], one observation
    # per column. Stan's categorical-logit primitive accepts only one vector of
    # logits at a time, so provide the density and pointwise companions for the
    # public matrix form explicitly instead of relying on row-wise broadcasting.
    categorical_logit_lpmf(y::int[n], eta::matrix[k,n])::real = begin
        rv = 0.0
        for i in 1:n
            rv += categorical_logit_lpmf(y[i], eta[:, i])
        end
        rv
    end
    categorical_logit_lpmfs(y::int[n], eta::matrix[k,n])::vector[n] = begin
        rv::vector[n]
        for i in 1:n
            rv[i] = categorical_logit_lpmf(y[i], eta[:, i])
        end
        rv
    end
    # Shared-simplex form: n draws against one probability vector (e.g.
    # `s ~ dirichlet(…); y::int[n] ~ categorical(s)`). The scalar fallback
    # carries the REQUIRED `::real` (rule documented below): the singular
    # `categorical_lpmf(args...)` stub is itself unannotated (snag
    # sbbrmi-of-s-diri-78c6f9e7).
    categorical_lpmfs(args...)::real = categorical_lpmf(args...)
    categorical_lpmfs(y::int[n], p::vector[k])::vector[n] = begin
        rv::vector[n]
        for i in 1:n
            rv[i] = categorical_lpmf(y[i], p)
        end
        rv
    end
    # Scalar-fallback pointwise-density companions delegate to the `_lpdf`, which
    # returns `real`; the explicit `::real` return type is REQUIRED — without it
    # the varargs signature monomorphises to a Stan function with return type
    # `anything`, which `stanc` rejects. This surfaced when a constrained per-cell
    # parameter inside a plate (`cell[g] ~ dirichlet(…)`) fetched `dirichlet_lpdfs`
    # (snag plate-constraine-90607054).
    multi_normal_lpdfs(args...)::real = multi_normal_lpdf(args...)
    multi_normal_cholesky_lpdfs(args...)::real = multi_normal_cholesky_lpdf(args...)
    multi_normal_prec_lpdfs(args...)::real = multi_normal_prec_lpdf(args...)
    multi_student_t_lpdfs(args...)::real = multi_student_t_lpdf(args...)
    multi_student_t_cholesky_lpdfs(args...)::real = multi_student_t_cholesky_lpdf(args...)
    wishart_lpdfs(args...)::real = wishart_lpdf(args...)
    inv_wishart_lpdfs(args...)::real = inv_wishart_lpdf(args...)
    wishart_cholesky_lpdfs(args...)::real = wishart_cholesky_lpdf(args...)
    inv_wishart_cholesky_lpdfs(args...)::real = inv_wishart_cholesky_lpdf(args...)
    lkj_corr_lpdfs(args...)::real = lkj_corr_lpdf(args...)
    dirichlet_lpdfs(args...)::real = dirichlet_lpdf(args...)
    # Explicit-size spelling of the array density below; the sizes are implied by
    # `L` itself, so it delegates.
    @lhs lkj_corr_cholesky_lpdf(L::cholesky_factor_corr[m,n], x::real, m::int, n::int)::real =
        lkj_corr_cholesky_lpdf(L, x)
    lkj_corr_cholesky_lpdfs(args...)::real = lkj_corr_cholesky_lpdf(args...)
    # A CONSTRAINED-MATRIX element (`cholesky_factor_corr[K]`, e.g. a per-stratum
    # plate cell, snag stanblocks-plate-248f67d3) is itself ndim-1, so the
    # `anything[n]` broadcast form below would steal the fetch from the scalar
    # fallback and then broadcast over the constraint size — peeling the matrix
    # to `anything`. Route the element to the density DIRECTLY: more specific
    # center, same arity, so the ndim-2 whole-array broadcast is untouched.
    lkj_corr_cholesky_lpdfs(L::cholesky_factor_corr[m], x::real)::real = lkj_corr_cholesky_lpdf(L, x)
    lkj_corr_cholesky_lpdfs(L::anything[n], x) = jbroadcasted(lkj_corr_cholesky_lpdfs, L, x)
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
    # Token form delegates to the explicit-loop form above — no guard (todo 19n8abc).
    ordered_logistic_rng(int[n], eta::vector[n], c::vector[m])::int[n] = ordered_logistic_rng(eta, c)
    # Single native arg — same immunity as the 1-arg loops (todo 19n8abc):
    # no `n == 0` guard. The `to_vector` is load-bearing: the vectorized
    # native returns `array[] real`, so without it stanc rejects the helper
    # (snag replaying-a-zero-ab1b204c).
    vector_exponential_rng(rate::real, n::int)::vector[n] = to_vector(exponential_rng(rep_vector(rate, n)))
    end
    # Stan's `lkj_corr_cholesky_rng(int K, real eta)` returns a K×K Cholesky
    # factor. WITHOUT the `::matrix[n,n]` the tracetype is `anything`, so the
    # auto-GQ redraw of a cv-tainted `L::cholesky_factor_corr[K] ~
    # lkj_corr_cholesky(eta)` blew up in `stan_code` with
    # `tracetype not defined for L::anything = lkj_corr_cholesky_rng(...)`
    # (snag build-a-declarat-ab2d2471; the trap was already recorded in the
    # primer's cv section). `matrix` — not `cholesky_factor_corr` — matches the
    # `dirichlet_rng`/`simplex` precedent: a gq redraw declares the natural
    # unconstrained container, not the constrained parameter type.
    lkj_corr_cholesky_rng(n::int, eta::real)::matrix[n,n]
    # Plain-corr twin of the above: same int-dimension native, unconstrained
    # matrix draw (the `matrix`, not `corr_matrix`, precedent stands).
    lkj_corr_rng(n::int, eta::real)::matrix[n,n]

    # Scalar log-CDF companion signatures used by distribution HOFs. The
    # `@builtin_module` manifest declares names for every probability family,
    # including families for which Stan has no CDF. Only this Stan-supported
    # set is admitted by the capability gate in lpxf_builtin.jl.
    normal_cdf(args...)
    std_normal_lcdf(args...)
    std_normal_lccdf(args...)
    normal_lcdf(args...)
    normal_lccdf(args...)
    # Numerically stable erfc reformulation of the normal log-CDF/CCDF, selected
    # for the `normal` family by `logcdf_expr`/`logccdf_expr` (lpxf_builtin.jl) so
    # every `censored`/`truncated`/`interval_censored(normal, …)` tail mass emits
    # these instead of Stan's native companions. Stan Math's own
    # `normal_lcdf`/`normal_lccdf` reverse-mode AD rules are inaccurate (~1e-6 vs
    # ~1e-14 finite-difference truth) in the practical regime, and `normal_lccdf`
    # diverges (value -Inf, gradient +Inf) past ~9σ where the erfc form is still
    # correct to ~37σ. `log(0.5·erfc(∓z/√2))` (z = (x-loc)/scale) autodiffs
    # through exact-derivative primitives (`erfc`, `log`). `@stanonly`: the name
    # is not in the probability family, so `@deffun` would otherwise install a
    # Julia target calling the Stan-only `erfc` token — these are Stan-emission
    # only. See snag censored-normal-410c4e35 for the isolated-kernel measurement.
    #
    # Past ~37.6σ the erfc form breaks: `erfc` goes subnormal (gradient ±Inf),
    # then underflows to 0 (value -Inf, gradient 0·Inf = NaN). A NaN gradient
    # survives a `log_sum_exp` whose other term is finite, so a censored mixture
    # rejects every start (snag fit-a-left-censo-942540f9). Beyond 35σ the
    # helpers therefore switch to `std_normal_lcdf_tail`, the asymptotic
    # Mills-ratio expansion log Φ(z) = -z²/2 - log(-z) - log(2π)/2 + log1p(S),
    # S = Σₖ₌₁⁶ (-1)ᵏ(2k-1)!!/z²ᵏ, again built from exact-derivative primitives.
    # At |z| ≥ 35 the first omitted term is < 3.3e-17, so value and gradient sit
    # at the double-precision floor (within 1.6e-16 relative of a 512-bit
    # reference, out to 1e8σ). Stan's native `std_normal_lcdf` is no substitute: its
    # deep-tail gradient is off by 7e-6 at 40σ and 4e-4 at 60σ (BridgeStan
    # 2.9.0). The ternary evaluates only the selected branch, so no underflowed
    # erfc node reaches the AD tape; for |z| ≤ 35 the erfc branch is unchanged.
    @stanonly begin
    std_normal_lcdf_tail(z::real)::real = begin
        w = inv(square(z))
        -0.5 * square(z) - log(-z) - 0.9189385332046728 +
            log1p(w * (-1 + w * (3 + w * (-15 + w * (105 + w * (-945 + w * 10395))))))
    end
    normal_lcdf_stable(x::real, loc::real, scale::real)::real =
        (x - loc) / scale < -35 ? std_normal_lcdf_tail((x - loc) / scale) :
            log(erfc(-(x - loc) / (scale * sqrt(2.)))) - log(2.)
    normal_lccdf_stable(x::real, loc::real, scale::real)::real =
        (x - loc) / scale > 35 ? std_normal_lcdf_tail((loc - x) / scale) :
            log(erfc((x - loc) / (scale * sqrt(2.)))) - log(2.)
    # Stan Math computes these three CCDFs as `log(1 - inc_beta(…))`, which is
    # -Inf once the upper-tail mass falls below ~1e-16
    # (`neg_binomial_2_lccdf(249 | 70, 50)` is -Inf; the mass is 5.1e-21). The
    # incomplete-beta symmetry 1 - I_p(a, b) = I_{1-p}(b, a) evaluates the same
    # tail directly, to full relative precision. Selected only by the truncated
    # draws' inversion (`conditioning_logccdf`, lpxf_builtin.jl); the densities
    # keep Stan's companions. snag truncated-rng-re-bd555cae.
    neg_binomial_2_lccdf_precise(n::int, mu::real, phi::real)::real =
        n < 0 ? 0.0 : log(inc_beta(n + 1.0, phi, mu / (mu + phi)))
    neg_binomial_lccdf_precise(n::int, alpha::real, beta::real)::real =
        n < 0 ? 0.0 : log(inc_beta(n + 1.0, alpha, 1.0 / (1.0 + beta)))
    binomial_lccdf_precise(n::int, N::int, theta::real)::real =
        n < 0 ? 0.0 : (n >= N ? negative_infinity() : log(inc_beta(n + 1.0, N - n, theta)))
    end
    student_t_lcdf(args...)
    student_t_lccdf(args...)
    cauchy_lcdf(args...)
    cauchy_lccdf(args...)
    beta_lcdf(args...)
    beta_lccdf(args...)
    beta_proportion_lcdf(args...)
    beta_proportion_lccdf(args...)
    lognormal_lcdf(args...)
    lognormal_lccdf(args...)
    exponential_lcdf(args...)
    exponential_lccdf(args...)
    gamma_lcdf(args...)
    gamma_lccdf(args...)
    inv_gamma_lcdf(args...)
    inv_gamma_lccdf(args...)
    weibull_lcdf(args...)
    weibull_lccdf(args...)
    uniform_lcdf(args...)
    uniform_lccdf(args...)
    chi_square_lcdf(args...)
    chi_square_lccdf(args...)
    inv_chi_square_lcdf(args...)
    inv_chi_square_lccdf(args...)
    scaled_inv_chi_square_lcdf(args...)
    scaled_inv_chi_square_lccdf(args...)
    frechet_lcdf(args...)
    frechet_lccdf(args...)
    rayleigh_lcdf(args...)
    rayleigh_lccdf(args...)
    von_mises_lcdf(args...)
    von_mises_lccdf(args...)
    double_exponential_lcdf(args...)
    double_exponential_lccdf(args...)
    logistic_lcdf(args...)
    logistic_lccdf(args...)
    gumbel_lcdf(args...)
    gumbel_lccdf(args...)
    skew_normal_lcdf(args...)
    skew_normal_lccdf(args...)
    exp_mod_normal_lcdf(args...)
    exp_mod_normal_lccdf(args...)
    skew_double_exponential_lcdf(args...)
    skew_double_exponential_lccdf(args...)
    pareto_lcdf(args...)
    pareto_lccdf(args...)
    pareto_type_2_lcdf(args...)
    pareto_type_2_lccdf(args...)
    bernoulli_lcdf(args...)
    bernoulli_lccdf(args...)
    binomial_lcdf(args...)
    binomial_lccdf(args...)
    beta_binomial_lcdf(args...)
    beta_binomial_lccdf(args...)
    neg_binomial_lcdf(args...)
    neg_binomial_lccdf(args...)
    neg_binomial_2_lcdf(args...)
    neg_binomial_2_lccdf(args...)
    poisson_lcdf(args...)
    poisson_lccdf(args...)
    discrete_range_lcdf(args...)
    discrete_range_lccdf(args...)
    student_t_cdf(args...)

    @juliacompat append_row(x, y, z, args...) = append_row(append_row(x, y), z, args...)
    @juliacompat append_col(x, y, z, args...) = append_col(append_col(x, y), z, args...)

    # hcat for building a design matrix from column vectors. Narrow on purpose:
    # Stan's matrix / array-of-vectors / array-of-ints are distinct types, so
    # add more signatures only as sbimpl (or other callers) actually need them.
    @juliacompat hcat(x::vector[n])::matrix[n,1] = to_matrix(x, n, 1)
    @juliacompat hcat(x::vector[n], y::vector[n])::matrix[n,2] = append_col(x, y)
    @juliacompat hcat(x::matrix[m, n], y::vector[m])::matrix[m, n+1] = append_col(x, y)
    @juliacompat hcat(x, y, z, args...) = hcat(hcat(x, y), z, args...)

    # reshape: vector -> matrix only, matching Base.reshape(v, m, k) with fully
    # specified dims. Grow as needed.
    @juliacompat reshape(v::vector[n], m::int, k::int)::matrix[m, k] = to_matrix(v, m, k)

    # `jmap(f, x)` — element-wise map whose output CONTAINER is inferred from
    # `f`'s per-element return type (`typeof(f(x[1]))`): an `int`-returning `f`
    # over an `int` array gives `array[] int`, a `real`-returning `f` gives
    # `vector[n]`. A single definition therefore covers both element kinds — no
    # separate real `map` / int `imap` (`ibroadcasted`) variants are needed
    # (prong 3). `x::anything[n]` accepts a `vector` / `row_vector` / `array[] T`
    # (1-dim); `jbroadcasted` remains the arbitrary-arity elementwise construct.
    @juliacompat jmap(f, x::anything[n]) = begin
        rv::typeof(f(x[1]))[n]
        for i in 1:n
            rv[i] = f(x[i])
        end
        rv
    end

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
    neg_binomial_lpmf(args...)
    neg_binomial_2_log_lpmf(args...)
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
    neg_binomial_2_log_rng(::real, ::real)::int
    neg_binomial_2_log_rng(::vector[n], ::real)::int[n]
    neg_binomial_2_log_rng(::real, ::vector[n])::int[n]
    neg_binomial_2_log_rng(::vector[n], ::vector[n])::int[n]
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
    skew_double_exponential_rng(::real, ::real, ::real)::real
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
    lchoose(n::real, k::real)::real
    inc_beta(a::real, b::real, x::real)::real
    gamma_p(a::real, x::real)::real
    gamma_q(a::real, x::real)::real
    owens_t(h::real, a::real)::real
    bessel_first_kind(v::int, x::real)::real
    bessel_second_kind(v::int, x::real)::real
    modified_bessel_first_kind(v::int, x::real)::real
    modified_bessel_second_kind(v::int, x::real)::real
    # Log space, so a large `z` does not overflow; the order `v` may be any real.
    log_modified_bessel_first_kind(v::real, z::real)::real
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
    @juliacompat merge_missing(y_obs::vector[n_obs], y_mis::vector[n_mis], ii_obs::int[n_obs], ii_mis::int[n_mis])::vector[n_obs+n_mis] = begin
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

# --- Sequential marginalization ------------------------------------------------
# `sequential_marginalize` is the general "integrate out a sequential latent" combinator: a
# forward filter over an observation series `y`, taking user PREDICT / OBSERVE /
# SIMULATE functions rather than fixed linear-algebra objects. Given an initial
# belief `b0` and
#   predict  :: belief -> belief             one-step state prediction
#   observe  :: (belief, yₜ) -> (belief, ℓₜ) condition on yₜ; ℓₜ = log p(yₜ|y₁:ₜ₋₁)
#   simulate :: belief -> (belief, yₜ)       draw yₜ from the one-step predictive
# it provides the marginal log-density ∑ₜ ℓₜ (`_lpdf`), the per-observation
# leave-future-out conditionals [ℓ₁ … ℓ_T] (`_lpdfs`) and forward-simulated
# predictive draws (`_rng`). `belief` may be any value (scalar, tuple, vector …)
# so long as the three functions agree on its type. Because Stan has no
# first-class functions the closures are resolved at trace time and inlined into
# one generated UDF per call site (the capture-dedup in `func_args` /
# `expand_call_args` lets predict/observe/simulate share captured parameters).
# A belief whose SIZE derives from a captured matrix (e.g. a Kalman `(m,P)` tuple
# where `m = A*m` is sized by the captured `A`) is fully supported: the hoisted
# capture's dims are materialized at the UDF body top (`int A_m = dims(A)[1];`,
# `functions.jl` `_capture_dim_binds`), so an EKF / matrix-Kalman expressed
# through this HOF transpiles and compiles. (The matrix-parameterized `kalman`
# family below remains the ergonomic shortcut for the linear-Gaussian case.)
@deffun @stanonly begin
    @lpxf sequential_marginalize_lpdf(y::anything[T], b0, predict, observe, simulate)::real =
        sum(sequential_marginalize_lpdfs(y, b0, predict, observe, simulate))
    sequential_marginalize_lpdfs(y::anything[T], b0, predict, observe, simulate)::vector[T] = begin
        # Internal locals are `sm_`-prefixed so they never collide with a hoisted
        # closure capture: predict/observe/simulate close over model variables
        # whose names the user chose, and those captures become positional Stan
        # parameters of THIS generated UDF (func_args). A bare `r`/`b`/`t` here
        # would clash with a capture of the same name ("Identifier r is already
        # in use"). The `sm_` prefix keeps the body hygienic against any
        # realistic capture name.
        sm_ll::vector[T]
        sm_b = b0
        for sm_t in 1:T
            sm_bp = predict(sm_b)
            sm_step = observe(sm_bp, y[sm_t])
            sm_b = sm_step[1]
            sm_ll[sm_t] = sm_step[2]
        end
        sm_ll
    end
    # Explicit `for sm_t in 1:T` loop — empty input runs zero iterations, no
    # guard (audit todo 19n8abc). Same for the `kalman` / `hmm_forward` draws.
    sequential_marginalize_rng(vector[T], b0, predict, observe, simulate)::vector[T] = begin
        # See `_lpdfs` above: `sm_`-prefixed locals stay clear of hoisted captures.
        sm_yy::vector[T]
        sm_b = b0
        for sm_t in 1:T
            sm_bp = predict(sm_b)
            sm_step = simulate(sm_bp)
            sm_b = sm_step[1]
            sm_yy[sm_t] = sm_step[2]
        end
        sm_yy
    end
end

# --- Kalman filter (linear-Gaussian) -------------------------------------------
# Matrix-parameterized convenience family with two observation shapes:
#   SCALAR obs      `y ~ kalman(m0, P0, A, Q, c, r)` — yₜ = c'·zₜ + N(0, r),
#                    c::vector[K], r::real, y::vector[T].
#   MULTIVARIATE obs `y ~ kalman(m0, P0, A, Q, C, R)` — yₜ = C·zₜ + N(0, R),
#                    C::matrix[d,K], R::matrix[d,d], y::matrix[T,d] (row t = yₜ).
# zₜ = A·zₜ₋₁ + N(0, Q). Declared matrix dims keep the state size in UDF scope
# (unlike the closure route above). Returns the exact marginal log-density;
# `_lpdfs` gives the per-step one-step-ahead conditionals and `_rng`
# forward-simulates the series.
@deffun @stanonly begin
    @lpxf kalman_lpdf(y::vector[T], m0::vector[K], P0::matrix[K,K],
                      A::matrix[K,K], Q::matrix[K,K], c::vector[K], r::real)::real =
        sum(kalman_lpdfs(y, m0, P0, A, Q, c, r))
    kalman_lpdfs(y::vector[T], m0::vector[K], P0::matrix[K,K],
                 A::matrix[K,K], Q::matrix[K,K], c::vector[K], r::real)::vector[T] = begin
        ll::vector[T]
        m = m0
        P = P0
        for t in 1:T
            mp = A * m
            Pp = quad_form_sym(P, A') + Q
            yhat = c' * mp
            S = quad_form_sym(Pp, c) + r
            Kg = Pp * c / S
            m = mp + Kg * (y[t] - yhat)
            P = Pp - Kg * (c' * Pp)
            ll[t] = normal_lpdf(y[t], yhat, sqrt(S))
        end
        ll
    end
    kalman_rng(vector[T], m0::vector[K], P0::matrix[K,K],
               A::matrix[K,K], Q::matrix[K,K], c::vector[K], r::real)::vector[T] = begin
        yy::vector[T]
        m = m0
        P = P0
        for t in 1:T
            mp = A * m
            Pp = quad_form_sym(P, A') + Q
            yhat = c' * mp
            S = quad_form_sym(Pp, c) + r
            yt = normal_rng(yhat, sqrt(S))
            Kg = Pp * c / S
            m = mp + Kg * (yt - yhat)
            P = Pp - Kg * (c' * Pp)
            yy[t] = yt
        end
        yy
    end
end

# Multivariate-observation overloads of `kalman` (C::matrix[d,K], R::matrix[d,d];
# y::matrix[T,d], row t = yₜ). `@lhs` — not `@lpxf` — registers the base tracetype
# for this signature; the scalar block above already registered kalman's
# lpxf/rng/likelihood dispatch hooks, and one @lpxf per base name is the limit.
# Multiple dispatch selects this overload from the matrix argument shapes.
@deffun @stanonly begin
    @lhs kalman_lpdf(y::matrix[T,d], m0::vector[K], P0::matrix[K,K],
                     A::matrix[K,K], Q::matrix[K,K], C::matrix[d,K], R::matrix[d,d])::real =
        sum(kalman_lpdfs(y, m0, P0, A, Q, C, R))
    kalman_lpdfs(y::matrix[T,d], m0::vector[K], P0::matrix[K,K],
                 A::matrix[K,K], Q::matrix[K,K], C::matrix[d,K], R::matrix[d,d])::vector[T] = begin
        ll::vector[T]
        m = m0
        P = P0
        for t in 1:T
            mp = A * m
            Pp = quad_form_sym(P, A') + Q
            yhat = C * mp
            S = quad_form_sym(Pp, C') + R
            Kg = mdivide_right_spd(Pp * C', S)
            m = mp + Kg * (row(y, t)' - yhat)
            P = Pp - Kg * C * Pp
            ll[t] = multi_normal_lpdf(row(y, t)', yhat, S)
        end
        ll
    end
    kalman_rng(matrix[T,d], m0::vector[K], P0::matrix[K,K],
               A::matrix[K,K], Q::matrix[K,K], C::matrix[d,K], R::matrix[d,d])::matrix[T,d] = begin
        yy::matrix[T,d]
        m = m0
        P = P0
        for t in 1:T
            mp = A * m
            Pp = quad_form_sym(P, A') + Q
            yhat = C * mp
            S = quad_form_sym(Pp, C') + R
            yt = multi_normal_rng(yhat, S)
            Kg = mdivide_right_spd(Pp * C', S)
            m = mp + Kg * (yt - yhat)
            P = Pp - Kg * C * Pp
            yy[t,:] = yt'
        end
        yy
    end
end

# --- Hidden Markov model (discrete latent states) -------------------------------
# `y ~ hmm_forward(rho, Gamma, emit, emit_rng)` — forward-filter marginal
# likelihood for a K-state HMM (user decision `07j39xa`, option A). The emission
# is USER code: `emit(yₜ)::vector[K]` returns the per-state log-density row,
# closing over emission parameters like any HOF closure. An HMM couples each
# observation to the parameters through that user-chosen emission, so there is
# no fixed linear-algebra object to take as args (the `kalman` mold does not
# fit). The package owns the fixed forward recursion over the filtered state
# probs `alpha::vector[K]`, in log space via `log_sum_exp`; `emit_rng(k)`
# draws `yₜ | zₜ = k` for the `_rng` forward-simulation leg (`categorical_rng`
# draws the state, `one_hot_vector` re-points the belief at the draw). `K`
# unifies from the declared `rho::vector[K]` / `Gamma::matrix[K,K]` dims (the
# kalman pattern — no capture scoping work). `Gamma` is row-stochastic
# (`Gamma[m,n] = p(zₜ=n | zₜ₋₁=m)`), matching Stan's `hmm_marginal` convention.
# Internal locals are `hmm_`-prefixed (the `sm_` hygiene rationale in
# `sequential_marginalize`: a capture named `r`/`b`/`t` must never collide).
# DISCRETE observations use the SAME bare spelling (`y ~ hmm_forward(…)`): Stan
# resolves a bare `~` to the `_lpdf`/`_lpmf` specialization by the variate type
# itself. The `hmm_forward_lpmf` wrapper below exists so the fetch can pull the
# `_lpmf` specialization INSTEAD of the `_lpdf` one for int lhs (a
# closure-specialised `_lpdf` wrapper with an int first arg is illegal Stan;
# see `_dual_lpmf_call`, lpxf_builtin.jl). Plain `@deffun`, deliberately NOT
# `@lpxf` (that would overwrite the base hooks). The `_lpdfs` leg is shared and
# obs-generic (`anything[T]`); the `int[T]` rng overload serves the GQ draw.
@deffun @stanonly begin
    @lpxf hmm_forward_lpdf(y::anything[T], rho::vector[K], Gamma::matrix[K,K],
                           emit, emit_rng)::real =
        sum(hmm_forward_lpdfs(y, rho, Gamma, emit, emit_rng))
    hmm_forward_lpmf(y::int[T], rho::vector[K], Gamma::matrix[K,K],
                     emit, emit_rng)::real =
        sum(hmm_forward_lpdfs(y, rho, Gamma, emit, emit_rng))
    hmm_forward_lpdfs(y::anything[T], rho::vector[K], Gamma::matrix[K,K],
                      emit, emit_rng)::vector[T] = begin
        hmm_ll::vector[T]
        hmm_alpha = rho
        for hmm_t in 1:T
            hmm_ap = Gamma' * hmm_alpha
            hmm_joint = log(hmm_ap) + emit(y[hmm_t])
            hmm_ell = log_sum_exp(hmm_joint)
            hmm_alpha = exp(hmm_joint - hmm_ell)
            hmm_ll[hmm_t] = hmm_ell
        end
        hmm_ll
    end
    hmm_forward_rng(vector[T], rho::vector[K], Gamma::matrix[K,K],
                    emit, emit_rng)::vector[T] = begin
        hmm_yy::vector[T]
        hmm_alpha = rho
        for hmm_t in 1:T
            hmm_ap = Gamma' * hmm_alpha
            hmm_z = categorical_rng(hmm_ap)
            hmm_yt = emit_rng(hmm_z)
            hmm_alpha = one_hot_vector(rows(rho), hmm_z)
            hmm_yy[hmm_t] = hmm_yt
        end
        hmm_yy
    end
    hmm_forward_rng(int[T], rho::vector[K], Gamma::matrix[K,K],
                    emit, emit_rng)::int[T] = begin
        hmm_yy::int[T]
        hmm_alpha = rho
        for hmm_t in 1:T
            hmm_ap = Gamma' * hmm_alpha
            hmm_z = categorical_rng(hmm_ap)
            hmm_yt = emit_rng(hmm_z)
            hmm_alpha = one_hot_vector(rows(rho), hmm_z)
            hmm_yy[hmm_t] = hmm_yt
        end
        hmm_yy
    end
end

# Elementwise vectorised-lpdf/lpmf companions — one top-level @eval @deffun loop
# (mirrors the *_rng loop below). `deffun` maps over a begin-block statement-wise,
# so lifting these out of the @deffun block above is equivalent; `$(params...)`
# preserves each dist's exact param names so emitted Stan is byte-identical. Each
# dist gets a scalar-passthrough fallback + an `obs::anything[n]` broadcasted form.
# Joint dists (multi_normal/dirichlet), the matrix lkj form, the args...-splat
# forms (bernoulli/binomial), the GLM + ordered_logistic custom bodies, and dummy
# stay hand-written in the @deffun block above.
#
# `_elementwise_pointwise` marks a companion that broadcasts its scalar density
# over the observation (`jbroadcasted`, which aligns every array argument with
# it), so its family treats the elements independently. An exchangeable
# ordered prior relies on it (`_exchangeable_prior`, passes.jl). A family with
# a shared or row-structured argument (`categorical(p)`, `ordered_logistic(eta,
# c)`, the GLMs, a custom `@lpxf` family) is not one.
_elementwise_pointwise(f) = false
for lpmfs in (bernoulli_lpmfs, bernoulli_logit_lpmfs, binomial_lpmfs, binomial_logit_lpmfs)
    @eval _elementwise_pointwise(::typeof($lpmfs)) = true
end
for (base, params) in (
    (:normal_lpdf, (:loc, :scale)),
    (:cauchy_lpdf, (:loc, :scale)),
    (:lognormal_lpdf, (:loc, :scale)),
    (:gamma_lpdf, (:alpha, :beta)),
    (:beta_lpdf, (:alpha, :beta)),
    (:exponential_lpdf, (:rate,)),
    (:uniform_lpdf, (:lo, :hi)),
    (:neg_binomial_lpmf, (:alpha, :beta)),
    (:neg_binomial_2_lpmf, (:mu, :phi)),
    (:neg_binomial_2_log_lpmf, (:eta, :phi)),
    (:poisson_lpmf, (:lambda,)),
    (:poisson_log_lpmf, (:alpha,)),
    (:inv_gamma_lpdf, (:alpha, :beta)),
    (:double_exponential_lpdf, (:mu, :sigma)),
    (:logistic_lpdf, (:mu, :sigma)),
    (:weibull_lpdf, (:alpha, :sigma)),
    (:gumbel_lpdf, (:mu, :beta)),
    (:chi_square_lpdf, (:nu,)),
    (:skew_normal_lpdf, (:mu, :sigma, :alpha)),
    (:skew_double_exponential_lpdf, (:mu, :sigma, :tau)),
    (:frechet_lpdf, (:alpha, :sigma)),
    (:rayleigh_lpdf, (:sigma,)),
    (:loglogistic_lpdf, (:alpha, :beta)),
    (:von_mises_lpdf, (:mu, :kappa)),
    (:inv_chi_square_lpdf, (:nu,)),
    (:scaled_inv_chi_square_lpdf, (:nu, :sigma)),
    (:exp_mod_normal_lpdf, (:mu, :sigma, :lambda)),
    (:pareto_lpdf, (:y_min, :alpha)),
    (:pareto_type_2_lpdf, (:mu, :lambda, :alpha)),
    (:beta_binomial_lpmf, (:trials, :alpha, :beta)),
    (:student_t_lpdf, (:nu, :loc, :scale)),
)
    lpdfs = Symbol(base, :s)
    @eval @deffun @stanonly $lpdfs(args...) = $base(args...)
    @eval @deffun @stanonly $lpdfs(obs::anything[n], $(params...)) = jbroadcasted($lpdfs, obs, $(params...))
    @eval _elementwise_pointwise(::typeof($lpdfs)) = true
end

# --- Ragged vector/matrix usertypes + Julia-dispatched accessors -------------
# A ragged vector is conceptually `Vector{<:AbstractVector{<:Real}}` — a
# variable-length collection of real subvectors. A RaggedMatrix uses the same
# flat-memory scheme for varying matrix shapes and carries per-group row/column
# sizes. SLIC encodes either as a tagged ntup with fields:
#   mem  :: vector[total]   — concatenation of all subvectors
#   ends :: int[n_groups]   — inclusive 1-based end index of each subvector
# Standard Julia dispatch on the `RaggedVector` tag drives `length(rv)`
# / `rv[i]` so users don't need bespoke `ragged_*` accessors.
@usertype struct RaggedVector
    mem  :: vector
    ends :: int[]
end
@usertype struct RaggedMatrix
    mem  :: vector
    ends :: int[]
    rows :: int[]
    cols :: int[]
end
# Built-in usertypes get aliased into the `builtin` submodule so SLIC's
# `_is_builtin_name` lookup picks them up at any user `@slic` site —
# same convention used for `vector`/`real`/etc. above.
@eval builtin const RaggedVector = $RaggedVector
@eval builtin const RaggedMatrix = $RaggedMatrix

# Ragged DATA (`Vector{Vector{<:Real}}`) → nominal `RaggedVector` at ingest, so it is a
# first-class indexable container in ANY model body (`y[g]` slices to the g-th group
# vector), not only inside a plate. Both components are DATA, so a materialised
# `tuple(vector, array[] int)` data var is legal — the compile-time-pairing dance
# (above) exists only to avoid an int *parameter* tuple. Because
# `RaggedVector <: types.usertype <: types.ntup` with `arg_types` keyed `(:mem, :ends)`,
# every existing certified-ntup consumer (`_ragged_group_arg`, `_plate_is_ragged_iterable`)
# keeps matching; the NEW `y[g]`-anywhere capability comes from `RaggedVector`'s getindex
# dispatch (a bound data name is not a construction → the tuple-representation UDF).
# (decision 2026-07-17T00-14-01-598-1g0cf6y, approach B — unify data/param ragged carriers.)
stan_type(expr, value::AbstractVector{<:AbstractVector{<:Real}}; kwargs...) =
    _ragged_data_type(RaggedVector, expr, to_ragged(value); kwargs...)
# Each component of the encoded `r` is ordinary data (`<expr>_mem`, `<expr>_ends`, …).
_ragged_data_type(T, expr, r::NamedTuple; kwargs...) = StanType(T, tuple();
    arg_types=NamedTuple{keys(r)}(map((k, v) -> stan_type(Symbol(expr, "_", k), v), keys(r), values(r))),
    value=r, kwargs...,
)

# `RaggedMatrix` extends the same flat-memory representation to matrix-valued
# groups. `ends` indexes each flattened matrix, while data-qualified `rows` and
# `cols` reconstruct the selected group with `to_matrix`. Matrix DATA
# (`Vector{<:AbstractMatrix{<:Real}}`, e.g. one design matrix per group) ingests as
# one, so `X[g]` is the g-th matrix anywhere (snag `ragged-matrix-da-163255e9`).
stan_type(expr, value::AbstractVector{<:AbstractMatrix{<:Real}}; kwargs...) =
    _ragged_data_type(RaggedMatrix, expr, to_ragged(value); kwargs...)

# A `RaggedVector` is a COMPILE-TIME pairing of its two components, never a
# materialised Stan tuple. Reason: when `mem` is parameter-derived
# (transformed parameters) and `ends` is data — the ragged constrained
# parameter case — the tuple `(mem, ends)` would land in `transformed
# parameters` as `tuple(vector, array[] int)`, and Stan forbids an
# integer-valued (transformed) parameter (the `ends` component):
# "(Transformed) Parameters cannot be integers." Bundling `mem` with `ends`
# also poisons every group-size expression, since `ragged_length(rv, i)`
# would carry `mem`'s parameter qual into a Stan size declaration.
#
# So the constructor binds VERBATIM (emitting no Stan statement, like a
# closure) and `length(rv)` / `rv[i]` / `rv.field` lower to expressions over
# the SEPARATE `mem` (vector) and `ends` (int[]) StanExprs. Every offset and
# group length then references only `ends` (data), keeping them data-qualified
# — so the per-group slice `rv[g] = mem[start(g):end(g)]` is valid Stan (a
# parameter vector sliced by data indices). The all-data case (both components
# data) lowers identically, just landing in `transformed data`.

# `ends`-array-taking ragged offset accessors: data-computable (the offsets
# depend only on the data `ends` array), so they stay legal inside Stan size
# declarations. Overloads of the legacy `x::ntup` accessors above — additive,
# so no `@builtin_module` manifest edit (and no live-app const wedge, §R13).
@deffun begin
    @juliacompat ragged_start(ends::int[k], i::int)::int = if i == 1
        1
    else
        1 + ends[i-1]
    end
    @juliacompat ragged_end(ends::int[k], i::int)::int = ends[i]
    # Data-qualified group length: `ragged_length(ends, g)` is legal in a Stan
    # size declaration, so a downstream `z::vector[ragged_length(ends, g)]` param
    # can be sized by a group. (`length(rv[g])` would instead be `num_elements`
    # of a parameter-valued slice, which — like `length(<any param vector>)` —
    # is not folded to a data size, §R9; size from `ends` for a top-level decl.)
    @juliacompat ragged_length(ends::int[k], i::int)::int = ragged_end(ends, i) - ragged_start(ends, i) + 1
end

# Tuple-representation accessors (retained). These fire when a RaggedVector is
# a genuine Stan tuple VALUE — i.e. a `@deffun`/sub-model PARAMETER of type
# RaggedVector, where `rv` is a bound name (not a construction). There the tuple
# is a legal function-argument type (Stan forbids an int tuple only as a
# top-level (transformed) parameter DECLARATION, not as a function arg), so
# passing the whole RaggedVector into a function keeps its size info intact.
@deffun begin
    Base.length(rv::RaggedVector)::int = size(rv.ends)
    Base.lastindex(rv::RaggedVector)::int = size(rv.ends)
    Base.getindex(rv::RaggedVector, i::int)::typeof(rv.mem[1])[ragged_length(rv, i)] =
        rv.mem[ragged_start(rv, i):ragged_end(rv, i)]
    Base.length(rv::RaggedMatrix)::int = size(rv.ends)
    Base.lastindex(rv::RaggedMatrix)::int = size(rv.ends)
    Base.getindex(rv::RaggedMatrix, i::int)::matrix[rv.rows[i],rv.cols[i]] =
        to_matrix(
            rv.mem[ragged_start(rv, i):ragged_end(rv, i)],
            rv.rows[i],
            rv.cols[i],
        )
    # Downstream cast: view a ragged flat `(mem, ends)` — e.g. a ragged `plate`
    # result — as a dense `matrix[K, N]`, column `i` = group `i`, WHEN every group
    # is the same length `K`. A plate returns ragged by default (its cell width is
    # not statically fixed); `as_matrix` is how the author ASSERTS uniformity so the
    # result supports matrix arithmetic (`as_matrix(pred) * w`). The per-group
    # equal-length loop `reject`s a genuinely ragged value at runtime; `mem` is
    # group-concatenated, so column-major `to_matrix` lands each group in its own
    # column. `ends` is data, so the `matrix[ragged_length(ends,1), n]` result size
    # is data-qualified. The user-facing 1-arg `as_matrix(rv::RaggedVector)` is
    # lowered to this 2-arg impl by the inline hook below. Snag
    # plate-return-tup-5c3fa1c8 (BRM); NO auto-lowering — this is the explicit
    # downstream override the plate contract intends.
    as_matrix(mem::vector[m], ends::int[n])::matrix[ragged_length(ends, 1), n] = begin
        k = ragged_length(ends, 1)
        for i in 1:n
            @stan_assert ragged_length(ends, i) == k "as_matrix: the ragged value is not a matrix — group i has a different length than group 1; every group must be the same length to view it as a matrix[K, N]. Keep it ragged (index per group with rv[g] / the descriptor's segments) if the widths genuinely differ."
        end
        to_matrix(mem, k, n)
    end
end

# Group selection `rv[idx]` and group reductions `sum(rv)` / `mean(rv)` over a
# ragged value's flat `(mem, ends)`, lowered by the hooks below. A RaggedVector is a
# vector of groups, as Julia's vector of vectors: `rv[idx]` is the groups `idx`, and
# `sum(rv)` adds the groups elementwise (`rv[1] + rv[2] + …`). The selection offsets
# depend only on the data `ends` and `idx`, so they size declarations and are
# evaluated in Julia by the descriptor: `@juliacompat`. Snag selecting-severa-027a8191.
@deffun begin
    # Length of the selected group `g`, which must be one of the groups.
    @juliacompat ragged_select_length(ends::int[n], g::int)::int = begin
        @stan_assert g >= 1 && g <= n "RaggedVector selection: rv[idx] selects groups 1 to length(rv), and idx holds a value outside that range."
        ragged_length(ends, g)
    end
    # Number of entries in the groups `idx`.
    @juliacompat ragged_select_total(ends::int[n], idx::int[k])::int = begin
        total = 0
        for j in 1:k
            total = total + ragged_select_length(ends, idx[j])
        end
        total
    end
    # Inclusive ends of the groups `idx`, laid end to end: the selection's own ends.
    @juliacompat ragged_select_ends(ends::int[n], idx::int[k])::int[k] = begin
        rv::int[k]
        total = 0
        for j in 1:k
            total = total + ragged_select_length(ends, idx[j])
            rv[j] = total
        end
        rv
    end
    # Positions in `mem` of the groups `idx`, laid end to end: `mem[...]` gathers them.
    @juliacompat ragged_select_index(ends::int[n], idx::int[k])::int[ragged_select_total(ends, idx)] = begin
        total = ragged_select_total(ends, idx)
        rv::int[total]
        pos = 0
        for j in 1:k
            for e in ragged_start(ends, idx[j]):ragged_end(ends, idx[j])
                pos = pos + 1
                rv[pos] = e
            end
        end
        rv
    end
    # `rv[1] + rv[2] + … + rv[n]`; every group must have the length of group 1.
    ragged_sum(mem::vector[m], ends::int[n])::vector[ragged_length(ends, 1)] = begin
        k = ragged_length(ends, 1)
        rv = rep_vector(0.0, k)
        for i in 1:n
            @stan_assert ragged_length(ends, i) == k "sum: the RaggedVector's groups have different lengths. sum(rv) adds the groups elementwise (rv[1] + rv[2] + ...), so every group must have the same length; sum(rv.mem) is the total of every entry."
            rv = rv + mem[ragged_start(ends, i):ragged_end(ends, i)]
        end
        rv
    end
    ragged_sum(mem::int[m], ends::int[n])::int[ragged_length(ends, 1)] = begin
        k = ragged_length(ends, 1)
        rv = rep_array(0, k)
        for i in 1:n
            @stan_assert ragged_length(ends, i) == k "sum: the RaggedVector's groups have different lengths. sum(rv) adds the groups elementwise (rv[1] + rv[2] + ...), so every group must have the same length; sum(rv.mem) is the total of every entry."
            lo = ragged_start(ends, i)
            for j in 1:k
                rv[j] = rv[j] + mem[lo + j - 1]
            end
        end
        rv
    end
    ragged_mean(mem::anything[m], ends::int[n])::vector[ragged_length(ends, 1)] =
        to_vector(ragged_sum(mem, ends)) / n
end

# Constructors bind ragged StanExprs verbatim into `info` (mirrors the closure
# verbatim-bind) so their component StanExprs survive, and emit no Stan
# declaration (`nothing` → `forward!(::BlockExpr)` skips them).
forward!(x::AssignmentExpr{Symbol,<:StanExpr2{<:RaggedVector}}; info) = begin
    name, rhs = x.args
    name in keys(info) && _is_submodel_info(info) && return nothing
    name in keys(info) && error(
        "RaggedVector: rebinding `$name` is not supported — a RaggedVector is a " *
        "SLIC-side compile-time pairing of its (mem, ends) components."
    )
    info[name] = rhs
    nothing
end
forward!(x::AssignmentExpr{Symbol,<:StanExpr2{<:RaggedMatrix}}; info) = begin
    name, rhs = x.args
    name in keys(info) && _is_submodel_info(info) && return nothing
    name in keys(info) && error(
        "RaggedMatrix: rebinding `$name` is not supported — a RaggedMatrix is a " *
        "SLIC-side compile-time pairing of its (mem, ends, rows, cols) components."
    )
    info[name] = rhs
    nothing
end
# Defensive: a ragged StanExpr must never be re-forwarded as a bare symbol
# (mirrors the closure `StanExpr{Symbol,...}` passthrough).
forward!(x::StanExpr{Symbol,<:StanType{<:RaggedVector}}; info) = x
forward!(x::StanExpr{Symbol,<:StanType{<:RaggedMatrix}}; info) = x

# A ragged StanExpr is a COMPILE-TIME CONSTRUCTION iff its `expr` is the
# constructor `CanonicalExpr` — the model-body case a verbatim-bind produces.
# A ragged *parameter* inside a `@deffun` body is
# instead a bound Symbol and stays a real Stan tuple: the accessor hooks below
# fire ONLY for constructions and fall through (to the tuple-representation UDFs
# above) otherwise. This lets either ragged carrier be passed to a function
# without losing its size info.
_is_ragged_construction(rv::StanExpr) = expr(rv) isa CanonicalExpr
_ragged_mem(rv::StanExpr) = expr(rv).args[1]
_ragged_ends(rv::StanExpr) = expr(rv).args[2]
_ragged_rows(rv::StanExpr) = expr(rv).args[3]
_ragged_cols(rv::StanExpr) = expr(rv).args[4]

# Distribution arguments in a compiler-owned ragged prior loop are mapped only
# when their carrier is itself ragged. This keeps shared scalar/dense arguments
# shared and makes e.g. `p::simplex[Ks] ~ dirichlet(alpha)` index a ragged
# `alpha` as `alpha[g]` without inventing ambiguous auto-indexing for ordinary
# vectors.
expand_inline_or_trace(x::CanonicalExpr{typeof(_ragged_group_arg)}; info) = begin
    arg, g = x.args
    if center_type(arg) <: RaggedVector || center_type(arg) <: RaggedMatrix
        forward!(CanonicalExpr(getindex, arg, g); info)
    elseif center_type(arg) <: types.ntup && keys(type(arg).info.arg_types) == (:mem, :ends)
        # Fallback for a bare `(mem, ends)` ntup that is NOT the nominal
        # `RaggedVector`. Since 197f5be nested-vector DATA mints a `RaggedVector`
        # and so takes the branch above; this reaches only a hand-built
        # `(; mem, ends)` named-tuple carrier — still semantically ragged, so slice
        # this certified representation the same way. Dense vectors remain shared
        # distribution arguments because auto-indexing them would be ambiguous.
        mem = forward!(CanonicalExpr(Base.getfield, arg, 1); info)
        ends = forward!(CanonicalExpr(Base.getfield, arg, 2); info)
        lo = _trace_stan_call(builtin.ragged_start, ends, g; info)
        hi = _trace_stan_call(builtin.ragged_end, ends, g; info)
        _trace_stan_call(getindex, mem, _trace_stan_call(Colon(), lo, hi; info); info)
    else
        arg
    end
end

# --- ragged observation: generated-quantities companions ---------------------
# snag ragged-observati-6a26481b. `forward.jl` injects the gq twin loop as SOURCE,
# but source cannot spell family-companion selection (`rng_expr` / `lpxf_expr`,
# including the distribution-HOF specs in `lpxf_builtin.jl`). These two markers
# carry `(group prototype, base-family token, per-group args…)` through the
# retrace and rebuild the distribution call HERE, where the registry is in scope.
#
# Rebuilding the call rather than special-casing each family is what makes nested
# base-family HOFs (`weighted`, `conditioned`, `clamped`, …) route through the
# very same group RHS: `lpxf_expr`/`rng_expr`'s own HOF dispatch does the work.
_ragged_group_call(x::CanonicalExpr) = _normalized_hof_call(
    CanonicalExpr(_family_function(x.args[2]), x.args[3:end]...; x.kwargs...))
# The BASE family of a group RHS — one level through a distribution HOF, so
# `weighted(normal, w, …)` is judged by `normal`, exactly as `lpxf_expr` /
# `likelihood_expr` resolve it. The HOF itself has no `lpxf_expr` method.
_ragged_base_family(rhs::CanonicalExpr) = begin
    spec = _distribution_hof(head(rhs))
    isnothing(spec) ? head(rhs) : _distribution_hof_family(spec, rhs)
end
# A ragged observation's carrier and its family's probability KIND must agree: a
# discrete family (`_lpmf`) needs an INTEGER-valued group (an int-backed
# `RaggedVector` — `mem::array[] int`), a continuous family (`_lpdf`) a real
# `vector` group. A discrete draw into a real carrier (or a real draw into an
# int carrier) has no valid Stan form, so reject the MISMATCH at tracing with
# both the family and the carrier named. `ct` is the center type of the group
# slice (int for an integer ragged observation), so a discrete family over an
# integer ragged observation now lowers exactly like a real one (snag
# ragged-int-obser-771dd259 — the integer ragged carrier).
_ragged_carrier_kind(ct) = ct === types.int ? :lpmf : :lpdf
_assert_ragged_family_carrier(rhs::CanonicalExpr, ct) = begin
    family = _ragged_base_family(rhs)
    kind = _probability_kind(family)
    kind === _ragged_carrier_kind(ct) || error(
        kind === :lpmf ?
        string("Ragged observation: family `", nameof(family), "` is discrete (resolves to ",
            nameof(lpxf_expr(family)), "), but the ragged observation is real-valued (`", ct,
            "`). A discrete family needs an INTEGER-valued ragged observation — pass ",
            "`Vector{Vector{Int}}` data (its groups then back an `array[] int`).") :
        string("Ragged observation: family `", nameof(family), "` is continuous (resolves to ",
            nameof(lpxf_expr(family)), "), but the ragged observation is integer-valued. Its ",
            "predictive draw has no integer form — pass real (`Float64`) ragged data, or use ",
            "a discrete family."),
    )
    rhs
end
# A group draw is the ONE place a family's SIZED rng companion becomes mandatory:
# a ragged group is a `vector`, so the scalar `foo_rng(args…)` Stan ships natively
# is not enough. Without a `foo_rng(vector[n], args…)::vector[n]` overload the
# failure surfaces from deep inside `tracetype` as an internal
# `` `tracetype` not defined for _arg30_2::anything ``, which names neither the
# family nor what is missing. Re-raise with both.
_ragged_missing_sized_rng(rhs::CanonicalExpr, ct, detail) = begin
    family = _ragged_base_family(rhs)
    rng = try string(nameof(rng_expr(head(rhs)))) catch; string(nameof(family), "_rng") end
    error(
        "Ragged observation: family `", nameof(family), "` has no SIZED predictive ",
        "companion, so its per-group draw cannot be built. A ragged group is a Stan ",
        "`", ct, "`, not a scalar, so `", rng, "` needs the sized-token overload\n",
        "    @deffun ", rng, "(", ct, "[n], <the family's args…>)::", ct, "[n] = …\n",
        "alongside its scalar form — the same protocol every other vector-shaped ",
        "predictive draw uses (see `normal_rng(vector[n], …)` in `builtin.jl`). ", detail,
    )
end
_ragged_group_rng(token, rhs::CanonicalExpr, ct) = begin
    rv = try
        rng_expr(token, rhs)
    catch e
        _ragged_missing_sized_rng(rhs, ct, string("Underlying error: ", sprint(showerror, e)))
    end
    # A missing sized overload does not always THROW: `@deffun` dispatch can fall
    # through to an untyped/vararg form that traces to `anything`, which only fails
    # later, at the slice assignment, as an internal `` `tracetype` not defined for
    # _argNN::anything ``. Catch that shape here, where the family is still known.
    center_type(rv) === types.anything && _ragged_missing_sized_rng(
        rhs, ct, "It resolved to an untyped (`anything`) result instead.",
    )
    rv
end
expand_inline_or_trace(x::CanonicalExpr{typeof(_ragged_group_draw)}; info) = begin
    proto = x.args[1]
    ct = center_type(proto)
    rhs = _assert_ragged_family_carrier(_ragged_group_call(x), ct)
    # Sized token — the SAME protocol every other vector-shaped predictive draw
    # uses, so a custom `@lpxf` family only needs its ordinary sized
    # `foo_rng(vector[n], args…)::vector[n]` companion (stanblocks-use §8). For an
    # integer ragged observation `ct` is `int`, so the token selects the discrete
    # `foo_rng(int[n], …)::int[n]` overload (builtin.jl `$drng(int[n], …)` block).
    token = StanExpr(ct, StanType(types.tokenof{ct}, stan_size(proto); value=ct, qual=:data))
    _ragged_group_rng(token, rhs, ct)
end
expand_inline_or_trace(x::CanonicalExpr{typeof(_ragged_group_density)}; info) = begin
    rhs = _assert_ragged_family_carrier(_ragged_group_call(x), center_type(x.args[1]))
    # The AGGREGATE density of the whole group — one scalar. Never `_lpdfs`.
    lpxf_expr(x.args[1], rhs)
end

# `v[mask]` with `mask :: bool[n]` (an element-wise comparison result such as
# `cmt .== 1`) is BOOLEAN-MASK selection, which Stan has no syntax for. Lower it
# to ordinary integer indexing over the true-positions — `v[findall(mask)]` —
# intercepting before the generic `int[n]` getindex tracetype (a `bool[n]` would
# otherwise be read as an integer index list). To keep it zero-RUNTIME-cost we
# HOIST `idx = findall(mask)` into its own statement rather than inlining the
# `findall` call at the index site: a data-derived mask makes the binding
# data-qualified, so `distribute!` routes it to `transformed data` (materialised
# ONCE at startup, no per-iteration / no-AD recompute), and every use of that
# `idx` is plain integer indexing. Falls back to the inline form outside a
# statement context, and a SCALAR `bool` index (l_ndim 0) is NOT a mask — it
# falls through to normal integer indexing. This getindex overload is the ONLY
# way `bool` differs from `int`; everywhere else `bool[n]` dispatches as `int[n]`.
#
# INSIDE A PLATE loop (`_plate_context(info) !== nothing`) the hoist is SUPPRESSED and
# the `findall` stays inline: a per-cell mask is data-derived, so the hoisted `idx`
# would land in the transformed-data COPY of the plate loop that `distribute!` splits
# off — out of scope in the model COPY where the cmt-keyed obs uses it (stanc
# "Identifier not in scope"). Recomputing the index inline keeps it in the using
# block; the plate emitter mirrors this for an explicit `idx = findall(...)` local
# (forward.jl). Snag plate-cell-int.
expand_inline_or_trace(x::CanonicalExpr{typeof(getindex),<:Tuple{<:StanExpr2,<:StanExpr2{<:types.bool}}}; info) =
    _getindex_bool_mask(x; info)
_getindex_bool_mask(x::CanonicalExpr; info) = begin
    stan_ndim(type(x.args[2])) >= 1 || return fold_shape_query(_trace_stan_expr(x, info))
    idx_val = forward!(CanonicalExpr(Base.findall, x.args[2]); info)
    forward!(CanonicalExpr(getindex, x.args[1], _hoist_index(idx_val, :boolmask_idx_; info)); info)
end
# Bind an index computation to a fresh `<stem><id>` in its own statement, so a
# data-derived one lands in `transformed data` (computed once) and every use reads
# the name. With no statement context, or inside a plate loop, it stays inline (see
# above).
_hoist_index(val, stem; info) = begin
    pending = _get_inline_pending(info)
    (pending === nothing || _plate_context(info) !== nothing) && return val
    id = _next_inline_id(info)
    name = Symbol(stem, id)
    while name in keys(info)
        id = _next_inline_id(info); name = Symbol(stem, id)
    end
    push!(pending, forward!(CanonicalExpr(:(=), name, val); info))
    info[name]
end
# A ragged value indexed by a boolean mask selects groups like the integer indices
# `findall(mask)`. Breaks the tie between the mask hook above and the ragged
# getindex hooks below.
expand_inline_or_trace(x::CanonicalExpr{typeof(getindex),<:Tuple{<:StanExpr2{<:RaggedVector},<:StanExpr2{<:types.bool}}}; info) =
    _getindex_bool_mask(x; info)
expand_inline_or_trace(x::CanonicalExpr{typeof(getindex),<:Tuple{<:StanExpr2{<:RaggedMatrix},<:StanExpr2{<:types.bool}}}; info) =
    _getindex_bool_mask(x; info)

# `rv[idx]` with an integer array or a range selects GROUPS, as Julia's `vv[idx]`
# does on a vector of vectors: the groups `idx`, in that order, as a RaggedVector
# construction (so every RaggedVector consumer takes it). Its memory gathers the
# selected entries and its ends are the selected lengths laid end to end; both index
# arrays are computed from the data `ends` and `idx` and hoisted. A ragged data
# input, a plate result, an arithmetic result and a `@deffun` argument all select
# the same way. Snag selecting-severa-027a8191.
_is_group_selection(idx) = stan_ndim(type(idx)) == 1
# A range index `a:b` is not a Stan value outside an index position; spell it out.
_index_array(idx; info) = begin
    e = expr(idx)
    (e isa CanonicalExpr && head(e) isa Colon && length(e.args) == 2) || return idx
    _trace_stan_call(builtin.robust_linspaced_int_array, only(stan_size(type(idx))), e.args...; info)
end
_ragged_select(rv, idx; info) = begin
    ends = _ragged_field(rv, 2; info)
    idx = _index_array(idx; info)
    sel_ends = _hoist_index(_trace_stan_call(builtin.ragged_select_ends, ends, idx; info), :ragged_select_ends_; info)
    sel_index = _hoist_index(_trace_stan_call(builtin.ragged_select_index, ends, idx; info), :ragged_select_index_; info)
    mem = forward!(CanonicalExpr(getindex, _ragged_field(rv, 1; info), sel_index); info)
    (; idx, mem, ends=sel_ends)
end

# `rv[i]` → `mem[ragged_start(ends, i):ragged_end(ends, i)]` (a parameter vector
# sliced by data bounds). Intercepts before `tracetype`, so a construction never
# needs a tuple-taking UDF or a materialised tuple.
expand_inline_or_trace(x::CanonicalExpr{typeof(getindex),<:Tuple{<:StanExpr2{<:RaggedVector},<:StanExpr2{<:types.int}}}; info) =
    if _is_group_selection(x.args[2])
        sel = _ragged_select(x.args...; info)
        forward!(CanonicalExpr(RaggedVector, sel.mem, sel.ends); info)
    elseif _is_ragged_construction(x.args[1])
        rv, i = x.args
        record = _ragged_elementwise_record(rv)
        record === nothing || return _ragged_elementwise_group(record, i; info)
        ends = _ragged_ends(rv)
        lo = _trace_stan_call(builtin.ragged_start, ends, i; info)
        hi = _trace_stan_call(builtin.ragged_end, ends, i; info)
        _trace_stan_call(getindex, _ragged_mem(rv), _trace_stan_call(Colon(), lo, hi; info); info)
    else
        fold_shape_query(_trace_stan_expr(x, info))
    end
# `as_matrix(rv)` on a ragged CONSTRUCTION → lower to the 2-arg `as_matrix(mem, ends)`
# impl on the bare components, so the `matrix[ragged_length(ends,1), n]` result size
# references the DATA `ends` symbol directly (never the param-containing tuple, which
# stanc rejects in a size declaration). The runtime uniform-width `reject` lives in
# that impl. Snag plate-return-tup-5c3fa1c8.
expand_inline_or_trace(x::CanonicalExpr{typeof(as_matrix),<:Tuple{<:StanExpr2{<:RaggedVector}}}; info) =
    if _is_ragged_construction(x.args[1])
        rv = x.args[1]
        _trace_stan_call(builtin.as_matrix, _ragged_mem(rv), _ragged_ends(rv); info)
    else
        error(
            "as_matrix: expected a ragged `plate` result / RaggedVector construction; got a ",
            "bound RaggedVector value. Call `as_matrix` on the ragged result directly ",
            "(`as_matrix(pred)`), not on a name rebound from one."
        )
    end
# `rm[i]` → reconstruct the selected flat slice with its data-only dimensions.
expand_inline_or_trace(x::CanonicalExpr{typeof(getindex),<:Tuple{<:StanExpr2{<:RaggedMatrix},<:StanExpr2{<:types.int}}}; info) =
    if _is_group_selection(x.args[2])
        rm = x.args[1]
        sel = _ragged_select(rm, x.args[2]; info)
        rows = forward!(CanonicalExpr(getindex, _ragged_field(rm, 3; info), sel.idx); info)
        cols = forward!(CanonicalExpr(getindex, _ragged_field(rm, 4; info), sel.idx); info)
        forward!(CanonicalExpr(RaggedMatrix, sel.mem, sel.ends, rows, cols); info)
    elseif _is_ragged_construction(x.args[1])
        rv, i = x.args
        ends = _ragged_ends(rv)
        lo = _trace_stan_call(builtin.ragged_start, ends, i; info)
        hi = _trace_stan_call(builtin.ragged_end, ends, i; info)
        slice = _trace_stan_call(getindex, _ragged_mem(rv), _trace_stan_call(Colon(), lo, hi; info); info)
        nr = _trace_stan_call(getindex, _ragged_rows(rv), i; info)
        nc = _trace_stan_call(getindex, _ragged_cols(rv), i; info)
        _trace_stan_call(builtin.to_matrix, slice, nr, nc; info)
    else
        fold_shape_query(_trace_stan_expr(x, info))
    end
# `sum(rv)` / `mean(rv)` reduce over the GROUPS, as Julia's `sum`/`mean` of a vector
# of vectors do: `sum(rv) == rv[1] + rv[2] + …`, as long as each group (every group
# must have that length; Stan checks at runtime). The total of every entry is
# `sum(rv.mem)`. Snag selecting-severa-027a8191.
expand_inline_or_trace(x::CanonicalExpr{typeof(sum),<:Tuple{<:StanExpr2{<:RaggedVector}}}; info) =
    _ragged_reduce(builtin.ragged_sum, only(x.args); info)
expand_inline_or_trace(x::CanonicalExpr{typeof(builtin.mean),<:Tuple{<:StanExpr2{<:RaggedVector}}}; info) =
    _ragged_reduce(builtin.ragged_mean, only(x.args); info)
_ragged_reduce(f, rv; info) =
    _trace_stan_call(f, _ragged_field(rv, 1; info), _ragged_field(rv, 2; info); info)
# `length(rv)` / `lastindex(rv)` → number of groups = `num_elements(ends)`.
expand_inline_or_trace(x::CanonicalExpr{<:Union{typeof(length),typeof(lastindex)},<:Tuple{<:StanExpr2{<:RaggedVector}}}; info) =
    _is_ragged_construction(x.args[1]) ? _trace_stan_call(length, _ragged_ends(x.args[1]); info) : fold_shape_query(_trace_stan_expr(x, info))
expand_inline_or_trace(x::CanonicalExpr{<:Union{typeof(length),typeof(lastindex)},<:Tuple{<:StanExpr2{<:RaggedMatrix}}}; info) =
    _is_ragged_construction(x.args[1]) ? _trace_stan_call(length, _ragged_ends(x.args[1]); info) : fold_shape_query(_trace_stan_expr(x, info))
# `rv.mem` / `rv.ends` (and the matrix shape fields) → the stored component;
# field access lowers to `getfield(rv, position)`, resolved against the
# constructor arguments here.
expand_inline_or_trace(x::CanonicalExpr{typeof(Base.getfield),<:Tuple{<:StanExpr2{<:RaggedVector},<:StanExpr2{<:types.int}}}; info) =
    _is_ragged_construction(x.args[1]) ? expr(x.args[1]).args[expr(x.args[2])] : fold_shape_query(_trace_stan_expr(x, info))
expand_inline_or_trace(x::CanonicalExpr{typeof(Base.getfield),<:Tuple{<:StanExpr2{<:RaggedMatrix},<:StanExpr2{<:types.int}}}; info) =
    _is_ragged_construction(x.args[1]) ? expr(x.args[1]).args[expr(x.args[2])] : fold_shape_query(_trace_stan_expr(x, info))

# --- Per-element reads of a ragged group --------------------------------------
# Snag scan-per-step-re-621a8f94 (reported by BRM). A plate cell sees a ragged
# input as one group: the slice `ys.mem[ragged_start(ys.ends, g):ragged_end(ys.ends, g)]`
# (`_plate_input_accessor`), the same slice of a cell-local's flat memory, or
# `rv[g]` of a bound RaggedVector. A loop over the group that indexes it by a
# scalar, or bounds itself by its length, re-reads the whole group at every
# step: the C++ stanc emits copies an `int`-array slice and a gather
# (`mu[rows[g]]`) on each read, so a `@scan` over a group of `n` rows was
# O(n^2). These reads go to the flat memory instead:
#
#   group[t]       → mem[ragged_start(ends, g) + t - 1]
#   length(group)  → ragged_length(ends, g)
#   x[I][t]        → x[I[t]]         (`I` an integer index array)
#
# `x[I][t]` is exact. `group[t]` no longer checks `t` against the group's own
# length, as the flat reads of an observed group already do (`_ragged_element`
# in forward.jl). A general slice `x[a:b][t]` keeps Stan's bound check on `t`.
_same_rendering(a, b) = sprint(show, a) == sprint(show, b)
_traced_call(x, f, n) = begin
    e = x isa StanExpr ? expr(x) : x
    e isa CanonicalExpr && head(e) === f && length(e.args) == n ? e : nothing
end
# `(; mem, ends, group)` of `mem[ragged_start(ends, g):ragged_end(ends, g)]`.
_ragged_slice_group(x; info) = begin
    e = _traced_call(x, getindex, 2)
    e === nothing && return nothing
    r = e.args[2] isa StanExpr ? expr(e.args[2]) : e.args[2]
    (r isa CanonicalExpr && head(r) isa Colon && length(r.args) == 2) || return nothing
    lo = _traced_call(r.args[1], builtin.ragged_start, 2)
    hi = _traced_call(r.args[2], builtin.ragged_end, 2)
    (lo === nothing || hi === nothing) && return nothing
    all(i -> _same_rendering(lo.args[i], hi.args[i]), 1:2) || return nothing
    (; mem=e.args[1], ends=lo.args[1], group=lo.args[2])
end
# `rv[g]` of a bound RaggedVector (traced as `getindex_RaggedVector(rv, g)`); a
# construction's `rv[g]` already traces to the slice above.
_ragged_bound_group(x; info) = begin
    e = _traced_call(x, getindex, 2)
    e === nothing && return nothing
    rv, g = e.args
    (rv isa StanExpr2{<:RaggedVector} && !_is_ragged_construction(rv)) || return nothing
    (; mem=_ragged_field(rv, 1; info), ends=_ragged_field(rv, 2; info), group=g)
end
_ragged_group(x; info) = something(_ragged_slice_group(x; info), _ragged_bound_group(x; info), Some(nothing))
_is_scalar_int(x) = x isa StanExpr2{<:types.int} && stan_ndim(type(x)) == 0
_is_index_array(x) = x isa StanExpr2{<:types.int} && stan_ndim(type(x)) == 1 &&
    !(expr(x) isa CanonicalExpr && head(expr(x)) isa Colon)
expand_inline_or_trace(x::CanonicalExpr{typeof(getindex)}; info) = begin
    length(x.args) == 2 && _is_scalar_int(x.args[2]) || return invoke(expand_inline_or_trace, Tuple{CanonicalExpr}, x; info)
    sliced, t = x.args
    g = _ragged_group(sliced; info)
    g === nothing || return forward!(canonical(
        Expr(:ref, g.mem, _ragged_offset(_trace_stan_call(builtin.ragged_start, g.ends, g.group; info), t))
    ); info)
    gather = _traced_call(sliced, getindex, 2)
    gather !== nothing && _is_index_array(gather.args[2]) && return forward!(canonical(
        Expr(:ref, gather.args[1], Expr(:ref, gather.args[2], t))
    ); info)
    invoke(expand_inline_or_trace, Tuple{CanonicalExpr}, x; info)
end
expand_inline_or_trace(x::CanonicalExpr{<:Union{typeof(length),typeof(lastindex),typeof(builtin.num_elements)}}; info) = begin
    g = length(x.args) == 1 ? _ragged_group(x.args[1]; info) : nothing
    g === nothing ? invoke(expand_inline_or_trace, Tuple{CanonicalExpr}, x; info) :
        _trace_stan_call(builtin.ragged_length, g.ends, g.group; info)
end

# --- Elementwise arithmetic on RaggedVector operands --------------------------
# Snag raggedvector-ari-19d214fa (reported by BRM). A RaggedVector is flat memory
# split into groups, so ELEMENTWISE arithmetic is arithmetic on `mem` that keeps
# the group `ends`: `2.0 * rv`, `a + b * rv`, `rv / s`, `rv .^ 2`, `-rv`, and
# `rv1 + rv2` / `rv1 .* rv2` when both carry the same `ends`. The accepted forms
# mirror the scalar-array ones (`_lower_scalar_array_broadcast` /
# `_lower_scalar_array_scale`, forward.jl): binary `+`/`-` and dotted
# `.*`/`./`/`.^` between RaggedVectors and numeric scalars, unary `-`, and scalar
# scaling `s * rv` / `rv * s` / `rv / s`. Every other form — a dense operand (per
# group or per element?), `rv * rv`, `s / rv`, plain `^` — returns `nothing` and
# keeps the actionable `_reject_ragged_vector_arithmetic` floor.
#
# The result is a RaggedVector CONSTRUCTION, so every consumer of one (`rv[g]`,
# `length`, `.mem`/`.ends`, `as_matrix`, ragged observations, inline RaggedVector
# UDFs) takes it unchanged. It also records its operands, so `rv[g]` slices each
# operand by group BEFORE the arithmetic (`2.0 * x[g]`) instead of recomputing the
# whole-memory result once per group.
#
# Two RaggedVector operands must share their group ends. Identical `ends`
# expressions need no check. Otherwise differing trace-time values fail here, and
# Stan rejects at runtime too, because a `StanModel` rebinds data without
# re-tracing: the whole-object `ends` go through `ragged_common_ends`, and each
# group slice validates only its own two boundaries (`ragged_matched_group`).
@deffun begin
    ragged_common_ends(ends::int[n], other::int[m])::int[n] = begin
        @stan_assert n == m "RaggedVector arithmetic: the two operands have different numbers of groups; elementwise arithmetic needs the same group ends."
        for i in 1:n
            @stan_assert ends[i] == other[i] "RaggedVector arithmetic: the two operands have different group ends; elementwise arithmetic needs the same group ends."
        end
        ends
    end
    ragged_matched_group(ends::int[n], other::int[m], i::int)::int = begin
        @stan_assert n == m "RaggedVector arithmetic: the two operands have different numbers of groups; elementwise arithmetic needs the same group ends."
        @stan_assert ends[i] == other[i] "RaggedVector arithmetic: the two operands have different group ends; elementwise arithmetic needs the same group ends."
        if i > 1
            @stan_assert ends[i-1] == other[i-1] "RaggedVector arithmetic: the two operands have different group ends; elementwise arithmetic needs the same group ends."
        end
        i
    end
end
_is_ragged_vector_operand(a) = a isa StanExpr && center_type(a) <: RaggedVector
_is_ragged_arithmetic_scalar(a) = a isa StanExpr && _is_numeric_scalar(type(a))
_ragged_arithmetic_form(f, args) = begin
    all(a -> _is_ragged_vector_operand(a) || _is_ragged_arithmetic_scalar(a), args) || return false
    n = length(args)
    f === (-) && n == 1 && return true
    n == 2 || return false
    f === (+) || f === (-) ||
        (f isa Base.BroadcastFunction && _broadcast_op(f) in (+, -, *, /, ^)) ||
        (f === (*) && count(_is_ragged_arithmetic_scalar, args) == 1) ||
        (f === (/) && _is_ragged_arithmetic_scalar(args[2]))
end
_ragged_elementwise_record(rv) = get(type(rv).info, :ragged_elementwise, nothing)
_ragged_field(rv, i; info) = forward!(CanonicalExpr(Base.getfield, rv, i); info)
# The ends a group slice of `rv` is cut at. An arithmetic result names its first
# RaggedVector operand's ends: its own group slices validate the other operand.
_ragged_group_ends(rv; info) = begin
    record = _ragged_elementwise_record(rv)
    record === nothing ? _ragged_field(rv, 2; info) : record.group_ends
end
_same_ragged_ends(a, b) = sprint(show, a) == sprint(show, b)
_ragged_common_ends(rvs; info) = begin
    ends = map(rv -> _ragged_field(rv, 2; info), rvs)
    length(ends) == 1 && return only(ends)
    a, b = ends
    _same_ragged_ends(a, b) && return a
    hasvalue(a) && hasvalue(b) && getvalue(a) != getvalue(b) && error(
        "RaggedVector arithmetic: the operands have different group ends (`", a, "` = ",
        getvalue(a), ", `", b, "` = ", getvalue(b), "). Elementwise arithmetic between two ",
        "RaggedVectors needs the same groups; index one group with `rv[g]` to combine ",
        "groups of different sizes.",
    )
    _trace_stan_call(builtin.ragged_common_ends, a, b; info)
end
_lower_ragged_arithmetic(x::CanonicalExpr; info) = begin
    any(_is_ragged_vector_operand, x.args) || return nothing
    f = head(x)
    # `a + b + rv` / `a * b * rv` arrive n-ary: fold left into binary steps.
    (f === (+) || f === (*)) && length(x.args) > 2 && return forward!(
        CanonicalExpr(f, CanonicalExpr(f, x.args[1], x.args[2]), x.args[3:end]...); info,
    )
    _ragged_arithmetic_form(f, x.args) || return nothing
    rvs = filter(_is_ragged_vector_operand, collect(x.args))
    mems = map(a -> _is_ragged_vector_operand(a) ? _ragged_field(a, 1; info) : a, x.args)
    mem = forward!(CanonicalExpr(f, mems...); info)
    ends = _ragged_common_ends(rvs; info)
    rv = forward!(CanonicalExpr(RaggedVector, mem, ends); info)
    remake(rv; ragged_elementwise=(;
        f, operands=x.args, group_ends=_ragged_group_ends(first(rvs); info),
    ))
end
# Group `g` of an arithmetic result: the same arithmetic over each operand's group.
_ragged_elementwise_group(record, g; info) = begin
    slices = map(record.operands) do a
        _is_ragged_vector_operand(a) || return a
        ends = _ragged_group_ends(a; info)
        ag = _same_ragged_ends(ends, record.group_ends) ? g :
            _trace_stan_call(builtin.ragged_matched_group, record.group_ends, ends, g; info)
        forward!(CanonicalExpr(getindex, a, ag); info)
    end
    forward!(CanonicalExpr(record.f, slices...); info)
end

# --- EachCol / EachRow — first-class column/row VIEWS of a matrix -------------
# `EachCol(X)` / `EachRow(X)` make a matrix's columns / rows an indexable container
# everywhere (`EachCol(X)[j]` = the j-th column) and a plate iterable — the
# eachcol/eachrow analogue of RaggedVector (decision 2026-07-17T02-25-53-568-igj5dy).
# Like RaggedVector, a construction is a COMPILE-TIME view (verbatim-bound, emitting
# no wrapper tuple): `[j]` lowers directly to Stan's native `col(X, j)` / `row(X, i)`,
# `length` to `cols(X)` / `rows(X)`.
@usertype struct EachCol
    X :: matrix
end
@usertype struct EachRow
    X :: matrix
end
@eval builtin const EachCol = $EachCol
@eval builtin const EachRow = $EachRow

# Tuple-representation accessors — fire when the view is a BOUND name (a @deffun /
# submodel param), not a construction (mirrors the RaggedVector UDFs).
@deffun begin
    Base.length(ec::EachCol)::int = cols(ec.X)
    Base.lastindex(ec::EachCol)::int = cols(ec.X)
    Base.getindex(ec::EachCol, j::int)::vector[rows(ec.X)] = col(ec.X, j)
    Base.length(er::EachRow)::int = rows(er.X)
    Base.lastindex(er::EachRow)::int = rows(er.X)
    Base.getindex(er::EachRow, i::int)::row_vector[cols(er.X)] = row(er.X, i)
end

# Constructions bind verbatim (mirrors RaggedVector), emitting no Stan statement.
forward!(x::AssignmentExpr{Symbol,<:StanExpr2{<:EachCol}}; info) = begin
    name, rhs = x.args
    name in keys(info) && _is_submodel_info(info) && return nothing
    name in keys(info) && error("EachCol: rebinding `$name` is not supported — it is a compile-time view of a matrix.")
    info[name] = rhs
    nothing
end
forward!(x::AssignmentExpr{Symbol,<:StanExpr2{<:EachRow}}; info) = begin
    name, rhs = x.args
    name in keys(info) && _is_submodel_info(info) && return nothing
    name in keys(info) && error("EachRow: rebinding `$name` is not supported — it is a compile-time view of a matrix.")
    info[name] = rhs
    nothing
end
forward!(x::StanExpr{Symbol,<:StanType{<:EachCol}}; info) = x
forward!(x::StanExpr{Symbol,<:StanType{<:EachRow}}; info) = x

_is_view_construction(v::StanExpr) = expr(v) isa CanonicalExpr
_view_mat(v::StanExpr) = expr(v).args[1]

# `EachCol(X)[j]` → `col(X, j)`; `EachRow(X)[i]` → `row(X, i)` (construction case).
expand_inline_or_trace(x::CanonicalExpr{typeof(getindex),<:Tuple{<:StanExpr2{<:EachCol},<:StanExpr2{<:types.int}}}; info) =
    _is_view_construction(x.args[1]) ? _trace_stan_call(builtin.col, _view_mat(x.args[1]), x.args[2]; info) : fold_shape_query(_trace_stan_expr(x, info))
expand_inline_or_trace(x::CanonicalExpr{typeof(getindex),<:Tuple{<:StanExpr2{<:EachRow},<:StanExpr2{<:types.int}}}; info) =
    _is_view_construction(x.args[1]) ? _trace_stan_call(builtin.row, _view_mat(x.args[1]), x.args[2]; info) : fold_shape_query(_trace_stan_expr(x, info))
# `length(EachCol(X))` / `lastindex` → `cols(X)`; EachRow → `rows(X)`.
expand_inline_or_trace(x::CanonicalExpr{<:Union{typeof(length),typeof(lastindex)},<:Tuple{<:StanExpr2{<:EachCol}}}; info) =
    _is_view_construction(x.args[1]) ? _trace_stan_call(builtin.cols, _view_mat(x.args[1]); info) : fold_shape_query(_trace_stan_expr(x, info))
expand_inline_or_trace(x::CanonicalExpr{<:Union{typeof(length),typeof(lastindex)},<:Tuple{<:StanExpr2{<:EachRow}}}; info) =
    _is_view_construction(x.args[1]) ? _trace_stan_call(builtin.rows, _view_mat(x.args[1]); info) : fold_shape_query(_trace_stan_expr(x, info))
# `EachCol(X).X` / `EachRow(X).X` → the captured matrix (construction case).
expand_inline_or_trace(x::CanonicalExpr{typeof(Base.getfield),<:Tuple{<:StanExpr2{<:EachCol},<:StanExpr2{<:types.int}}}; info) =
    _is_view_construction(x.args[1]) ? expr(x.args[1]).args[expr(x.args[2])] : fold_shape_query(_trace_stan_expr(x, info))
expand_inline_or_trace(x::CanonicalExpr{typeof(Base.getfield),<:Tuple{<:StanExpr2{<:EachRow},<:StanExpr2{<:types.int}}}; info) =
    _is_view_construction(x.args[1]) ? expr(x.args[1]).args[expr(x.args[2])] : fold_shape_query(_trace_stan_expr(x, info))

# --- Sized-token rng overloads (generated via @eval @deffun) -----------------
# gq `x::T[n] ~ dist(args...)` synthesizes `dist_rng(T[n], args...)` which
# dispatches here on the token's shape. Stan's native `*_rng` broadcasts scalars
# against vector args natively, so we only need `rep_vector` in the all-scalar
# case (to give Stan a shape to work with). Our convention: output is always
# `vector[n]` / `int[n]` (not Stan's native `real[n]`), so the vector-token
# path wraps continuous rngs in `to_vector`.

# std_normal token (0-arg)
# Empty-segment guard (todo 19n8abc): the native call mixes a container with
# the scalar `1`, so the token-0 draw needs the same early return as the 2-arg
# loop below. (The zero-arg `std_normal_rng()` scalar form is unaffected.)
@deffun std_normal_rng(real[n])::real[n] = if n == 0
    rv::real[n]
    rv
else
    normal_rng(rep_vector(0, n), 1)
end
@deffun std_normal_rng(vector[n])::vector[n] = if n == 0
    rv::vector[n]
    rv
else
    to_vector(normal_rng(rep_vector(0, n), 1))
end

# 2-arg continuous families. Two semantic cases:
#   - all scalar: need `rep_vector` so Stan produces a shaped output
#   - catch-all (at least one shape-[n] container): Stan native broadcasts
# Julia dispatch picks the more-specific `(a::real, b::real)` when both scalar.
# Empty-segment guard (todo 19n8abc — generalised from snag
# sized-rng-compan-225549a9): Stan Math ≤5.3.0 sizes a vectorized draw off ALL
# args with scalars counting as size 1, so a scalar arg beside an empty
# segment runs the draw loop once over a null data pointer (SIGSEGV — proven
# for student_t under ASan). Fixed upstream in stan-dev/math@6db3b739, after
# the BridgeStan 2.9.0 bundle — until a fixed BridgeStan ships, return the
# empty draw directly when the size token is 0. The `else` branch is the exact
# previous body, so non-empty draws are unchanged. Same `if n == 0` idiom as
# `robust_linspaced_int_array` above.
for dist in (:normal, :cauchy, :lognormal, :gamma, :inv_gamma, :beta, :uniform,
             :weibull, :frechet, :double_exponential, :logistic, :gumbel,
             :pareto, :scaled_inv_chi_square, :von_mises, :loglogistic)
    drng = Symbol(dist, :_rng)
    @eval @deffun $drng(real[n],   a::real, b::real)::real[n]   = if n == 0
        rv::real[n]
        rv
    else
        $drng(rep_vector(a, n), b)
    end
    @eval @deffun $drng(vector[n], a::real, b::real)::vector[n] = if n == 0
        rv::vector[n]
        rv
    else
        to_vector($drng(rep_vector(a, n), b))
    end
    @eval @deffun $drng(real[n],   a, b)::real[n]   = if n == 0
        rv::real[n]
        rv
    else
        $drng(a, b)
    end
    @eval @deffun $drng(vector[n], a, b)::vector[n] = if n == 0
        rv::vector[n]
        rv
    else
        to_vector($drng(a, b))
    end
end

# 1-arg continuous families. Single native arg, so the draw loop is bound by
# that arg's own size — no scalar can inflate it (audit todo 19n8abc): immune
# by construction, no guard.
for dist in (:exponential, :chi_square, :inv_chi_square, :rayleigh)
    drng = Symbol(dist, :_rng)
    @eval @deffun $drng(real[n],   a::real)::real[n]   = $drng(rep_vector(a, n))
    @eval @deffun $drng(vector[n], a::real)::vector[n] = to_vector($drng(rep_vector(a, n)))
    @eval @deffun $drng(real[n],   a)::real[n]   = $drng(a)
    @eval @deffun $drng(vector[n], a)::vector[n] = to_vector($drng(a))
end

# 3-arg continuous families. Two leading-arg regimes:
#   - SCALAR leading arg (`nu::real`) — `student_t`'s dof, or a scalar location:
#     rep_vector the SECOND arg in the all-scalar case so Stan has a shape to
#     draw against; a vector second/third arg broadcasts natively.
#   - VECTOR leading arg — a per-observation LOCATION. `skew_double_exponential`
#     / `skew_normal` / `exp_mod_normal` / `pareto_type_2` in a regression emit
#     `dist(mu_vec, sigma, tau)` with a scalar OR vector scale, so the auto-GQ
#     sized-token draw is `dist_rng(<token>, vector, scale, real)`. The location
#     is already `vector[n]`, so delegate straight to native — the family's
#     `@defsig` below lists `(vector[n], vector[n], real)` /
#     `(vector[n], vector[n], vector[n])` and `(vector[n], real, real)` /
#     `(vector[n], real, vector[n])`. Without these overloads that draw matches
#     no method and the generated-quantities block fails to trace (snags
#     stanblocks-skew, sbbrmi-response-f3ed0938).
for dist in (:student_t, :skew_normal, :exp_mod_normal,
             :skew_double_exponential, :pareto_type_2)
    drng = Symbol(dist, :_rng)
    # Empty-segment guard (snag sized-rng-compan-225549a9): Stan Math ≤5.3.0
    # sizes a vectorized draw off ALL args with scalars counting as size 1,
    # so a scalar arg beside empty segments runs the draw loop once over a
    # null data pointer (SIGSEGV — proven for student_t under ASan; the
    # scalar `nu` is the whole family difference vs safe `normal_rng`).
    # Fixed upstream in stan-dev/math@6db3b739, after the BridgeStan 2.9.0
    # bundle — until a fixed BridgeStan ships, return the empty draw
    # directly when the size token is 0. The `else` branch is the exact
    # previous body, so non-empty draws are unchanged. Same `if n == 0`
    # idiom as `robust_linspaced_int_array` above.
    @eval @deffun $drng(real[n],   nu::real, a::real, b::real)::real[n]   = if n == 0
        rv::real[n]
        rv
    else
        $drng(nu, rep_vector(a, n), b)
    end
    @eval @deffun $drng(vector[n], nu::real, a::real, b::real)::vector[n] = if n == 0
        rv::vector[n]
        rv
    else
        to_vector($drng(nu, rep_vector(a, n), b))
    end
    @eval @deffun $drng(real[n],   nu::real, a, b)::real[n]   = if n == 0
        rv::real[n]
        rv
    else
        $drng(nu, a, b)
    end
    @eval @deffun $drng(vector[n], nu::real, a, b)::vector[n] = if n == 0
        rv::vector[n]
        rv
    else
        to_vector($drng(nu, a, b))
    end
    # Vector LEADING arg (per-observation location): both leading args are vectors.
    @eval @deffun $drng(real[n],   loc::vector[n], scale::vector[n], b)::real[n]   = if n == 0
        rv::real[n]
        rv
    else
        $drng(loc, scale, b)
    end
    @eval @deffun $drng(vector[n], loc::vector[n], scale::vector[n], b)::vector[n] = if n == 0
        rv::vector[n]
        rv
    else
        to_vector($drng(loc, scale, b))
    end
    # Vector LOCATION with a scalar SCALE (`y ~ dist(mu_vec, sigma, tau)` —
    # the SBBRMI regression shape, snag sbbrmi-response-f3ed0938): Stan
    # broadcasts the scalar natively, so delegate straight through like the
    # vector-vector overloads above (the family's `@defsig` lists
    # `(vector[n], real, real)` / `(vector[n], real, vector[n])`).
    @eval @deffun $drng(real[n],   loc::vector[n], scale::real, b)::real[n]   = if n == 0
        rv::real[n]
        rv
    else
        $drng(loc, scale, b)
    end
    @eval @deffun $drng(vector[n], loc::vector[n], scale::real, b)::vector[n] = if n == 0
        rv::vector[n]
        rv
    else
        to_vector($drng(loc, scale, b))
    end
end

# A matrix variate for the continuous elementwise families (snag
# matrix-parameter-94e1f96c). Stan vectorises these families over a scalar,
# `vector`, `row_vector` or 1-D array variate, never a `matrix`, so a
# `z::matrix[m, n] ~ normal(0, 1)` prior or a `Y ~ normal(mu, sigma)` matrix
# observation emitted a statement stanc rejects (`Ill-typed arguments supplied
# to function "normal": (matrix, real, real)`), and its generated-quantities
# draw had no companion at all. Each family gets three matrix overloads, whose
# arguments are scalars or matrices of the variate's own shape
# (`_check_matrix_variate_args`, lpxf_builtin.jl, refuses any other shape at
# trace time); `matrix_variate_arg` flattens a matrix argument like the variate:
#   - the density, a Stan UDF overload of the library name (as
#     `multinomial_lpmf` above), so the model block keeps `z ~ normal(0, 1);`.
#     Its body calls the family's `_lupdf`, which inherits the caller's propto
#     flag: a `~` drops the normalising constants exactly as
#     `to_vector(z) ~ normal(0, 1)` does, and an explicit `normal_lpdf(z | 0, 1)`
#     keeps them;
#   - the pointwise companion, one entry per element, laid out like the variate;
#   - the sized `_rng`, the family's vector draw reshaped column-major, which
#     is how `to_vector` lays the variate out.
# No `@lhs`: it would register a base `normal(::real, ::real)` tracetype
# returning a matrix, which every untyped `x ~ normal(0, 1)` would then pick
# up. The variate's shape always comes from a typed LHS or from the data.
@eval builtin function matrix_variate_arg end
const matrix_variate_arg = builtin.matrix_variate_arg
@deffun matrix_variate_arg(x::real, m::int, n::int)::real = x
@deffun matrix_variate_arg(x::matrix[o, p], m::int, n::int)::vector[o * p] = begin
    @stan_assert o == m && p == n "matrix variate: a matrix argument of an elementwise family must have the variate's own shape."
    to_vector(x)
end
for (dist, params) in (
    (:std_normal, ()),
    (:normal, (:loc, :scale)), (:cauchy, (:loc, :scale)), (:lognormal, (:loc, :scale)),
    (:gamma, (:alpha, :beta)), (:inv_gamma, (:alpha, :beta)), (:beta, (:alpha, :beta)),
    (:uniform, (:lo, :hi)), (:weibull, (:alpha, :sigma)), (:frechet, (:alpha, :sigma)),
    (:double_exponential, (:mu, :sigma)), (:logistic, (:mu, :sigma)),
    (:gumbel, (:mu, :beta)), (:pareto, (:y_min, :alpha)),
    (:scaled_inv_chi_square, (:nu, :sigma)), (:von_mises, (:mu, :kappa)),
    (:loglogistic, (:alpha, :beta)),
    (:exponential, (:rate,)), (:chi_square, (:nu,)), (:inv_chi_square, (:nu,)),
    (:rayleigh, (:sigma,)),
    (:student_t, (:nu, :loc, :scale)), (:skew_normal, (:mu, :sigma, :alpha)),
    (:exp_mod_normal, (:mu, :sigma, :lambda)),
    (:skew_double_exponential, (:mu, :sigma, :tau)),
    (:pareto_type_2, (:mu, :lambda, :alpha)),
)
    dlpdf, dlupdf = Symbol(dist, :_lpdf), Symbol(dist, :_lupdf)
    dlpdfs, drng = Symbol(dist, :_lpdfs), Symbol(dist, :_rng)
    @eval builtin function $dlupdf end
    @eval const $dlupdf = builtin.$dlupdf
    @eval @deffun $dlupdf(args...)::real
    @eval _matrix_variate_family(::typeof(builtin.$dist)) = true
    flat_args = [:(matrix_variate_arg($p, m, n)) for p in params]
    @eval @deffun $dlpdf(obs::matrix[m, n], $(params...))::real =
        $dlupdf(to_vector(obs), $(flat_args...))
    # `std_normal` has no vector pointwise companion to reshape (a vector
    # observation of it has no `y_likelihood` either).
    dist === :std_normal ||
        @eval @deffun $dlpdfs(obs::matrix[m, n], $(params...))::matrix[m, n] =
            to_matrix($dlpdfs(to_vector(obs), $(flat_args...)), m, n)
    @eval @deffun $drng(matrix[m, n], $(params...))::matrix[m, n] =
        to_matrix($drng(vector[m * n], $(flat_args...)), m, n)
end

# 1-arg discrete families (output is int[n]; no to_vector wrap). Single native
# arg — same immunity as the 1-arg continuous loop above (todo 19n8abc).
for dist in (:bernoulli, :bernoulli_logit, :poisson, :poisson_log)
    drng = Symbol(dist, :_rng)
    @eval @deffun $drng(int[n], p::real)::int[n] = $drng(rep_vector(p, n))
    @eval @deffun $drng(int[n], p)::int[n]       = $drng(p)
end

# 2-arg discrete families. Same scalar+container mix as the 2-arg continuous
# loop (todo 19n8abc) — same empty-segment guard, `else` branch unchanged.
for dist in (:neg_binomial, :neg_binomial_2, :neg_binomial_2_log)
    drng = Symbol(dist, :_rng)
    @eval @deffun $drng(int[n], a::real, b::real)::int[n] = if n == 0
        rv::int[n]
        rv
    else
        $drng(rep_vector(a, n), b)
    end
    @eval @deffun $drng(int[n], a, b)::int[n]             = if n == 0
        rv::int[n]
        rv
    else
        $drng(a, b)
    end
end

# binomial: N::int[n] is already a container, so Stan broadcasts scalar p natively.
# Empty-segment guard (todo 19n8abc): a scalar `p` beside an empty `N`
# replays the 2-arg-loop segfault, so both overloads early-return.
@deffun binomial_rng(int[n], N::int[n], p)::int[n] = if n == 0
    rv::int[n]
    rv
else
    binomial_rng(N, p)
end
# The lpmf also broadcasts a scalar trial count across vector observations.
# Expand it to the observation length so Stan's native RNG returns int[n]
# instead of the token call falling through to the scalar catch-all.
@deffun binomial_rng(int[n], N::int, p)::int[n] = if n == 0
    rv::int[n]
    rv
else
    binomial_rng(rep_array(N, n), p)
end

# The native GLM RNG already returns one integer per design-matrix row. The
# generated-quantities path also passes the observed int-array's sized token;
# unwrap that token while asserting the output and matrix row counts agree.
# Empty-segment guard (todo 19n8abc): a 0-row design matrix early-returns
# the empty draw instead of reaching a vectorized native over `m` rows.
@deffun bernoulli_logit_glm_rng(int[m], X::matrix[m,n], alpha, beta)::int[m] = if m == 0
    rv::int[m]
    rv
else
    bernoulli_logit_glm_rng(X, alpha, beta)
end

# The other three GLM families have no native Stan rng at all; their plain
# forms (declared with bodies above) lower to the base family's rng over
# `alpha + X * beta`. The token path just unwraps, exactly as bernoulli does.
@deffun normal_id_glm_rng(vector[m], X::matrix[m,n], alpha, beta, sigma)::vector[m] = if m == 0
    rv::vector[m]
    rv
else
    normal_id_glm_rng(X, alpha, beta, sigma)
end
@deffun normal_id_glm_rng(real[m], X::matrix[m,n], alpha, beta, sigma)::real[m] = if m == 0
    rv::real[m]
    rv
else
    to_array_1d(normal_id_glm_rng(X, alpha, beta, sigma))
end
@deffun poisson_log_glm_rng(int[m], X::matrix[m,n], alpha, beta)::int[m] = if m == 0
    rv::int[m]
    rv
else
    poisson_log_glm_rng(X, alpha, beta)
end
@deffun neg_binomial_2_log_glm_rng(int[m], X::matrix[m,n], alpha, beta, phi)::int[m] = if m == 0
    rv::int[m]
    rv
else
    neg_binomial_2_log_glm_rng(X, alpha, beta, phi)
end

# binomial_logit: Stan ships `binomial_logit_lpmf` (and a fused
# `binomial_logit_glm_lpmf`) but NO `binomial_logit_rng` in any flavour —
# `bernoulli_logit_glm_rng` is the only `*_glm_rng` in Stan Math at all. Lower
# the token-path call to `binomial_rng(N, inv_logit(eta))` so SBBRMI's
# generated_quantities for a `BinomialLogit(N, eta)` likelihood compile
# under stanc.
# Empty-segment guard (todo 19n8abc): same scalar-mix shape as `binomial_rng`
# above — `inv_logit(eta)` is scalar whenever `eta` is.
@deffun binomial_logit_rng(int[n], N::int[n], eta)::int[n] = if n == 0
    rv::int[n]
    rv
else
    binomial_rng(N, inv_logit(eta))
end
# Stan's lpmf also broadcasts a scalar trial count across vector observations.
# Give the RNG an array-valued argument so it returns one draw per observation;
# this covers both vector `eta` and the all-scalar `N`/`eta` case.
@deffun binomial_logit_rng(int[n], N::int, eta)::int[n] = if n == 0
    rv::int[n]
    rv
else
    binomial_rng(rep_array(N, n), inv_logit(eta))
end

# beta_binomial: (trials, alpha, beta). Empty-segment guard (todo 19n8abc):
# scalar shape args beside the `N` container replay the scalar-mix failure.
@deffun beta_binomial_rng(int[n], N::int[n], a::real, b::real)::int[n] = if n == 0
    rv::int[n]
    rv
else
    beta_binomial_rng(N, a, b)
end
@deffun beta_binomial_rng(int[n], N::int[n], a, b)::int[n] = if n == 0
    rv::int[n]
    rv
else
    beta_binomial_rng(N, a, b)
end
# The lpmf also broadcasts a scalar trial count across vector observations.
# Expand it to the observation length so Stan's native RNG returns int[n].
@deffun beta_binomial_rng(int[n], N::int, a, b)::int[n] = if n == 0
    rv::int[n]
    rv
else
    beta_binomial_rng(rep_array(N, n), a, b)
end

# dirichlet: native already returns vector[n]; token path just unwraps. Single
# draw sized by the one container arg — no scalar-inflated vectorized loop, so
# outside the guarded class (audit todo 19n8abc).
@deffun dirichlet_rng(vector[n], alpha::vector[n])::vector[n] = dirichlet_rng(alpha)

# categorical / categorical_logit: repeat one shared probability/logit vector.
# Explicit `for i in 1:n` loops with scalar native calls — empty input runs
# zero iterations, so no guard (audit todo 19n8abc).
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

# Plate-produced varying logits use matrix[k, n], with one categorical vector
# per observation column. Match the density overload above in auto-GQ draws.
@deffun categorical_logit_rng(int[n], eta::matrix[k,n])::int[n] = begin
    rv::int[n]
    for i in 1:n
        rv[i] = categorical_logit_rng(eta[:, i])
    end
    rv
end

# multi_normal / multi_normal_cholesky: native already returns vector[n].
# Single draws sized by the location vector alone — no scalar-inflated
# vectorized loop, so outside the guarded class (audit todo 19n8abc).
@deffun multi_normal_rng(vector[n], loc::vector[n], cov)::vector[n]          = multi_normal_rng(loc, cov)
@deffun multi_normal_cholesky_rng(vector[n], loc::vector[n], scale)::vector[n] = multi_normal_cholesky_rng(loc, scale)
# Precision-parameterized MVN: Stan Math ships no `multi_normal_prec_rng`
# (verified via `stanc --dump-stan-math-signatures`), so the sized-token draw
# routes through the covariance form. Exact: N(mu, Omega^-1) == N(mu, inv(Omega)).
@deffun multi_normal_prec_rng(vector[n], loc::vector[n], prec)::vector[n] =
    multi_normal_rng(loc, inverse(prec))
@deffun multi_student_t_rng(vector[n], nu, loc::vector[n], scale)::vector[n] =
    multi_student_t_rng(nu, loc, scale)
@deffun multi_student_t_cholesky_rng(vector[n], nu, loc::vector[n], scale)::vector[n] =
    multi_student_t_cholesky_rng(nu, loc, scale)

# Array-of-vector draws (`array[m] vector[n]`, SLIC `vector[m, n]`). The density
# side accepts the whole array, so `b::vector[m, n] ~ multi_*(…)` is one sampled
# carrier — written by hand, or produced by a plate whose single cell is a
# shared-argument multivariate prior (`_plate_vectorized_sample`, forward.jl).
# When no likelihood reaches it (a prior-only program) or it is cv-held-out, it
# is re-drawn in generated quantities, where each Stan multivariate RNG returns
# one vector: draw row by row, from a shared location or from row `i`'s own.
# `for i in 1:m` runs zero iterations for an empty array, and each native call is
# sized by its location vector alone, so outside the guarded class (todo 19n8abc).
# snag centered-ranef-p-892a7790.
@deffun begin
    multi_normal_rng(vector[m, n], loc::vector[n], cov)::vector[m, n] = begin
        rv::vector[m, n]
        for i in 1:m
            rv[i] = multi_normal_rng(loc, cov)
        end
        rv
    end
    multi_normal_rng(vector[m, n], loc::vector[m, n], cov)::vector[m, n] = begin
        rv::vector[m, n]
        for i in 1:m
            rv[i] = multi_normal_rng(loc[i], cov)
        end
        rv
    end
    multi_normal_cholesky_rng(vector[m, n], loc::vector[n], scale)::vector[m, n] = begin
        rv::vector[m, n]
        for i in 1:m
            rv[i] = multi_normal_cholesky_rng(loc, scale)
        end
        rv
    end
    multi_normal_cholesky_rng(vector[m, n], loc::vector[m, n], scale)::vector[m, n] = begin
        rv::vector[m, n]
        for i in 1:m
            rv[i] = multi_normal_cholesky_rng(loc[i], scale)
        end
        rv
    end
    multi_normal_prec_rng(vector[m, n], loc::vector[n], prec)::vector[m, n] = begin
        cov = inverse(prec)
        rv::vector[m, n]
        for i in 1:m
            rv[i] = multi_normal_rng(loc, cov)
        end
        rv
    end
    multi_normal_prec_rng(vector[m, n], loc::vector[m, n], prec)::vector[m, n] = begin
        cov = inverse(prec)
        rv::vector[m, n]
        for i in 1:m
            rv[i] = multi_normal_rng(loc[i], cov)
        end
        rv
    end
    multi_student_t_rng(vector[m, n], nu, loc::vector[n], scale)::vector[m, n] = begin
        rv::vector[m, n]
        for i in 1:m
            rv[i] = multi_student_t_rng(nu, loc, scale)
        end
        rv
    end
    multi_student_t_rng(vector[m, n], nu, loc::vector[m, n], scale)::vector[m, n] = begin
        rv::vector[m, n]
        for i in 1:m
            rv[i] = multi_student_t_rng(nu, loc[i], scale)
        end
        rv
    end
    multi_student_t_cholesky_rng(vector[m, n], nu, loc::vector[n], scale)::vector[m, n] = begin
        rv::vector[m, n]
        for i in 1:m
            rv[i] = multi_student_t_cholesky_rng(nu, loc, scale)
        end
        rv
    end
    multi_student_t_cholesky_rng(vector[m, n], nu, loc::vector[m, n], scale)::vector[m, n] = begin
        rv::vector[m, n]
        for i in 1:m
            rv[i] = multi_student_t_cholesky_rng(nu, loc[i], scale)
        end
        rv
    end
end

# Natively-constrained square-matrix PARAMETERS. A family over `cov_matrix`,
# `corr_matrix`, `cholesky_factor_cov` or `cholesky_factor_corr` samples a single
# matrix (`<ct>[n]`, sized by one dim since `r_ndim(square_matrix) == 1`) or an
# array of them (`<ct>[m, n]`, i.e. `array[m] <ct>[n]`). Stan's densities take
# one matrix, so the array form is a UDF overload of the same `_lpdf` name that
# sums the element densities (typed by the sized element signatures beside the
# `@lhs` rows); `As ~ family(...)` then resolves to it. Its trailing arguments
# stay untyped so it never reads `dims(L)[2]`, which an empty array lacks. When
# no likelihood reaches the parameter (a prior-only program) or it is
# cv-held-out, it is re-drawn in generated quantities from the sized token of its
# DECLARED shape, `square_matrix[n]` or `square_matrix[m, n]` whatever the
# constrained family (the mangled name still carries it, e.g.
# `lkj_corr_cholesky_cholesky_factor_corr_rng`). The draw is a plain `matrix`,
# after the `dirichlet_rng`/`simplex` precedent; the lkj natives take the
# DIMENSION, not a container. Array draws loop `for i in 1:m`, which runs zero
# times for an empty array, and no native call is scalar-inflated, so all are
# outside the guarded class (todo 19n8abc).
for (dist, params, draw) in (
    (:wishart, (:nu, :Sigma), :(wishart_rng(nu, Sigma))),
    (:inv_wishart, (:nu, :Sigma), :(inv_wishart_rng(nu, Sigma))),
    (:wishart_cholesky, (:nu, :LSigma), :(wishart_cholesky_rng(nu, LSigma))),
    (:inv_wishart_cholesky, (:nu, :LSigma), :(inv_wishart_cholesky_rng(nu, LSigma))),
    (:lkj_corr, (:eta,), :(lkj_corr_rng(n, eta))),
    (:lkj_corr_cholesky, (:eta,), :(lkj_corr_cholesky_rng(n, eta))),
)
    dlpdf = Symbol(dist, :_lpdf)
    drng = Symbol(dist, :_rng)
    @eval @deffun $dlpdf(L::square_matrix[m, n], $(params...))::real = begin
        rv = 0.0
        for i in 1:m
            rv += $dlpdf(L[i], $(params...))
        end
        rv
    end
    @eval @deffun $drng(square_matrix[n], $(params...))::matrix[n, n] = $draw
    @eval @deffun $drng(square_matrix[m, n], $(params...))::matrix[m, n, n] = begin
        rv::matrix[m, n, n]
        for i in 1:m
            rv[i] = $draw
        end
        rv
    end
end
# Matrix-draw families observed as plain data: the gq token is a 2-dim matrix,
# so the sized-token slot is `matrix[n,m]`. Each delegates to its native form.
@deffun wishart_rng(matrix[n,m], nu, Sigma)::matrix[n,m] = wishart_rng(nu, Sigma)
@deffun inv_wishart_rng(matrix[n,m], nu, Sigma)::matrix[n,m] = inv_wishart_rng(nu, Sigma)
@deffun wishart_cholesky_rng(matrix[n,m], nu, L)::matrix[n,m] = wishart_cholesky_rng(nu, L)
@deffun inv_wishart_cholesky_rng(matrix[n,m], nu, L)::matrix[n,m] = inv_wishart_cholesky_rng(nu, L)
@deffun lkj_corr_rng(matrix[n,m], eta)::matrix[n,m] = lkj_corr_rng(n, eta)

# =============================================================================
# Longitudinal dose/time-response building blocks.
# Contract: StanBlocks ships the censored obs-model
# triad + the parametric mean KERNELS + index/broadcast helpers; a downstream
# term composes e.g. `log_y = baseline[series] + affectable .* time_resp .* exp(dose_resp)`
# and wires its own hierarchy. All resolve via the builtin path (no import).
# =============================================================================
@deffun begin
    # --- censored-normal observation model (limits of quantification) --------
    @stanonly begin
    # Scalar form: obs at/below the lower LOQ contributes the left-tail mass
    # (`normal_lcdf`), at/above the upper LOQ the right-tail mass
    # (`normal_lccdf`), otherwise the usual density. Indexed-accumulator idiom
    # (real[1]) keeps a single trailing return without early-return-in-branch.
    truncated_normal_lpdf(obs::real, loc::real, scale::real, lloq::real, uloq::real)::real = begin
        rv::real[1]
        rv[1] = normal_lpdf(obs, loc, scale)
        if obs <= lloq
            rv[1] = normal_lcdf(obs, loc, scale)
        else
            if obs >= uloq
                rv[1] = normal_lccdf(obs, loc, scale)
            end
        end
        rv[1]
    end
    # Pointwise vector form (generated_quantities log-lik term).
    truncated_normal_lpdfs(obs::vector[n], loc::vector[n], scale::vector[n], lloq::vector[n], uloq::vector[n])::vector[n] = begin
        rv::vector[n]
        for i in 1:n
            rv[i] = truncated_normal_lpdf(obs[i], loc[i], scale[i], lloq[i], uloq[i])
        end
        rv
    end
    # Vector form = sum of pointwise; `@lhs` opts it into base-level
    # `obs ~ truncated_normal(loc, scale, lloq, uloq)` sampling.
    @lhs truncated_normal_lpdf(obs::vector[n], loc::vector[n], scale::vector[n], lloq::vector[n], uloq::vector[n])::real =
        sum(truncated_normal_lpdfs(obs, loc, scale, lloq, uloq))
    # Posterior-predictive draw: sample then clamp into [lloq, uloq]. Per-element
    # `normal_rng(loc[i], scale[i])`, matching the `_lpdfs` loop idiom right above.
    # HISTORY: this form was originally FORCED — the whole-vector
    # `draws::vector[n] = to_vector(normal_rng(loc, scale))` infers its size from
    # `scale` (`dims(scale)[1]`), and the signature-dimension alias table recorded
    # `n` only against the FIRST `vector[n]` arg `loc`, so the typed-assignment
    # shape check threw. That gap is CLOSED (functions.jl `fun_size_alias_names`
    # now records every guarded occurrence), so the whole-vector form transpiles
    # too; the per-element draw is kept because it is correct Stan, costs nothing,
    # and reads like its `_lpdfs` sibling.
    truncated_normal_rng(loc::vector[n], scale::vector[n], lloq::vector[n], uloq::vector[n])::vector[n] = begin
        rv::vector[n]
        for i in 1:n
            rv[i] = fmin(fmax(lloq[i], normal_rng(loc[i], scale[i])), uloq[i])
        end
        rv
    end
    # Sized-token gq path (delegates to the native form; cf. multi_normal_rng).
    # Bare `vector[n]` (no `::`) is the token slot, matching the tokenof shape.
    # The delegate is the explicit per-element loop above — no guard (todo 19n8abc).
    truncated_normal_rng(vector[n], loc::vector[n], scale::vector[n], lloq::vector[n], uloq::vector[n])::vector[n] =
        truncated_normal_rng(loc, scale, lloq, uloq)

    # Heavy-tailed censored obs model. Direct analog of `truncated_normal` + a
    # leading `dof` arg. NOTE the censored branches evaluate the tail mass at the
    # LOQ LIMITS (`lloq`/`uloq`), whereas `truncated_normal_lpdf` evaluates it at
    # `obs`; the two coincide when a censored observation is recorded at its
    # limit, and each family keeps its own published semantics. `~
    # truncated_student_t(dof, loc, scale, lloq, uloq)` samples; SLIC auto-emits
    # the GQ log-lik + predictive draw.
    truncated_student_t_lpdf(obs::real, dof::real, loc::real, scale::real, lloq::real, uloq::real)::real = begin
        rv::real[1]
        rv[1] = student_t_lpdf(obs, dof, loc, scale)
        if obs <= lloq
            rv[1] = student_t_lcdf(lloq, dof, loc, scale)
        else
            if obs >= uloq
                rv[1] = student_t_lccdf(uloq, dof, loc, scale)
            end
        end
        rv[1]
    end
    # Pointwise vector form (generated_quantities log-lik term).
    truncated_student_t_lpdfs(obs::vector[n], dof::vector[n], loc::vector[n], scale::vector[n], lloq::vector[n], uloq::vector[n])::vector[n] = begin
        rv::vector[n]
        for i in 1:n
            rv[i] = truncated_student_t_lpdf(obs[i], dof[i], loc[i], scale[i], lloq[i], uloq[i])
        end
        rv
    end
    # Vector form = sum of pointwise; `@lhs` opts it into base-level
    # `obs ~ truncated_student_t(dof, loc, scale, lloq, uloq)` sampling.
    @lhs truncated_student_t_lpdf(obs::vector[n], dof::vector[n], loc::vector[n], scale::vector[n], lloq::vector[n], uloq::vector[n])::real =
        sum(truncated_student_t_lpdfs(obs, dof, loc, scale, lloq, uloq))
    # Posterior-predictive draw: sample then clamp into [lloq, uloq]. Per-element
    # `student_t_rng(dof[i], loc[i], scale[i])`, same rationale as
    # `truncated_normal_rng` above (originally forced by the size-alias gap, now
    # kept for symmetry with the `_lpdfs` loop).
    truncated_student_t_rng(dof::vector[n], loc::vector[n], scale::vector[n], lloq::vector[n], uloq::vector[n])::vector[n] = begin
        rv::vector[n]
        for i in 1:n
            rv[i] = fmin(fmax(lloq[i], student_t_rng(dof[i], loc[i], scale[i])), uloq[i])
        end
        rv
    end
    # Sized-token gq path (delegates to the native form; cf. multi_normal_rng).
    # Bare `vector[n]` (no `::`) is the token slot, matching the tokenof shape.
    # The delegate is the explicit per-element loop above — no guard (todo 19n8abc).
    truncated_student_t_rng(vector[n], dof::vector[n], loc::vector[n], scale::vector[n], lloq::vector[n], uloq::vector[n])::vector[n] =
        truncated_student_t_rng(dof, loc, scale, lloq, uloq)
    end

    # --- parametric mean kernels (per-observation; BRM does the [series] index) ---
    # Single-peak time response: with xi = (log t - loc)*exp(log_slope), the
    # value exp(log_inv_logit(xi)+log_inv_logit(-xi))*mag peaks once at log t=loc
    # and →0 as t→0 or t→∞ (log_slope stored unconstrained → exp() positive).
    @juliacompat biomarker_time_response(log_time::vector[n], loc::vector[n], log_slope::vector[n], mag::vector[n])::vector[n] = begin
        xi::vector[n] = (log_time - loc) .* exp(log_slope)
        exp(log_inv_logit(xi) + log_inv_logit(-xi)) .* mag
    end
    # Log dose response (sigmoid in log space): caller exp()s it in the compose.
    @juliacompat biomarker_dose_response(log_dose::vector[n], loc::vector[n], log_slope::vector[n])::vector[n] = begin
        xi::vector[n] = (log_dose - loc) .* exp(log_slope)
        log_inv_logit(xi)
    end

    # --- index / broadcast helpers (transformed-data) ------------------------
    # Column-major linear indices: `xy` with `vec(M)[xy] == M[x,y]` for an
    # (max(x) × max(y)) matrix `M`.
    @juliacompat linear_idxs(x::int[n], y::int[n])::int[n] = begin
        m = max(x)
        rv::int[n]
        for i in 1:n
            rv[i] = x[i] + (y[i] - 1) * m
        end
        rv
    end
    # Elementwise max(x[i], y) / (x[i] > y) for a vector x and scalar y.
    @juliacompat broadcasted_max(x::vector[n], y::real)::vector[n] = begin
        rv::vector[n]
        for i in 1:n
            rv[i] = fmax(x[i], y)
        end
        rv
    end
    @juliacompat broadcasted_gt(x::vector[n], y::real)::vector[n] = begin
        rv::vector[n]
        for i in 1:n
            rv[i] = x[i] > y
        end
        rv
    end
    # True-positions of a 0/1 integer mask, à la Julia's `findall(mask)`. Returns
    # the 1-based indices where `m[i] != 0` as a data-sized `int[sum(m)]` array —
    # so `idx = findall(cmt .== 1)` materialises a transformed-data integer index
    # column ONCE (`cmt` is data), and both `y[idx]` and `mu[idx]` share it at zero
    # runtime/gradient cost. Boolean-mask indexing (which Stan lacks) is thus
    # expressed as ordinary integer-array indexing over a precomputed index.
    Base.findall(m::int[n])::int[sum(m)] = begin
        k = sum(m)
        rv::int[k]
        j = 1
        for i in 1:n
            if m[i] != 0
                rv[j] = i
                j += 1
            end
        end
        rv
    end
end

# The math builtins Stan vectorises elementwise: unary ones, binary ones over
# real arguments, binary ones whose first argument is an int (a Bessel order, a
# binary label), and the ternary `fma`. The `@defsig` groups below type them
# over containers (stanc 2.39's tables); their Julia twins apply the scalar twin
# to each element (`_julia_elementwise`, functions.jl).
const _ELEMENTWISE_UNARY = (sqrt, inv_sqrt, exp, log, log10, sin, cos, asin, acos, tan, atan,
    cosh, sinh, tanh, acosh, asinh, atanh,
    log1m, inv_logit, logit, log_inv_logit, log1m_exp, expm1, Phi, lgamma, abs,
    log1p_exp, Base.inv, Base.log1p,
    exp2, log2, cbrt, ceil, floor, round, trunc,
    square, erf, erfc, tgamma, digamma, trigamma,
    Phi_approx, inv_Phi, inv_cloglog, lambert_w0)
const _ELEMENTWISE_BINARY = (fmin, fmax, fdim, fmod, atan2, lbeta, lchoose, log_diff_exp,
    owens_t, gamma_p, gamma_q, log_modified_bessel_first_kind)
const _ELEMENTWISE_INT_BINARY = (binary_log_loss, bessel_first_kind, bessel_second_kind,
    modified_bessel_first_kind, modified_bessel_second_kind)
# Each with its argument count.
const _ELEMENTWISE_NARY = (
    ((f, 2) for f in (_ELEMENTWISE_BINARY..., _ELEMENTWISE_INT_BINARY...))..., (fma, 3),
)
for f in _ELEMENTWISE_UNARY
    @eval jcall(tag::Val{$(QuoteNode(nameof(f)))}, scope::_JuliaScope, x::AbstractArray) =
        _julia_elementwise(tag, scope, x)
end
# One method per mix of scalar and container arguments that holds a container;
# the all-scalar call stays with the scalar twin.
for (f, n) in _ELEMENTWISE_NARY, kinds in Iterators.product(ntuple(_ -> (Number, AbstractArray), n)...)
    all(==(Number), kinds) && continue
    xs = [Symbol(:x, i) for i in 1:n]
    @eval jcall(tag::Val{$(QuoteNode(nameof(f)))}, scope::_JuliaScope,
        $((:($x::$k) for (x, k) in zip(xs, kinds))...)) = _julia_elementwise(tag, scope, $(xs...))
end

@defsig begin
    Union{typeof.(_ELEMENTWISE_UNARY)...} => begin
        (real,)=>real
        (vector[n],)=>vector[n]
        (row_vector[n],)=>row_vector[n]
        (real[n],)=>real[n]
        (matrix[m,n],)=>matrix[m,n]
    end
    # real -> int conversion. Stan's `to_int` requires a `data`-qualified
    # argument; StanBlocks places deterministic functions of data in
    # `transformed data`, where the argument is data-qualified, so a data-side
    # `to_int(round(...))` (dPCR partition counts) satisfies Stan's contract.
    typeof(to_int) => begin
        (real,) => int
        (real[n],) => int[n]
    end
    Union{typeof.((log_sum_exp, ))...} => begin
        (real, real) => real
        (vector[n], vector[n]) => vector[n]
    end
    # Binary builtins Stan vectorises: a scalar with a container, or two
    # containers of one kind (stanc rejects `fmax(vector, row_vector)` and
    # `fmax(array[] real, vector)`). `types.int <: types.real`, so every row
    # also takes an int scalar, and the `real[n]` rows an `int[n]` in either
    # slot, which stanc promotes to `array[] real` (the mixed
    # `(array[] int, array[] real)` pair included; verified on stanc3 v2.39.0).
    # The scalar rows are in the "Scalar math" block. Nested arrays
    # (`array[] vector`, `array[,] real`) are not registered.
    Union{typeof.(_ELEMENTWISE_BINARY)...} => begin
        (real, vector[n]) => vector[n]
        (vector[n], real) => vector[n]
        (vector[n], vector[n]) => vector[n]
        (real, row_vector[n]) => row_vector[n]
        (row_vector[n], real) => row_vector[n]
        (row_vector[n], row_vector[n]) => row_vector[n]
        (real, matrix[m,n]) => matrix[m,n]
        (matrix[m,n], real) => matrix[m,n]
        (matrix[m,n], matrix[m,n]) => matrix[m,n]
        (real, real[n]) => real[n]
        (real[n], real) => real[n]
        (real[n], real[n]) => real[n]
    end
    # The int-first-argument binaries: an int with any container, or an int
    # array with a scalar or a 1-D container (stanc has no
    # `(array[] int, matrix)`).
    Union{typeof.(_ELEMENTWISE_INT_BINARY)...} => begin
        (int, vector[n]) => vector[n]
        (int, row_vector[n]) => row_vector[n]
        (int, matrix[m,n]) => matrix[m,n]
        (int, real[n]) => real[n]
        (int[n], real) => real[n]
        (int[n], vector[n]) => vector[n]
        (int[n], row_vector[n]) => row_vector[n]
        (int[n], real[n]) => real[n]
    end
    # `fma(x, y, z)`: scalars mixed with ONE of vector, row_vector or matrix
    # (stanc has no array forms).
    typeof(fma) => begin
        (vector[n], real, real) => vector[n]
        (real, vector[n], real) => vector[n]
        (real, real, vector[n]) => vector[n]
        (vector[n], vector[n], real) => vector[n]
        (vector[n], real, vector[n]) => vector[n]
        (real, vector[n], vector[n]) => vector[n]
        (vector[n], vector[n], vector[n]) => vector[n]
        (row_vector[n], real, real) => row_vector[n]
        (real, row_vector[n], real) => row_vector[n]
        (real, real, row_vector[n]) => row_vector[n]
        (row_vector[n], row_vector[n], real) => row_vector[n]
        (row_vector[n], real, row_vector[n]) => row_vector[n]
        (real, row_vector[n], row_vector[n]) => row_vector[n]
        (row_vector[n], row_vector[n], row_vector[n]) => row_vector[n]
        (matrix[m,n], real, real) => matrix[m,n]
        (real, matrix[m,n], real) => matrix[m,n]
        (real, real, matrix[m,n]) => matrix[m,n]
        (matrix[m,n], matrix[m,n], real) => matrix[m,n]
        (matrix[m,n], real, matrix[m,n]) => matrix[m,n]
        (real, matrix[m,n], matrix[m,n]) => matrix[m,n]
        (matrix[m,n], matrix[m,n], matrix[m,n]) => matrix[m,n]
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
    # The row/column products and self products also take a vector (n rows,
    # one column) or a row vector (one row, n columns).
    typeof(columns_dot_product) => begin
        (matrix[m,n], matrix[m,n]) => row_vector[n]
        (vector[n], vector[n]) => row_vector[1]
        (row_vector[n], row_vector[n]) => row_vector[n]
    end
    typeof(columns_dot_self) => begin
        (matrix[m,n],) => row_vector[n]
        (vector[n],) => row_vector[1]
        (row_vector[n],) => row_vector[n]
    end
    typeof(rows_dot_self) => begin
        (matrix[m,n],) => vector[m]
        (vector[n],) => vector[n]
        (row_vector[n],) => vector[1]
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
        (row_vector[n],) => row_vector[n]
        (int, real) => real
        (int, int) => int
        (real, int) => real
        (real, real) => real
        # NOTE: scalar-array operand rows (`(int[n], int[n])`, `(int[n], int)`,
        # `(int, int[n])`, `(real[n], real[n])`, `(real, real[n])`) are deliberately
        # ABSENT — Stan has no elementwise `+ - * / ^` on `array[] int` / `array[] real`,
        # so they are rejected loudly by `_reject_scalar_array_elementwise` (functions.jl)
        # rather than silently emitting invalid Stan. Do not re-add them.
        (int, vector[n]) => vector[n]
        (real, vector[n]) => vector[n]
        (int, row_vector[n]) => row_vector[n]
        (real, row_vector[n]) => row_vector[n]
        (real, matrix[m,n]) => matrix[m,n]
        (vector[n], real) => vector[n]
        (vector[n], int) => vector[n]
        (vector[n], vector[m]) => vector[n]
        (row_vector[n], real) => row_vector[n]
        (row_vector[n], int) => row_vector[n]
        (row_vector[n], row_vector[m]) => row_vector[n]
        (matrix[m,n], real) => matrix[m,n]
    end
    Union{typeof.((+, -))...} => begin 
        (matrix[m,n],matrix[m,n]) => matrix[m,n]
    end
    Union{typeof.((*, ))...} => begin
        (vector[m], row_vector[n]) => matrix[m,n]
        (row_vector[n], vector[n]) => real
        (row_vector[m], matrix[m,n]) => row_vector[n]
        (matrix[m,n], vector[n]) => vector[m]
        (matrix[m,n], matrix[n,o]) => matrix[m,o]
    end
    typeof(adjoint) => begin
        (vector[n],) => row_vector[n]
        (row_vector[n],) => vector[n]
        (matrix[m,n],) => matrix[n,m]
    end
    typeof(transpose) => begin
        (vector[n],) => row_vector[n]
        (row_vector[n],) => vector[n]
        (matrix[m,n],) => matrix[n,m]
    end
    typeof(getindex) => begin 
        (int[m], int) => int
        (int[m], int[n]) => int[n]
        (int[m,n], int) => int[n]
        (int[m,n], int[o], int) => int[o]
        (int[m,n], int, int) => int
        # Array row-slice (`y[i, :]` desugars to `y[i, 1:n]`) and sub-array
        # (`y[a:b, c:d]`), mirroring the `matrix[m,n]` rows below. Without the
        # `(int[m,n], int, int[o])` row an int-array row-slice assignment
        # `y[i, :] = multinomial_rng(...)` degraded to `anything` (snag
        # multinomial-cust-59569d79).
        (int[m,n], int, int[o]) => int[o]
        (int[m,n], int[o], int[p]) => int[o, p]
        (real[m], int) => real
        (real[m], int[n]) => real[n]
        (real[m,n], int) => real[n]
        (real[m,n], int[o], int) => real[o]
        (real[m,n], int, int[o]) => real[o]
        (real[m,n], int[o], int[p]) => real[o, p]
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
        # A constrained square matrix (or an array of them) indexes through the
        # `matrix[...]` rows above as K-by-K (`xformal_type`). Specialisation: a
        # block of a Cholesky-correlation factor taken from an array of them
        # (`L[g, :, :]`, the per-cell `lkj_corr_cholesky_lpdf` loop) stays one.
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
    Union{typeof.((student_t_rng, skew_normal_rng, exp_mod_normal_rng,
                    skew_double_exponential_rng, pareto_type_2_rng))...} => begin
        (real, real, real) => real
        (real, vector[n], real) => real[n]
        (real, vector[n], vector[n]) => real[n]
        (vector[n], real, real) => real[n]
        (vector[n], real, vector[n]) => real[n]
        (vector[n], vector[n], real) => real[n]
        (vector[n], vector[n], vector[n]) => real[n]
    end
    typeof(beta_binomial_rng) => begin
        (int[n], real, real) => int[n]
        (int[n], vector[n], real) => int[n]
        (int[n], real, vector[n]) => int[n]
        (int[n], vector[n], vector[n]) => int[n]
    end
    # Stan's elementwise operators `.*` / `./` / `.^` (`elt_multiply`,
    # `elt_divide`, `pow`) take two operands of one native container type, or a
    # container and a scalar, and return that container — the complete
    # non-complex, non-array census of stanc 2.39 `--dump-stan-math-signatures`
    # (snag matrix-elementwi-c69bbd9a). Keyed on exactly these three: `.÷` and
    # the dotted comparisons also canonicalize to a `Base.BroadcastFunction`, but
    # Stan has no such operator for any operand type, so they must fail at trace
    # time instead of typing and rendering `BroadcastFunction(div)(…)` into the
    # program.
    Union{typeof(.*), typeof(./), typeof(.^)} => begin
        (real, real) => real
        # NOTE: scalar-array operand rows (`(real[n], real[n])`, `(real[n], real)`,
        # `(real, real[n])`, `(int[n], int)`, `(int[n], int[n])`) are deliberately
        # ABSENT — Stan has no elementwise `.* ./ .^` on `array[] int` / `array[] real`,
        # so they are rejected loudly by the `Base.BroadcastFunction` `tracetype` guard
        # (functions.jl) rather than silently emitting invalid Stan. Do not re-add them.
        (vector[n], real) => vector[n]
        (real, vector[n]) => vector[n]
        (vector[n], vector[n]) => vector[n]
        (row_vector[n], real) => row_vector[n]
        (real, row_vector[n]) => row_vector[n]
        (row_vector[n], row_vector[n]) => row_vector[n]
        (matrix[m,n], real) => matrix[m,n]
        (real, matrix[m,n]) => matrix[m,n]
        (matrix[m,n], matrix[m,n]) => matrix[m,n]
    end
    typeof(cholesky_decompose) => begin
        (matrix[n,n],) => matrix[n,n]
    end
    typeof(mdivide_right_tri_low) => begin
        (row_vector[n], matrix[n,n]) => row_vector[n] 
    end
    typeof(diag_pre_multiply) => begin
        (vector[m], matrix[m,n]) => matrix[m,n]
    end
    typeof(diag_post_multiply) => begin
        (matrix[m,n], vector[n]) => matrix[m,n] 
        (matrix[m,n], row_vector[n]) => matrix[m,n] 
    end
    typeof(gp_exp_quad_cov) => begin 
        # One-dimensional inputs (`array[] real`).
        (real[n], real, real) => matrix[n,n]
        (real[m], real[n], real, real) => matrix[m,n]
        # Multidimensional inputs (`array[] vector`), matching Stan's four
        # native self/cross-covariance overloads. `vector[n,d]` is SLIC's
        # spelling for `array[n] vector[d]`; the per-axis native scale is an
        # `array[d] real`, hence `real[d]` rather than `vector[d]` here.
        (vector[n,d], real, real) => matrix[n,n]
        (vector[n,d], real, real[d]) => matrix[n,n]
        (vector[m,d], vector[n,d], real, real) => matrix[m,n]
        (vector[m,d], vector[n,d], real, real[d]) => matrix[m,n]
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
        # Boolean-valued: `bool` (a subtype of `int`, so still an int everywhere
        # except that a `bool[n]` result used as a `getindex` argument means a
        # boolean mask). A scalar comparison stays semantically an int (renders as
        # `int`), so nothing downstream changes except mask-shaped indexing.
        (anything, anything) => bool
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
        (matrix[m,n],) => real[m*n]
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

# Julia's own functions for builtins the `@builtin_module` manifest declares
# as fresh functions. A body that names `mean` resolves the builtin, but one
# that holds the VALUE `Statistics.mean` (BRM's model scope, a `$(f)` splice)
# reached no tracetype and failed as `mean(::vector)::anything` (snag
# `mean-vector-trac-306ef1b4`). Each pair computes the same thing:
# `Statistics.std` / `Statistics.var` are Stan's `sd` / `variance` (both divide
# by N - 1; one element gives Stan 0, Julia NaN), `quantile` is Hyndman-Fan
# type 7 in both, and `hcat` / `reshape` are the `@juliacompat` forms written
# to match Base. `builtin.reverse` has no package-level alias (see its
# definition). Not here: `gamma`, `beta` and `binomial` are SLIC distribution
# families, while Julia's are special functions; `Base.tail` drops a tuple's
# first element, which is not Stan's `tail`.
for (f, g) in (
    (Statistics.mean, mean), (Statistics.std, sd), (Statistics.var, variance),
    (Statistics.quantile, quantile),
    (SpecialFunctions.erf, erf), (SpecialFunctions.erfc, erfc),
    (SpecialFunctions.digamma, digamma), (SpecialFunctions.trigamma, trigamma),
    (Base.log2, log2), (Base.cbrt, cbrt), (Base.fma, fma),
    (Base.hcat, hcat), (Base.reshape, reshape), (Base.reverse, builtin.reverse),
)
    @eval _slic_callable(::typeof($f)) = $g
end
# Of those, the Statistics functions take keywords in Julia and Stan's take none,
# so a keyword cannot reach the emitted call (functions.jl, `_reject_call_kwargs`).
_reject_call_kwargs(x::CanonicalExpr{<:Union{typeof.((mean, sd, variance, quantile))...}}) =
    isempty(x.kwargs) || error(
        "`", nameof(head(x)), "` takes no keyword arguments in Stan, so `",
        join(keys(x.kwargs), "`, `"), "` cannot be honoured; the keyword forms of ",
        "Julia's `Statistics` functions (`mean(x; dims=1)`, `std(x; corrected=false)`) ",
        "have no Stan counterpart. Call it without keywords, or write the computation out.",
    )

# Stan's per-axis multidimensional GP overload accepts `array[] real`, whereas
# an ordinary dense real vector in SLIC has Stan type `vector`. Keep the native
# @defsig rows exact and provide the consumer-facing vector convenience as a
# real UDF adapter; registering `(array[] vector, real, vector)` as native would
# transpile but fail stanc's signature check.
@deffun begin
    @stanonly begin
    gp_exp_quad_cov(x::vector[n,d], sigma::real,
                    length_scale::vector[d])::matrix[n,n] =
        gp_exp_quad_cov(x, sigma, to_array_1d(length_scale))
    gp_exp_quad_cov(x1::vector[m,d], x2::vector[n,d], sigma::real,
                    length_scale::vector[d])::matrix[m,n] =
        gp_exp_quad_cov(x1, x2, sigma, to_array_1d(length_scale))
    end
end

const TolODESolver = Union{typeof.((ode_rk45_tol, ode_ckrk_tol, ode_adams_tol, ode_bdf_tol))...}
const NoTolODESolver = Union{typeof.((ode_rk45, ode_ckrk, ode_adams, ode_bdf))...}
const ODESolver = Union{TolODESolver, NoTolODESolver}
const ReduceSumFunction = Union{typeof.((reduce_sum, reduce_sum_static))...}

tracetype(x::CanonicalExpr{<:ODESolver}) = StanType(
    types.vector, (stan_size(x.args[4], 1), stan_size(x.args[2], 1))
)

# ── Torsten `pmx_solve_*` analytical (closed-form) PK solvers ──
# Torsten (metrumresearchgroup/Torsten) extends Stan — a pmx-aware stanc3 fork plus
# a `torsten_math` submodule — with NONMEM-style event-schedule solvers. The
# analytical compartment models take the event arrays
#   time, amt, rate, ii  :: array[] real     (AutoDiffable)
#   evid, cmt, addl, ss   :: array[] int      (data)
# followed by the PK parameters `theta` (+ optional biovar/tlag), and return a
# `matrix[nCmt, nEvent]` of per-compartment amounts at each event. nCmt is fixed
# per model (onecpt = depot+central = 2; twocpt = +peripheral = 3); nEvent is the
# leading dim of the first argument (`time`) — the same shape-from-an-argument
# pattern as the native `ODESolver` tracetype above. `func_name(x::Function)`
# emits the bare Stan name and the generic `CanonicalExpr` show renders the
# positional call, so only the return-type tracetype is needed (no show method).
_pmx_solve_tracetype(x, ncmt) = StanType(
    types.matrix, (stan_expr(ncmt, ncmt), stan_size(x.args[1], 1))
)
tracetype(x::CanonicalExpr{typeof(pmx_solve_onecpt)}) = _pmx_solve_tracetype(x, 2)
tracetype(x::CanonicalExpr{typeof(pmx_solve_twocpt)}) = _pmx_solve_tracetype(x, 3)
# The effect-compartment variants append one effect compartment to the same
# closed-form models and take the same argument layout.
tracetype(x::CanonicalExpr{typeof(pmx_solve_onecpt_effcpt)}) = _pmx_solve_tracetype(x, 3)
tracetype(x::CanonicalExpr{typeof(pmx_solve_twocpt_effcpt)}) = _pmx_solve_tracetype(x, 4)
# `pmx_solve_linode(time, …, ss, K, biovar, tlag)` solves `y' = K y` by matrix
# exponential. nCmt is the row count of `K`: a `matrix`, or an `array[] matrix`
# (one system matrix per event), whose rows are its second-to-last dimension.
tracetype(x::CanonicalExpr{typeof(pmx_solve_linode)}) = StanType(
    types.matrix, (stan_size(x.args[9], stan_ndim(x.args[9]) - 1), stan_size(x.args[1], 1))
)

# ── Torsten functor (general-ODE) and coupled event solvers ──
#   pmx_solve_{rk45,bdf,adams}(f, nCmt, time, amt, rate, ii, evid, cmt, addl, ss,
#       theta[, biovar, tlag][, x_r, x_i][, rel_tol, abs_tol, max_num_steps
#       [, as_rel_tol, as_abs_tol, as_max_num_steps]])     → matrix[nCmt, nEvent]
#   pmx_solve_{onecpt,twocpt}_{rk45,bdf}(f, nOde, time, …, ss, theta, biovar,
#       tlag[, <tolerances>])                               → matrix[nPK + nOde, nEvent]
# The first family integrates the user's ODE `f` over the same NONMEM event
# schedule as the analytical solvers. The coupled family solves the one-/two-
# compartment PK states (nPK = 2 / 3) in closed form and only the user's `nOde`
# downstream states numerically; `f` receives the PK amounts as `y_pk`.
#
# Torsten's pmx stanc fixes the RHS signature; it is NOT Stan's variadic ODE
# interface, so the solver cannot forward trailing arguments to `f`:
#   vector f(real t, vector y, array[] real theta, data array[] real x_r, data array[] int x_i)
#   vector f(real t, vector y, vector y_pk, array[] real theta, data array[] real x_r, data array[] int x_i)
# Parameters reach `f` through `theta` (one row per event when `theta` is an
# `array[,] real`), data through the optional per-event `x_r`/`x_i` rows; Torsten
# passes empty arrays when they are omitted (and the coupled family has none).
# The RHS is fetched at exactly that call (`_pmx_rhs_call`), and the solver call
# names it in the function slot (`show(::CanonicalExpr{<:PMXFunctorSolver})`). A
# closure that captures values cannot be threaded through that fixed signature
# as it is; the call is rewritten to a generated adapter first (below).
const PMXODESolver = Union{typeof.((pmx_solve_rk45, pmx_solve_bdf, pmx_solve_adams))...}
const PMXCoupledSolver = Union{typeof.((
    pmx_solve_onecpt_rk45, pmx_solve_onecpt_bdf, pmx_solve_twocpt_rk45, pmx_solve_twocpt_bdf,
))...}
const PMXFunctorSolver = Union{PMXODESolver, PMXCoupledSolver}
_pmx_n_pk(::CanonicalExpr{<:Union{typeof(pmx_solve_onecpt_rk45), typeof(pmx_solve_onecpt_bdf)}}) = 2
_pmx_n_pk(::CanonicalExpr{<:Union{typeof(pmx_solve_twocpt_rk45), typeof(pmx_solve_twocpt_bdf)}}) = 3
_pmx_check_functor_call(x::CanonicalExpr) = begin
    _pmx_check_arguments(x)
    # `expand_inline_or_trace` replaces a capturing closure by its capture-free
    # adapter before the call is traced; one arriving here bypassed that rewrite.
    isempty(_closure_captures(x.args[1])) || error(
        "$(nameof(head(x))): internal error — a capturing right-hand-side closure ",
        "reached the solver's tracetype without the theta-packing rewrite."
    )
end
# `has_theta = false`: the short-closure form of the ODE family (below) omits it.
_pmx_check_arguments(x::CanonicalExpr, has_theta = true) = begin
    fname = nameof(head(x))
    nmin = has_theta ? 11 : 10
    length(x.args) >= nmin || error(
        "$fname: expected at least $nmin arguments `(f, n, time, amt, rate, ii, evid, cmt, ",
        "addl, ss", has_theta ? ", theta" : "", ", …)`, got $(length(x.args))."
    )
    n = x.args[2]
    (center_type(n) <: types.int && stan_ndim(n) == 0) || error(
        "$fname: the second argument (the compartment / ODE count) must be an `int`, ",
        "got `$(sigtype(n))`."
    )
end
tracetype(x::CanonicalExpr{<:PMXODESolver}) = begin
    _pmx_check_functor_call(x)
    StanType(types.matrix, (x.args[2], stan_size(x.args[3], 1)))
end
tracetype(x::CanonicalExpr{<:PMXCoupledSolver}) = begin
    _pmx_check_functor_call(x)
    n_cmt = stan_expr(CanonicalExpr(+, stan_expr(_pmx_n_pk(x)), x.args[2]))
    StanType(types.matrix, (n_cmt, stan_size(x.args[3], 1)))
end
_pmx_row(a, context) = stan_ndim(a) >= 2 ? _jb_elem(a, context) : a
_pmx_state(n) = StanExpr(:_, StanType(types.vector, (n,)))
_pmx_states(x::CanonicalExpr{<:PMXODESolver}) = (_pmx_state(x.args[2]),)
_pmx_states(x::CanonicalExpr{<:PMXCoupledSolver}) =
    (_pmx_state(x.args[2]), _pmx_state(stan_expr(_pmx_n_pk(x))))
# The optional tail after `theta` (argument 11, so the tail starts at `from`),
# as `(biovar_tlag, x_r_x_i, tolerances)`: `biovar, tlag` are arrays, the
# per-event `x_r, x_i` are 2-D arrays that follow them (ODE family only), and the
# tolerances are scalars.
_pmx_tail(x::CanonicalExpr{<:PMXODESolver}, from = 12) = begin
    tail = x.args[from:end]
    nb = length(tail) >= 2 && stan_ndim(tail[1]) >= 1 ? 2 : 0
    nx = nb == 2 && length(tail) >= 4 && stan_ndim(tail[3]) == 2 ? 2 : 0
    (tail[1:nb], tail[nb+1:nb+nx], tail[nb+nx+1:end])
end
_pmx_tail(x::CanonicalExpr{<:PMXCoupledSolver}, from = 12) =
    (x.args[from:min(from + 1, end)], (), x.args[from+2:end])
_pmx_has_data_rows(x) = !isempty(_pmx_tail(x)[2])
_pmx_data_rows(x, context) = _pmx_has_data_rows(x) ?
    (_jb_elem(x.args[14], context), _jb_elem(x.args[15], context)) :
    (StanExpr(:_, StanType(types.real, (stan_expr(0),))),
     StanExpr(:_, StanType(types.int, (stan_expr(0),))))
_pmx_rhs_call(x::CanonicalExpr{<:PMXFunctorSolver}, context) = CanonicalExpr(
    x.args[1], _jb_elem(x.args[3], context), _pmx_states(x)...,
    _pmx_row(x.args[11], context), _pmx_data_rows(x, context)...,
)
fetch_functions!(x::CanonicalExpr{<:PMXFunctorSolver}; info) = begin
    rhs = _pmx_rhs_call(x, _trace_context(info))
    fetch_functions!(rhs; info)
    # A right-hand side with no method for Torsten's call gets no definition and
    # would reach stanc as an undeclared identifier.
    get(info, sig_expr(rhs), nothing) === nothing && error(
        nameof(head(x)), ": the right-hand side `", _pmx_rhs_label(x.args[1]), "` has no ",
        "method for Torsten's call `(", join(_pmx_rhs_slots(x), ", "), ")` with types `(",
        join(map(a -> sigtype(a), rhs.args), ", "), ")`. Define it with that signature, ",
        "or call a function taking other arguments through a closure: ",
        "`(", join(_pmx_short_slots(x), ", "), ") -> f(", join(_pmx_short_slots(x), ", "), ", …)`."
    )
end
_pmx_rhs_label(f::StanExpr2{<:types.func}) = nameof(type(f).info.value)
_pmx_rhs_label(f::Function) = nameof(f)
_pmx_rhs_label(_) = "f"

# ── Capturing closures as functor right-hand sides: a generated adapter ──
# Torsten calls the right-hand side with its fixed signature, so a closure's
# captured values cannot follow as trailing call arguments the way `ode_rk45`
# threads them (`show(::CanonicalExpr{<:ODESolver})`). The call is rewritten
# before it is traced instead:
#   - each capture is flattened onto the END of an argument Torsten forwards to
#     the right-hand side: autodiff reals onto `theta` (onto every row of a
#     per-event `array[,] real` theta), model-scope data reals onto `x_r`, ints
#     onto `x_i`.
#     Reaching `x_r`/`x_i` inserts Torsten's documented defaults `biovar = 1`,
#     `tlag = 0` when the call omitted them, and empty per-event rows for the
#     other one of the pair when only one carries captures;
#   - the closure is replaced by a capture-free closure at Torsten's arity whose
#     body first re-reads each capture from that tail, re-binds the user's own
#     argument to the leading part, and then runs the user's body. It lifts to a
#     Stan function through the ordinary closure `_fundef`.
# Reading the captures from the END leaves the user's own arguments any size,
# so only a capture needs a literal size. The coupled solvers forward no
# `x_r`/`x_i`: their data reals ride `theta`, and an int capture is refused.
#
# A closure may also take just `(t, y)` (coupled: `(t, y, y_pk)`) and reach
# everything else through captures; the adapter supplies the remaining slots.
# The ODE family's call then omits `theta` as well, because only the right-hand
# side reads it. A coupled solver's call keeps it: its closed-form PK
# compartments read their parameters from the front of `theta`.
# A `@deffun` with extra arguments, `f(t, y, cl, v, ka)`, is called through
# such a closure: `(t, y) -> f(t, y, cl, v, ka)`.
@deffun begin
    @stanonly begin
    # A per-event `theta`/`x_r`/`x_i` argument with `extra` appended to each row.
    pmx_append_each(rows::real[n, k], extra::real[m])::real[n, k + m] = begin
        out::real[n, k + m]
        for i in 1:n
            out[i] = append_array(rows[i], extra)
        end
        out
    end
    pmx_append_each(rows::int[n, k], extra::int[m])::int[n, k + m] = begin
        out::int[n, k + m]
        for i in 1:n
            out[i] = append_array(rows[i], extra)
        end
        out
    end
    end
end
_pmx_rhs_slots(::CanonicalExpr{<:PMXODESolver}) = (:t, :y, :theta, :x_r, :x_i)
_pmx_rhs_slots(::CanonicalExpr{<:PMXCoupledSolver}) = (:t, :y, :y_pk, :theta, :x_r, :x_i)
_pmx_short_slots(::CanonicalExpr{<:PMXODESolver}) = (:t, :y)
_pmx_short_slots(::CanonicalExpr{<:PMXCoupledSolver}) = (:t, :y, :y_pk)
_pmx_forwards_data(::CanonicalExpr{<:PMXODESolver}) = true
_pmx_forwards_data(::CanonicalExpr{<:PMXCoupledSolver}) = false
_pmx_call_has_theta(::CanonicalExpr{<:PMXODESolver}, short) = !short
_pmx_call_has_theta(::CanonicalExpr{<:PMXCoupledSolver}, short) = true
# Only a full-arity closure without captures is already Torsten-shaped.
_pmx_needs_adapter(f, x) = false
_pmx_needs_adapter(f::StanExpr2{<:types.closure}, x) = !isempty(_closure_captures(f)) ||
    length(type(f).info.value.arg_names) != length(_pmx_rhs_slots(x))
expand_inline_or_trace(x::CanonicalExpr{<:PMXFunctorSolver}; info) =
    _pmx_needs_adapter(x.args[1], x) ?
        forward!(_pmx_adapter_call(x, type(x.args[1]).info.value; info); info) :
        invoke(expand_inline_or_trace, Tuple{CanonicalExpr}, x; info)
_pmx_adapter_call(x, cl; info) = begin
    fname = nameof(head(x))
    slots = _pmx_rhs_slots(x)
    short_slots = _pmx_short_slots(x)
    (isnothing(cl.vararg_name) && length(cl.arg_names) in (length(short_slots), length(slots))) || error(
        "$fname: the right-hand-side closure takes `(", join(vcat(string.(collect(cl.arg_names)),
        isnothing(cl.vararg_name) ? String[] : ["$(cl.vararg_name)..."]), ", "), ")`, but Torsten ",
        "calls it as `(", join(slots, ", "), ")`; a closure may also take just `(",
        join(short_slots, ", "), ")` and capture everything else."
    )
    short = length(cl.arg_names) == length(short_slots)
    has_theta = _pmx_call_has_theta(x, short)
    _pmx_check_arguments(x, has_theta)
    # The short slots are a prefix of Torsten's, so `zip` names exactly those.
    user = Dict(zip(slots, cl.arg_names))
    stmts = Meta.isexpr(cl.body, :block) ? cl.body.args : Any[cl.body]
    referenced = Set{Symbol}()
    _collect_all_syms!(cl.body, referenced)
    taken = union(referenced, cl.arg_names)
    caps = [_pmx_capture(x, name, v; info) for (name, v) in cl.captures]
    names = copy(user)
    prologue = Any[]
    packed = Dict{Symbol,Any}()
    for slot in (:theta, :x_r, :x_i)
        here = filter(c -> c.slot === slot, caps)
        isempty(here) && continue
        order, packed[slot] = _pmx_pack(here)
        a = names[slot] = _pmx_fresh(Symbol(slot, :_ext), taken)
        o = _pmx_fresh(Symbol(slot, :_end), taken)
        push!(prologue, :($o = size($a) - $(sum(c -> c.len, order))))
        get(user, slot, nothing) in referenced && push!(prologue, :($(user[slot]) = $a[1:$o]))
        first = 1
        for c in order
            push!(prologue, :($(c.name) = $(_pmx_unpack(c.T, c.dims, a, o, first))))
            first += c.len
        end
    end
    for slot in slots
        haskey(names, slot) || (names[slot] = _pmx_fresh(slot, taken))
    end
    adapter = _closure_stan_expr((
        arg_names = Tuple(names[s] for s in slots),
        vararg_name = nothing,
        body = Expr(:block, prologue..., stmts...),
        captures = OrderedDict{Symbol,Any}(),
        kwargs = (),
        source = cl.source,
        id = _next_closure_id(info),
        mod = cl.mod,
    ))
    biovar_tlag, data, tol = _pmx_tail(x, has_theta ? 12 : 11)
    all(a -> stan_ndim(a) == 0, tol) || error(
        "$fname: after ", has_theta ? "`theta`" : "the event arrays", " expected ",
        "`biovar, tlag`", _pmx_forwards_data(x) && !short ? ", then `x_r, x_i`," : "",
        " and then scalar tolerances; got an array among the tolerances."
    )
    short && !isempty(data) && error(
        "$fname: the closure `(", join(cl.arg_names, ", "), ")` cannot read the `x_r`/`x_i` ",
        "the call passes; take `(", join(slots, ", "), ")` to read them, or capture the data."
    )
    if haskey(packed, :x_r) || haskey(packed, :x_i)
        n = x.args[2]
        isempty(biovar_tlag) && (biovar_tlag = (
            CanonicalExpr(builtin.rep_array, 1.0, n), CanonicalExpr(builtin.rep_array, 0.0, n),
        ))
        x_r, x_i = isempty(data) ? (nothing, nothing) : data
        data = (
            _pmx_rows_with(x, x_r, get(packed, :x_r, nothing), 0.0),
            _pmx_rows_with(x, x_i, get(packed, :x_i, nothing), 0),
        )
    end
    CanonicalExpr(
        head(x), adapter, x.args[2:10]...,
        _pmx_theta_with(has_theta ? x.args[11] : nothing, get(packed, :theta, nothing)),
        biovar_tlag..., data..., tol...,
    )
end
# A capture as the adapter carries it: the slot it rides, its center type, and
# its literal value shape (`()` for a scalar) and flat length.
_pmx_capture(x, name, v; info) = begin
    fname = nameof(head(x))
    T = center_type(v)
    shape = _pmx_capture_shape(T, v)
    isnothing(shape) && error(
        "$fname: the right-hand-side closure captures `$name` of type `$(sigtype(v))`, ",
        "which cannot be packed into Torsten's `theta`/`x_r`/`x_i`. A capture may be an ",
        "`int` or `real` scalar or 1-D array, a `vector`, a `row_vector` or a `matrix`; ",
        "pass anything else through `theta` yourself."
    )
    dims = map(_pmx_literal_size, shape)
    any(isnothing, dims) && error(
        "$fname: the right-hand-side closure captures `$name` of type `$(sigtype(v))`, ",
        "whose size is not a literal. The generated right-hand side reads each capture ",
        "from the end of `theta`/`x_r`/`x_i` and cannot see a data size; capture its ",
        "elements as scalars, or pass it through `theta` yourself."
    )
    (; name, T, dims, value = v, len = prod(dims; init = 1), slot = _pmx_capture_slot(x, name, v, T; info))
end
_pmx_capture_shape(::Type{<:types.real}, v) = stan_ndim(v) <= 1 ? stan_size(v) : nothing
_pmx_capture_shape(::Type{<:types.any_vector}, v) = stan_ndim(v) == 1 ? stan_size(v) : nothing
_pmx_capture_shape(::Type{<:types.matrix}, v) = length(index_shape(v)) == 2 ? index_shape(v) : nothing
_pmx_capture_shape(::Type, v) = nothing
_pmx_literal_size(s::Integer) = Int(s)
_pmx_literal_size(s::StanExpr) = _pmx_literal_size(expr(s))
_pmx_literal_size(_) = nothing
_pmx_capture_slot(x, name, v, ::Type{<:types.int}; info) = _pmx_forwards_data(x) ? :x_i : error(
    "$(nameof(head(x))): the right-hand-side closure captures the int `$name`, but the ",
    "coupled solvers forward no `x_i` to the right-hand side, and `theta` holds reals. ",
    "Capture a real instead, or solve the whole system with `pmx_solve_rk45`/`bdf`/`adams`."
)
# Torsten's `x_r` is data-only. Inside a `@deffun` body even a `:data`-qualified
# value is a function argument (or derived from one), which Stan never treats
# as data, so only a model-scope data real rides `x_r`.
_pmx_capture_slot(x, name, v, ::Type; info) =
    qual(v) == :data && _pmx_forwards_data(x) && _is_model_decl_scope(info) ? :x_r : :theta
# The captures riding one slot in packed order — real scalars first, as one
# `[…]` literal — and the 1-D array they pack to at the call site.
_pmx_real_scalar(c) = isempty(c.dims) && !(c.T <: types.int)
_pmx_pack(caps) = begin
    scalars = filter(_pmx_real_scalar, caps)
    others = filter(!_pmx_real_scalar, caps)
    parts = Any[_pmx_flat(c.T, c.value, c.dims) for c in others]
    isempty(scalars) || pushfirst!(parts, CanonicalExpr(
        builtin.to_array_1d, CanonicalExpr(:vect, (c.value for c in scalars)...)
    ))
    (vcat(scalars, others), foldl((a, b) -> CanonicalExpr(builtin.append_array, a, b), parts))
end
_pmx_flat(::Type{<:types.int}, v, dims) = isempty(dims) ? CanonicalExpr(builtin.rep_array, v, 1) : v
_pmx_flat(::Type{<:types.real}, v, dims) = v
_pmx_flat(::Type, v, dims) = CanonicalExpr(builtin.to_array_1d, v)
# The adapter's expression re-reading a capture at `a[o + first]` onward, where
# `o` is the length of the user's own part of `a`.
_pmx_unpack(::Type{<:types.real}, dims, a, o, first) =
    isempty(dims) ? :($a[$o + $first]) : _pmx_segment(a, o, first, only(dims))
_pmx_unpack(::Type{<:types.vector}, dims, a, o, first) =
    :(to_vector($(_pmx_segment(a, o, first, only(dims)))))
_pmx_unpack(::Type{<:types.row_vector}, dims, a, o, first) =
    :(to_row_vector($(_pmx_segment(a, o, first, only(dims)))))
_pmx_unpack(::Type{<:types.matrix}, (m, n), a, o, first) =
    :(to_matrix($(_pmx_segment(a, o, first, m * n)), $m, $n))
_pmx_segment(a, o, first, len) = :($a[($o + $first):($o + $(first + len - 1))])
_pmx_fresh(base, taken) = begin
    name = base
    i = 1
    while name in taken
        name = Symbol(base, :_, i)
        i += 1
    end
    push!(taken, name)
    name
end
_pmx_theta_with(theta, ::Nothing) = theta
# Torsten refuses an empty `theta` ("the number of parameters per event is 0"),
# so a short closure with nothing to pack gets a placeholder it never reads.
_pmx_theta_with(::Nothing, ::Nothing) = CanonicalExpr(builtin.rep_array, 0.0, 1)
_pmx_theta_with(::Nothing, packed) = packed
_pmx_theta_with(theta, packed) = stan_ndim(theta) == 2 ?
    CanonicalExpr(builtin.pmx_append_each, theta, packed) :
    CanonicalExpr(builtin.append_array, theta, packed)
# A per-event `x_r`/`x_i` argument carrying `packed` on every row: the call's own
# rows extended, or `packed` repeated once per event (`empty` columns without it).
_pmx_rows_with(x, ::Nothing, ::Nothing, empty) =
    CanonicalExpr(builtin.rep_array, empty, CanonicalExpr(size, x.args[3]), 0)
_pmx_rows_with(x, rows, ::Nothing, _) = rows
_pmx_rows_with(x, ::Nothing, packed, _) =
    CanonicalExpr(builtin.rep_array, packed, CanonicalExpr(size, x.args[3]))
_pmx_rows_with(x, rows, packed, _) = CanonicalExpr(builtin.pmx_append_each, rows, packed)

# ── Generalised `jbroadcasted(f, a1, …, ak)` — trace-level element loop ──
# Applies `f` element-wise over its iterated args (any arity, any position):
# iterated args (anything with a leading dimension — `vector`/`row_vector`,
# `array[] T`, `matrix`) are indexed per element, scalar args pass through, and
# the loop size comes from the first iterated arg. The output CONTAINER is
# INFERRED from `f`'s per-element return type — an `int` element gives
# `array[] int`, a `real` element gives `vector[n]` — so int-array broadcasting
# stays int while the real path (every `*_lpmfs`/`*_lpdfs` caller, whose data /
# location args are native `vector`s) is unchanged. The iterability predicate is
# `stan_ndim >= 1` (NOT `l_ndim`, which is `array[]`-only and misses `vector`s —
# that gap silently broke the vectorised `normal(vector, real)` likelihood). The
# Stan function is generated per call shape (mirrors the @deffun body build + the
# reduce_sum trace-level fundef); `broadcasted_getindex` does the per-arg slice.
_jb_iterated(a) = stan_ndim(type(a)) >= 1
_jb_elem(a, context) = _jb_iterated(a) ?
    _stan_expr(CanonicalExpr(getindex, a, stan_expr(1, 1)), context) : a
_jb_semantic_rng(f::StanExpr2{<:types.func}) =
    endswith(string(nameof(type(f).info.value)), "_rng")
_jb_semantic_rng(_) = false
_jb_parent(f) = _jb_semantic_rng(f) ? jbroadcasted_rng : jbroadcasted
# Stan permits RNG calls only from generated quantities or from a user-defined
# function whose own name ends `_rng`. A broadcasted RNG cell therefore uses a
# semantic-suffix receiver for BOTH the call site and its generated definition;
# ordinary deterministic `jbroadcasted` names stay byte-identical.
func_name(::typeof(jbroadcasted), args) = invoke(
    func_name,
    Tuple{Any,Any},
    _jb_parent(first(args)),
    args,
)
_jb_infer(x::CanonicalExpr, context=nothing) = begin
    context = _context_or_new(context)
    f, dargs = x.args[1], x.args[2:end]
    ai = findfirst(_jb_iterated, dargs)
    isnothing(ai) && error("jbroadcasted: needs at least one iterated (vector/array) argument")
    n = stan_size(type(dargs[ai]), 1)
    elem_rt = type(_stan_expr(CanonicalExpr(f, map(a -> _jb_elem(a, context), dargs)...), context))
    stan_ndim(elem_rt) == 0 || error("jbroadcasted: `f` must return a scalar per element (got `$(sigtype(elem_rt))`)")
    # Preserve the exact int-family element center type: `int` stays `array[] int`
    # and `bool` (a comparison result) stays `array[] bool` (a mask), so
    # `cmt .== 1` is boolean-mask-indexable; a `real` element → `vector[n]`.
    container = center_type(elem_rt) <: types.int ? center_type(elem_rt) : types.vector
    (; f, dargs, ai, n, container)
end
tracetype(x::CanonicalExpr{typeof(jbroadcasted)}) = _tracetype(x, nothing)
_tracetype(x::CanonicalExpr{typeof(jbroadcasted)}, context) = begin
    inf = _jb_infer(x, context)
    StanType(inf.container, (inf.n,))
end
fundef(x::CanonicalExpr{typeof(jbroadcasted)}) = _fundef(x, nothing)
_fundef(x::CanonicalExpr{typeof(jbroadcasted)}, context) = begin
    inf = _jb_infer(x, context)
    k = length(inf.dargs)
    argnames = [Symbol("x", i) for i in 1:k]
    f_ph = anon_expr(:f, inf.f)
    arg_phs = [anon_expr(argnames[i], inf.dargs[i]) for i in 1:k]
    n_expr = StanExpr(:n, StanType(types.int, ()))
    # `nameof` gives the parseable type-token symbol for the body decl / return:
    # `:int`, `:vector`, or `:bool` (which renders as `int`). All are valid tokens.
    container_sym = nameof(inf.container)
    slices = [
        _jb_iterated(inf.dargs[j]) ?
            :($broadcasted_getindex($(argnames[j]), i)) : argnames[j]
        for j in 1:k
    ]
    body_ast = Expr(:block,
        :(rv :: $container_sym[n]),
        Expr(:for, :(i = 1:n), Expr(:block, :(rv[i] = f($(slices...))))),
        :rv,
    )
    info = OrderedDict{Symbol,Any}(:f => f_ph, :n => n_expr)
    for i in 1:k
        info[argnames[i]] = arg_phs[i]
    end
    _attach_trace_context!(info, context)
    info[:__mod__] = parentmodule(typeof(jbroadcasted))
    n_decl = string("int n = dims(", argnames[inf.ai], ")[1];")
    StanFunction3(
        "", StanType(inf.container, (n_expr,)), _jb_parent(inf.f),
        (; f=f_ph, (argnames[i] => arg_phs[i] for i in 1:k)...),
        [n_decl, forward!(canonical(ensure_xreturn(body_ast)); info)],
    )
end

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
