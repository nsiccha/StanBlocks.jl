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
macro builtin_module(x)
    @assert Meta.isexpr(x, :vcat)
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
    neg_binomial_2_lpdf
    bernoulli_lpmf
    bernoulli_logit_lpmf 
    bernoulli_logit_glm_lpmf
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
    diag_matrix
    mdivide_left_tri_low
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
] 

autokwargs(::CanonicalExpr{<:Union{typeof.((beta, beta_proportion))...}}) = (;lower=0, upper=1)
autokwargs(::CanonicalExpr{typeof(von_mises)}) = (;lower=0, upper=2pi)
autokwargs(x::CanonicalExpr{typeof(uniform)}) = (;lower=x.args[1], upper=x.args[2])
autokwargs(::CanonicalExpr{<:Union{typeof.((lognormal,chi_square,inv_chi_square,scaled_inv_chi_square,exponential,gamma,inv_gamma,weibull,frechet,rayleigh,loglogistic))...}}) = (;lower=0.)

@deffun begin 
    reduce_sum(args...)::real
    reduce_sum_static(args...)::real
    simple_reduce_sum(f, x, args...)::real = reduce_sum(simple_reduce_sum_helper, x, 1, f, args...)
    simple_reduce_sum_helper(x_slice::anything[n], slice_start, slice_end, f, args...)::real = begin 
        rv = 0.
        for i in 1:n
            rv += f(x_slice[i], args...)
        end
        rv
    end
    positive_infinity()::real
    negative_infinity()::real
    reject(x)::anything
    Base.print(x)::anything
    Base.size(x)::int
    Base.range(start::int, stop::int)::vector[stop]
    Base.sum(x)::real
    Base.sum(x::int[m])::int
    Base.sum(x::int[m,n])::int
    Base.sum(x::int[m,n,o])::int
    Base.:\(A::matrix[m, m], b::vector[m])::vector[m]
    dims(x::anything[_])::int[1]
    dims(x::anything[_, _])::int[2]
    dims(x::anything[_, _, _])::int[3]
    cols(::matrix[m,n])::int
    rows(::matrix[m,n])::int
    cumulative_sum(x::int[m])::int[m]
    cumulative_sum(x::real[m])::real[m]
    cumulative_sum(x::vector[m])::vector[m]
    diag_matrix(x::anything[n])::matrix[n,n]
    linspaced_array(n, x, y)::real[n]
    linspaced_vector(n, x, y)::vector[n]
    to_matrix(v, m, n)::matrix[m,n]
    rep_array(x::int, n)::int[n]
    rep_array(x::real, n)::real[n]
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
    neg_binomial_2_lpdf(args...)
    bernoulli_lpmf(args...)
    bernoulli_logit_lpmf(args...)
    bernoulli_logit_glm_lpmf(args...)
    multi_normal_lpdf(obs::vector[n], loc::vector[n], cov)
    dirichlet_lpdf(w::simplex[n], alpha::vector[n])
    lkj_corr_lpdf(L::corr_matrix, x::real)
    lkj_corr_cholesky_lpdf(L::cholesky_factor_corr, x::real)
    wishart_lpdf(L::cov_matrix[m], x::real, sigma::matrix[m,m])
    wishart_cholesky_lpdf(L::cholesky_factor_cov[m], x::real, sigma::matrix[m,m])

    lognormal_rng(loc::real, scale::real)::real
    student_t_rng(nu::real, loc::real, scale::real)::real
    multi_normal_rng(loc::vector[n], args...)::vector[n]
    multi_normal_cholesky_rng(loc::vector[n], scale)::vector[n]
    bernoulli_rng(::vector[n])::int[n]
    bernoulli_logit_rng(::vector[n])::int[n]
    bernoulli_logit_glm_rng(X::matrix[m,n], alpha, beta)::int[m]
    bernoulli_logit_glm_rng(X::matrix[m,n], alpha::real, beta) = bernoulli_logit_glm_rng(X, rep_vector(alpha, m), beta)
    beta_rng(args...)::real
    binomial_rng(args...)::int
    binomial_logit_rng(n::int[m], p::vector[m])::int[m]

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
    vector_std_normal_rng(n::int)::vector[n] = to_vector(normal_rng(rep_vector(0, n), 1))
    bernoulli_lpmfs(args...) = bernoulli_lpmf(args...)
    bernoulli_lpmfs(obs::anything[n], args...) = jbroadcasted(bernoulli_lpmfs, obs, args...)
    bernoulli_logit_lpmfs(args...) = bernoulli_logit_lpmf(args...)
    bernoulli_logit_lpmfs(obs::anything[n], args...) = jbroadcasted(bernoulli_logit_lpmfs, obs, args...)
    bernoulli_logit_glm_lpmfs(y::int[n], X, alpha, beta) = bernoulli_logit_lpmfs(y, alpha + X * beta) 
    binomial_lpmfs(args...) = binomial_lpmf(args...)
    binomial_lpmfs(y::int[n], args...) = jbroadcasted(binomial_lpmfs, y, args...)
    binomial_logit_lpmfs(args...) = binomial_logit_lpmf(args...)
    binomial_logit_lpmfs(y::int[n], args...) = jbroadcasted(binomial_logit_lpmfs, y, args...)
    normal_lpdfs(args...) = normal_lpdf(args...)
    normal_lpdfs(obs::anything[n], loc, scale) = jbroadcasted(normal_lpdfs, obs, loc, scale)
    multi_normal_lpdfs(args...) = multi_normal_lpdf(args...)
    vector_exponential_rng(rate::real, n::int)::vector[n] = exponential_rng(rep_vector(rate, n))
    lkj_corr_cholesky_rng(n::int, eta::real)
    
    normal_cdf(args...)
    normal_lcdf(args...)
    normal_lccdf(args...)


    Base.invperm(x::int[n])::int[n] = begin 
        rv = rep_array(0, n)
        for i in 1:n
            rv[x[i]] = i
        end
        rv
    end
    # Base.get(x::anything[n], i, d) = if 1 <= i <= n
    #     x[i]
    # else
    #     d
    # end
end
@defsig begin 
    Union{typeof.((sqrt, exp, log, log10, sin, cos, asin, acos, log1m, inv_logit, log_inv_logit, log1m_exp, expm1, Phi, lgamma, abs, log1p_exp, log1m_exp, Base.inv, Base.log1p))...} => begin 
        (real,)=>real
        (vector[n],)=>vector[n]
        (row_vector[n],)=>row_vector[n]
        (real[n],)=>real[n]
        (matrix[m,n],)=>matrix[m,n]
    end
    typeof(÷) => begin 
        (int, int) => int
    end
    Union{typeof.((+, -, ^, *, /))...} => begin 
        (real,) => real
        (vector[n],) => vector[n]
        # (real[n],) => real[n]
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
        # (matrix[m,n], int) => row_vector[n]
    end
    typeof(to_vector) => begin 
        (vector[n],) => vector[n]
        (real[n],) => vector[n]
        (matrix[m,n],) => vector[m*n]
    end
    typeof(to_array_1d) => begin 
        (vector[n],)=>real[n]
        (real[m,n],) => real[m*n]
        (int[m,n],) => int[m*n]
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
    Union{typeof.((min,max))...} => begin 
        (int, int) => int
        (int[n],)=>int
        (vector[n],)=>real
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
    typeof(add_diag) => begin 
        (matrix[n,n], real) => matrix[n,n]
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
    Union{typeof.((sort_asc, sort_desc))...} => begin 
        (int[n],)=>int[n]
        (real[n],)=>real[n]
        (vector[n],)=>vector[n]
    end
    Union{typeof.((sort_indices_asc, sort_indices_desc))...} => begin 
        (anything[n],)=>int[n]
    end
    Union{typeof.((|, &, ==, !=, <, <=, >, >=))...} => begin 
        (anything, anything) => int
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
    CanonicalExpr(x.args[1], x.args[3], x.args[2], x.args[8:end]...); info
)

fetch_functions!(x::CanonicalExpr{<:NoTolODESolver}; info) = fetch_functions!(
    CanonicalExpr(x.args[1], x.args[3], x.args[2], x.args[5:end]...); info
)

function reduce_sum_reconstruct end
function reduce_sum_deconstruct end
fetch_functions!(x::CanonicalExpr{<:ReduceSumFunction}; info) = begin
    fetch_functions!(
        CanonicalExpr(x.args[1], x.args[2], stan_expr(1), stan_expr(1), x.args[4:end]...); info
    )
    if any(arg->isa(arg, StanExpr2{<:types.tup}), x.args) 
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
fundef(x::CanonicalExpr{typeof(reduce_sum_reconstruct)}) = if any(arg->isa(arg, StanExpr2{<:types.tup}), x.args) 
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
Base.show(io::IO, x::CanonicalExpr{<:ReduceSumFunction}) = if any(arg->isa(arg, StanExpr2{<:types.tup}), x.args) 
    # Work around https://github.com/stan-dev/math/issues/3041
    print(io, CanonicalExpr(reduce_sum_deconstruct, stan_expr(reduce_sum_reconstruct), x.args[2], x.args[3], x.args[1], x.args[4:end]...))
else
    autoprint(io, head(x), "(", Join(
        (func_name(x.args[1], x.args[2:end]), filter(!always_inline, x.args[2:end])...), ", "
    ), ")")
end