builtin_module_names(x::Symbol) = x
builtin_module_names(x::Expr) = mapreduce(builtin_module_names, vcat, x.args)
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
    flat
    std_normal
    normal
    cauchy
    binomial_logit
    lognormal
    chi_square
    inv_chi_square
    scaled_inv_chi_square
    exponential
    gamma
    inv_gamma
    weibull
    frechet
    rayleigh
    loglogistic
    uniform
    beta
    beta_proportion
    von_mises
    multi_normal
    multi_normal_prec
    multi_normal_cholesky
    multi_gp
    multi_gp_cholesky
    multi_student_t
    multi_student_t_cholesky
    gaussian_dlm_obs
    dirichlet
    lkj_corr
    lkj_corr_cholesky
    wishart
    inv_wishart
    inv_wishart_cholesky
    wishart_cholesky
    neg_binomial_2
    std_normal_rng
    normal_rng
    exponential_rng
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
    inv_logit
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
    cumulative_sum
    reduce_sum
    log_sum_exp
    lgamma
    matrix_exp
    log1p_exp log1m_exp
    sort_asc sort_desc
    sort_indices_asc sort_indices_desc
    dot_product rows_dot_product
] 
function vector_std_normal_rng end

autokwargs(::CanonicalExpr{<:Union{typeof.((beta, beta_proportion))...}}) = (;lower=0, upper=1)
autokwargs(::CanonicalExpr{typeof(von_mises)}) = (;lower=0, upper=2pi)
autokwargs(x::CanonicalExpr{typeof(uniform)}) = (;lower=x.args[1], upper=x.args[2])
autokwargs(::CanonicalExpr{<:Union{typeof.((lognormal,chi_square,inv_chi_square,scaled_inv_chi_square,exponential,gamma,inv_gamma,weibull,frechet,rayleigh,loglogistic))...}}) = (;lower=0.)

@deffun begin 
    Base.print(x)::anything
    reject(x)::anything
    Base.size(x)::int
    Base.range(start::int, stop::int)::vector[stop]
    Base.sum(x)::real
    Base.sum(x::int[m])::int
    Base.sum(x::int[m,n])::int
    Base.sum(x::int[m,n,o])::int
    cumulative_sum(x::int[m])::int[m]
    cumulative_sum(x::real[m])::real[m]
    cumulative_sum(x::vector[m])::vector[m]
    linspaced_array(n, x, y)::real[n]
    linspaced_vector(n, x, y)::vector[n]
    to_matrix(v::anything, m, n)::matrix[m,n]
    rep_array(x::int, n)::int[n]
    rep_vector(v, n)::vector[n]
    rep_matrix(v::vector[m], n)::matrix[m, n]
    rep_matrix(x::real, m, n)::matrix[m,n]
    to_array_2d(v, m, n)::real[m,n]
    rows_dot_product(x::matrix[m,n], y::matrix[m,n])::vector[m]
    append_col(x::vector[n], y::vector[n])::matrix[n,2]
    append_col(x::matrix[m, n1], y::matrix[m, n2])::matrix[m, n1+n2]
    append_col(x::vector[m], y::matrix[m, n2])::matrix[m, 1+n2]
    dot_product(x::vector[n], y::vector[n])::real
    Base.:\(A::matrix[m, m], b::vector[m])::vector[m]
    matrix_exp(x::matrix[m,m])::matrix[m,m]
    append_array(lhs::anything[m],rhs::anything[n])::real[m+n]
    append_array(lhs::anything[m],rhs::real)::real[m+1]
    append_row(lhs::vector[m],rhs::real)::vector[m+1]
    append_row(lhs::vector[m],rhs::vector[n])::vector[m+n]
    vector_std_normal_rng(n::int)::vector[n] = begin
        rv::vector[n]
        for i in 1:n
            rv[i] = std_normal_rng()
        end
        rv
    end
    normal_lpdf(obs::anything, loc::anything, scale::anything)::real
    normal_lpdfs(obs::anything[n], loc::anything[n], scale::anything[n])::vector[n] = begin
        rv::vector[n]
        for i in 1:n
            rv[i] = normal_lpdf(obs[i], loc[i], scale[i])
        end
        rv
    end
    normal_lpdfs(obs::vector[n], loc::vector[n], scale::real)::vector[n] = begin
        rv::vector[n]
        for i in 1:n
            rv[i] = normal_lpdf(obs[i], loc[i], scale)
        end
        rv
    end
    vector_exponential_rng(rate::real, n::int)::vector[n] = begin
        rv::vector[n]
        for i in 1:n
            rv[i] = exponential_rng(rate)
        end
        rv
    end
    dirichlet_lpdf(w::simplex[n], alpha::vector[n])::real
    lkj_corr_lpdf(L::corr_matrix, x::real)::real
    lkj_corr_cholesky_lpdf(L::cholesky_factor_corr, x::real)::real
    wishart_lpdf(L::cov_matrix[m], x::real, sigma::matrix[m,m])::real
    wishart_cholesky_lpdf(L::cholesky_factor_cov[m], x::real, sigma::matrix[m,m])::real
end
@defsig begin 
    Union{typeof.((sqrt, exp, log, sin, cos, asin, acos, log1m, inv_logit, log_inv_logit, log1m_exp, expm1, Phi, lgamma, abs, log1p_exp, log1m_exp))...} => begin 
        (real,)=>real
        (vector[n],)=>vector[n]
        (real[n],)=>real[n]
        (matrix[m,n],)=>matrix[m,n]
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
    end
    typeof(std_normal_rng) => begin 
        () => real
    end
    typeof(normal_rng) => begin 
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
    typeof(!=) => begin 
        (anything, anything) => int
    end
end

const TolODESolver = Union{typeof.((ode_rk45_tol, ode_ckrk_tol, ode_adams_tol, ode_bdf_tol))...}
const NoTolODESolver = Union{typeof.((ode_rk45, ode_ckrk, ode_adams, ode_bdf))...}
const ODESolver = Union{TolODESolver, NoTolODESolver}

tracetype(x::CanonicalExpr{<:ODESolver}) = StanType(
    types.vector, (stan_size(x.args[4], 1), stan_size(x.args[2], 1))
)

fetch_functions!(x::CanonicalExpr{<:TolODESolver}; info) = fetch_functions!(
    CanonicalExpr(x.args[1], x.args[3], x.args[2], x.args[8:end]...); info
)

fetch_functions!(x::CanonicalExpr{<:NoTolODESolver}; info) = fetch_functions!(
    CanonicalExpr(x.args[1], x.args[3], x.args[2], x.args[5:end]...); info
)
# fundefexprs(::CanonicalExpr{<:StanExpr2{types.func}}) = error()