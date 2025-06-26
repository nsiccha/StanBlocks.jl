module builtin
function flat end
function std_normal end
function normal end
function cauchy end
# function binomial end
function binomial_logit end
function lognormal end
function chi_square end
function inv_chi_square end
function scaled_inv_chi_square end
function exponential end
function gamma end
function inv_gamma end
function weibull end
function frechet end
function rayleigh end
function loglogistic end
function uniform end
function beta end
function beta_proportion end
function von_mises end
function multi_normal end
function multi_normal_prec end
function multi_normal_cholesky end
function multi_gp end
function multi_gp_cholesky end
function multi_student_t end
function multi_student_t_cholesky end
function gaussian_dlm_obs end
function dirichlet end
function lkj_corr end
function lkj_corr_cholesky end
function wishart end
function inv_wishart end
function inv_wishart_cholesky end
function wishart_cholesky end
function neg_binomial_2 end

function std_normal_rng end
function normal_rng end
function exponential_rng end

function log1m end
function to_vector end
function to_row_vector end
function to_matrix end
function rep_vector end
function rep_matrix end
function linspaced_array end
function linspaced_vector end
function to_array_1d end
function to_array_2d end
function cholesky_decompose end
function diag_pre_multiply end
function diag_post_multiply end
function mdivide_right_tri_low end
function add_diag end 
function gp_exp_quad_cov end 
function inv_logit end
function log_inv_logit end
function log1m_exp end
function Phi end
function integrate_ode_rk45 end
function ode_rk45 end
function ode_ckrk end
function ode_adams end
function ode_bdf end
function append_array end
function append_row end
function append_col end

function reduce_sum end
function log_sum_exp end
function lgamma end

end
const flat = builtin.flat
const std_normal = builtin.std_normal
const normal = builtin.normal
const cauchy = builtin.cauchy
const std_normal_rng = builtin.std_normal_rng
const normal_rng = builtin.normal_rng
const exponential_rng = builtin.exponential_rng
const binomial_logit = builtin.binomial_logit
const lognormal = builtin.lognormal
const chi_square = builtin.chi_square
const inv_chi_square = builtin.inv_chi_square
const scaled_inv_chi_square = builtin.scaled_inv_chi_square
const exponential = builtin.exponential
const gamma = builtin.gamma
const inv_gamma = builtin.inv_gamma
const weibull = builtin.weibull
const frechet = builtin.frechet
const rayleigh = builtin.rayleigh
const loglogistic = builtin.loglogistic
const uniform = builtin.uniform
const beta = builtin.beta
const beta_proportion = builtin.beta_proportion
const von_mises = builtin.von_mises
const multi_normal = builtin.multi_normal
const multi_normal_prec = builtin.multi_normal_prec
const multi_normal_cholesky = builtin.multi_normal_cholesky
const multi_gp = builtin.multi_gp
const multi_gp_cholesky = builtin.multi_gp_cholesky
const multi_student_t = builtin.multi_student_t
const multi_student_t_cholesky = builtin.multi_student_t_cholesky
const gaussian_dlm_obs = builtin.gaussian_dlm_obs
const dirichlet = builtin.dirichlet
const lkj_corr = builtin.lkj_corr
const lkj_corr_cholesky = builtin.lkj_corr_cholesky
const wishart = builtin.wishart
const inv_wishart = builtin.inv_wishart
const inv_wishart_cholesky = builtin.inv_wishart_cholesky
const wishart_cholesky = builtin.wishart_cholesky
const neg_binomial_2 = builtin.neg_binomial_2
const log1m = builtin.log1m
const to_vector = builtin.to_vector
const to_row_vector = builtin.to_row_vector
const to_matrix = builtin.to_matrix
const rep_vector = builtin.rep_vector
const rep_matrix = builtin.rep_matrix
const linspaced_array = builtin.linspaced_array
const linspaced_vector = builtin.linspaced_vector
const to_array_1d = builtin.to_array_1d
const to_array_2d = builtin.to_array_2d
const cholesky_decompose = builtin.cholesky_decompose
const diag_pre_multiply = builtin.diag_pre_multiply
const diag_post_multiply = builtin.diag_post_multiply
const mdivide_right_tri_low = builtin.mdivide_right_tri_low
const add_diag = builtin.add_diag 
const gp_exp_quad_cov = builtin.gp_exp_quad_cov 
const inv_logit = builtin.inv_logit
const log_inv_logit = builtin.log_inv_logit
const log1m_exp = builtin.log1m_exp
const Phi = builtin.Phi
const integrate_ode_rk45 = builtin.integrate_ode_rk45
const ode_rk45 = builtin.ode_rk45
const ode_ckrk = builtin.ode_ckrk
const ode_adams = builtin.ode_adams
const ode_bdf = builtin.ode_bdf
const append_array = builtin.append_array
const append_row = builtin.append_row
const append_col = builtin.append_col

const reduce_sum = builtin.reduce_sum
const log_sum_exp = builtin.log_sum_exp
const lgamma = builtin.lgamma

function vector_std_normal_rng end

autokwargs(::CanonicalExpr{<:Union{typeof.((beta, beta_proportion))...}}) = (;lower=0, upper=1)
autokwargs(::CanonicalExpr{typeof(von_mises)}) = (;lower=0, upper=2pi)
autokwargs(x::CanonicalExpr{typeof(uniform)}) = (;lower=x.args[1], upper=x.args[2])
autokwargs(::CanonicalExpr{<:Union{typeof.((lognormal,chi_square,inv_chi_square,scaled_inv_chi_square,exponential,gamma,inv_gamma,weibull,frechet,rayleigh,loglogistic))...}}) = (;lower=0.)

@deffun begin 
    Base.range(start::int, stop::int)::vector[stop]
    linspaced_array(n, x, y)::real[n]
    linspaced_vector(n, x, y)::vector[n]
    to_matrix(v::anything, m, n)::matrix[m,n]
    rep_vector(v, n)::vector[n]
    rep_matrix(v::vector[m], n)::matrix[m, n]
    to_array_2d(v, m, n)::real[m,n]
    append_array(lhs::anything[m],rhs::anything[n])::real[m+n]
    append_array(lhs::anything[m],rhs::real)::real[m+1]
    append_row(lhs::vector[m],rhs::real)::vector[m+1]
    vector_std_normal_rng(n::int)::vector[n] = """
        vector[n] rv;
        for(i in 1:n){
            rv[i] = std_normal_rng();
        }
        return rv;
    """
    normal_lpdf(obs::anything, loc::anything, scale::anything)::real
    normal_lpdfs(obs::anything[n], loc::anything[n], scale::anything[n])::vector[n] = """
        vector[n] rv;
        for(i in 1:n){
            rv[i] = normal_lpdf(obs[i] | loc[i], scale[i]);
        }
        return rv;
    """
    normal_lpdfs(obs::vector[n], loc::vector[n], scale::real)::vector[n] = """
        vector[n] rv;
        for(i in 1:n){
            rv[i] = normal_lpdf(obs[i] | loc[i], scale);
        }
        return rv;
    """
    # exponential_lpdf(obs::anything, rate::anything)::real
    vector_exponential_rng(rate::real, n::int)::vector[n] = """
        vector[n] rv;
        for(i in 1:n){
            rv[i] = exponential_rng(rate);
        }
        return rv;
    """
    dirichlet_lpdf(w::simplex[n], alpha::vector[n])::real
    lkj_corr_lpdf(L::corr_matrix, x::real)::real
    lkj_corr_cholesky_lpdf(L::cholesky_factor_corr, x::real)::real
    wishart_lpdf(L::cov_matrix[m], x::real, sigma::matrix[m,m])::real
    wishart_cholesky_lpdf(L::cholesky_factor_cov[m], x::real, sigma::matrix[m,m])::real
end
@defsig begin 
    Union{typeof.((sqrt, exp, log, sin, cos, asin, acos, log1m, inv_logit, log_inv_logit, log1m_exp, expm1, Phi, lgamma))...} => begin 
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
        (real[m,n], int) => real[n] 
        (real[m], int[n]) => real[n]
        (real[m,n], int[o], int) => real[o] 
        (vector[m], int[n]) => vector[n]
        (vector[m], int) => real
        (vector[m,n], int) => vector[n]
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
    typeof(integrate_ode_rk45) => begin 
        (anything, real[m], real, real[n], real[p], anything, anything)=>real[m,n]
        (anything, real[m], real, real[n], real[p], anything, anything, real, real, int)=>real[m,n]
    end
    typeof(log_sum_exp) => begin 
        (real[n], ) => real
        (matrix[m,n], ) => real
        (row_vector[n], ) => real
        (vector[n], ) => real
    end
end

const ODESolver = Union{typeof.((ode_rk45, ode_ckrk, ode_adams, ode_bdf))...}

tracetype(x::CanonicalExpr{<:ODESolver}) = StanType(
    types.vector, (type(x.args[4]).size[1], type(x.args[2]).size[1])
)

fundefexprs(x::CanonicalExpr{<:ODESolver}) = allfundefexprs(
    CanonicalExpr(x.args[1], x.args[3], x.args[2], x.args[5:end]...)
)
# fundefexprs(::CanonicalExpr{<:StanExpr2{types.func}}) = error()