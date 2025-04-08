
function flat end
function std_normal end
function std_normal_rng end
function normal end
function normal_rng end

autotype(args...) = missing
autocons(args...) = (;)
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
autocons(::StanExpr{<:Union{typeof.((lognormal,chi_square,inv_chi_square,scaled_inv_chi_square,exponential,gamma,inv_gamma,weibull,frechet,rayleigh,loglogistic))...}}, args...) = (;lower=0.)
function uniform end
autocons(::StanExpr{typeof(uniform)}, lower, upper) = (;lower, upper)
function beta end
function beta_proportion end
autocons(::StanExpr{<:Union{typeof.((beta, beta_proportion))...}}, args...) = (;lower=0, upper=1)
function von_mises end
autocons(::StanExpr{<:Union{typeof.((von_mises,))...}}, args...) = (;lower=0, upper=2pi)

function multi_normal end
function multi_normal_prec end
function multi_normal_cholesky end
function multi_gp end
function multi_gp_cholesky end
function multi_student_t end
function multi_student_t_cholesky end
function gaussian_dlm_obs end
autotype(::StanExpr{<:Union{typeof.((multi_normal,multi_normal_prec,multi_normal_cholesky,multi_gp,multi_gp_cholesky,multi_student_t,multi_student_t_cholesky,gaussian_dlm_obs))...}}, args...) = :vector
function dirichlet end
autotype(::StanExpr{typeof(dirichlet)}, args...) = :dirichlet
function lkj_corr end
autotype(::StanExpr{typeof(lkj_corr)}, args...) = :corr_matrix
function lkj_corr_cholesky end
autotype(::StanExpr{typeof(lkj_corr_cholesky)}, args...) = :cholesky_factor_corr
function wishart end
function inv_wishart end
autotype(::StanExpr{<:Union{typeof.((wishart,inv_wishart))...}}, args...) = :cov_matrix
function inv_wishart_cholesky end
function wishart_cholesky end
autotype(::StanExpr{typeof(wishart_cholesky)}, args...) = :cholesky_factor_cov
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


traceutype(::StanExpr{typeof(linspaced_array)}, n, x, y) = (:real, (n, ))
traceutype(::StanExpr{typeof(linspaced_vector)}, n, x, y) = (:vector, (n, ))
traceutype(::StanExpr{typeof(to_matrix)}, v, m, n) = (:matrix, (m, n))
traceutype(::StanExpr{typeof(rep_vector)}, v, n) = (:vector, (n,))
traceutype(::StanExpr{typeof(rep_matrix)}, v, n) = (:matrix, (type(v).size[1], n))
traceutype(::StanExpr{typeof(to_array_2d)}, v, m, n) = (:real, (m, n))
traceutype(::StanExpr{typeof(getindex)}, m::StanExpr2{:matrix, 2}, i::StanExpr{Colon}, j::StanExpr2{:int, 0}) = (:vector, (type(m).size[1],))


xref(args...) = Expr(:ref, args...)
xtuple(args...) = Expr(:tuple, args...)
xutype(x::Symbol) = xutype(Expr(:ref, x))
xutype(x::Expr) = xtuple(Meta.quot(x.args[1]), xtuple(x.args[2:end]...))
defsig_expr(x::LineNumberNode) = x
defsig_expr(x::Expr) = if x.head == :block
    Expr(:block, defsig_expr.(x.args)...)
elseif x.head == :call
    @assert x.args[1] == :(=>)
    _, ftype, rhs = x.args
    @assert Meta.isexpr(rhs, :block)
    mod = @__MODULE__
    defs = map(rhs.args) do sig
        isa(sig, LineNumberNode) && return sig
        @assert Meta.isexpr(sig, :call)
        @assert sig.args[1] == :(=>)
        @assert Meta.isexpr(sig.args[2], :tuple)
        args = xutype.(sig.args[2].args)
        args = (zip(Symbol.(:arg, 1:length(args)), args))
        rv = xutype(sig.args[3])
        sig = (:(::$mod.StanExpr{<:$ftype}), [
            :($xarg::$mod.StanExpr2{$(arg.args[1]), $(length(arg.args[2].args))})
            for (xarg, arg) in (args)
        ]...)
        body = Expr(:block, [
            Expr(:(=), arg.args[2], :($mod.type($xarg).size))
            for (xarg, arg) in (args)
        ]..., rv)
        return :($mod.traceutype($(sig...)) = $body)
    end
    Expr(:block, defs...)
else
    dumperror(x)
end 
macro defsig(x)
    esc(defsig_expr(x))
end
xusig(x::Expr) = begin 
    @assert x.head == :(::)
    # x.args[1], xutype(x.args[2])
    x.args
end
sigtype(x::Symbol) = sigtype(xref(x))
sigtype(x::Expr) = begin
    @assert x.head == :ref
    sigtype(StanType(:data, x.args[1], data.((x.args[2:end]...,), 0)))
end
sigarg(x::Expr) = begin
    @assert x.head == :(::)
    "$(sigtype(x.args[2])) $(x.args[1])"
end
function funbody end
funbody(x::Expr) = begin 
    @assert x.head == :block
    funbody(x.args)
end
funbody(x::AbstractVector) = join(map(funbody, x), "\n")
funbody(x::LineNumberNode) = ""
funbody(x::String) = x
fundef_expr(x::LineNumberNode) = x
fundef_expr(x::Expr) = if x.head == :block
    Expr(:block, fundef_expr.(x.args)...)
else
    mod = @__MODULE__
    @assert x.head == :(=)
    lhs, rhs = x.args
    fcall, rv = xusig(lhs)
    @assert Meta.isexpr(fcall, :call)
    f, args... = fcall.args
    arg_names = map(arg->arg.args[1], args)
    arg_types = map(arg->arg.args[2], args)
    sig_rv = sigtype(rv)
    sig_args = join(sigarg.(args), ", ")
    fun_sizes = OrderedDict()
    for arg in args
        arg_name, arg_type = xusig(arg)
        Meta.isexpr(arg_type, :ref) || continue
        for (i, dim_name) in enumerate(arg_type.args[2:end])
            @assert isa(dim_name, Symbol)
            fun_sizes[dim_name] = "int $dim_name = dims($arg_name)[$i];"
        end
    end
    sig = (:(::$mod.StanExpr{typeof($f)}), [
        :($xarg::$mod.StanExpr2{$(arg.args[1]), $(length(arg.args[2].args))})
        for (xarg, arg) in zip(arg_names, xutype.(arg_types))
    ]...)
    stan_fundef = """
    $sig_rv $f($sig_args){
        $(funbody(collect(values(fun_sizes))))
        $(funbody(rhs))
    }
    """
    extra = if endswith(string(f), r"lp[md]f")
        fbase = Symbol(string(f)[1:end-length("_lpdf")])
        quote 
            function $fbase end
            $mod.@defsig typeof($fbase) => begin 
                ($(arg_types[2:end]...),) => $(arg_types[1])
            end
            $mod.fundef(::$mod.StanExpr{typeof($fbase)}, $(sig[3:end]...)) = $stan_fundef
        end
    else
        nothing
    end
    quote 
        function $f end
        $mod.@defsig typeof($f) => begin 
            ($(arg_types...),) => $rv 
        end
        $mod.fundef($(sig...)) = $stan_fundef
        $extra
    end
end
macro fundef(x)
    esc(fundef_expr(x))
end
@defsig begin 
    Union{typeof.((sqrt, exp, log, sin, cos, asin, acos, log1m, inv_logit, log_inv_logit, log1m_exp, Phi))...} => begin 
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
        (vector[m], int[n]) => vector[n]
        (vector[m], int) => real
        (vector[m,n], int) => vector[n]
        (matrix[m,n], int[o], int) => vector[o]
        (matrix[m,n], int, int[p]) => row_vector[p]
        (matrix[m,n], int[o], int[p]) => matrix[o, p]
        # (matrix[m,n], int) => row_vector[n]
    end
    typeof(to_vector) => begin 
        (vector[n],) => vector[n]
        (real[n],) => vector[n]
        (matrix[m,n],) => vector[tracecall(trace(:*;info=(;)), m,n)]
    end
    typeof(to_array_1d) => begin 
        (vector[n],)=>real[n]
        (real[m,n],) => real[tracecall(trace(:*;info=(;)), m,n)]
    end
    typeof(std_normal) => begin 
        () => real
    end
    typeof(std_normal_rng) => begin 
        () => real
    end
    typeof(normal) => begin 
        (real, real) => real
    end
    typeof(lognormal) => begin 
        (real, real) => real
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
    Base.BroadcastFunction => begin 
        (real, real) => real
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
    # EXTENSION FUNCTIONS
    typeof(eachcol) => begin 
        (matrix[m,n],) => vector[n,m]
    end
    typeof(eachrow) => begin 
        (matrix[m,n],) => row_vector[m, n]
    end
end

fundef(args...; kwargs...) = nothing
fundef(::StanExpr{typeof(eachcol)}, m) = """
    array[] vector eachcol(matrix X){
        array[2] int mn = dims(X);
        int m = mn[1];
        int n = mn[2];
        array[n] vector[m] rv;
        for(i in 1:n){
            rv[i,:] = X[:,i];
        }
        return rv;
    }
"""