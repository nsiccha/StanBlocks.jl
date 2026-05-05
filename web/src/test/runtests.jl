using TestModules
using Random, Statistics
using StanBlocks
using LogDensityProblems
import StanBlocks.stan: @deffun, full_cqual_eq, transpiles, compiles, stan_model, stan_code, instantiate
using PosteriorDB

# ──────────────────────────────────────────────────────────────────────────────
# Module-level definitions needed by tests (structs, @deffun, sub-models, etc.)
# ──────────────────────────────────────────────────────────────────────────────

# --- slic.jl helpers ---

msg(e::ErrorException) = e.msg
msg(e::AssertionError) = e.msg
msg(e::MethodError) = e.msg

@deffun begin
    simple_lpdf(y, x) = 0.
    simple_lpdfs(y, x) = 0.
    simple_rng(x) = 0.
    vararg_lpdf(y, args...) = 0.
    vararg_lpdfs(y, args...) = 0.
    vararg_rng(args...) = 0.

    my_lpdf(y, fargs...) = reject(1)
    my_lpdfs(args...) = reject(1)
    my_rng(args...) = reject(1)
    my_lpdf(y, ::typeof(simple), args...) = simple_lpdf(y, args...)
    my_lpdfs(y, ::typeof(simple), args...) = simple_lpdfs(y, args...)
    my_rng(::typeof(simple), args...) = simple_rng(args...)
    my_lpdf(y, ::typeof(vararg), args...) = vararg_lpdf(y, args...)

    fof_lpdf(y, f, args...) = my_lpdf(y, f, args...)
    fof_lpdfs(y, f, args...) = my_lpdfs(y, f, args...)
    fof_rng(f, args...) = my_rng(f, args...)
    srs2_lpdf(y, f, args...) = simple_reduce_sum(srs2_helper, rep_array(y, 1), f, args...)
    srs2_helper(y, f, args...) = my_lpdf(y, f, args...)
    srs2_lpdfs(y, f, args...) = 0.
    srs2_rng(f, args...) = 0.
end

# --- issue @deffun definitions (hoisted to module level) ---

@deffun begin
    issue10a(::vector[n]) = 0.
    issue10b(_::vector[n]) = 0.
end

# issue 12 sub-models
sm12a = @slic begin
    x ~ std_normal(;n)
    return x .* x
end
sm12b = @slic begin
    x ~ std_normal(;n)
    return x
end

# issue 15 sub-models
sm15a = @slic begin
    x = rep_vector(0., n)
    return x
end
sm15b = @slic begin
    x = rep_vector(0., n)
    xx = append_row(x, x)
    return xx
end

@deffun begin
    issue17_lpdf(y::vector[n]) = begin
        rv = 0.
        for i in 1:n
            rv += y[i]
        end
        for i in 1:n-1
            rv += y[i]
        end
        for i in 1:dims(y)[1]
            rv += y[i]
        end
        rv
    end
end

@deffun begin
    issue18_lpdf(y) = normal_cdf(y, 0, 1) + normal_lcdf(y, 0, 1) + normal_lccdf(y, 0, 1)
end

# issue 19 sub-models
sm19a = @slic begin
    x = rep_vector(0., n)
    xx = x .* x
    return xx
end
sm19b = @slic begin
    x = rep_vector(0., n)
    xx = append_row(x,x)
    return xx
end

# issue 20 module
module m20
    using StanBlocks
    @deffun begin
        f(x) = begin
            return x
        end
    end
    model = @slic begin
        x ~ std_normal(;n)
        return x
    end
    modela = @slic begin
        x ~ model(;n)
    end
    modelb = @slic begin
        x ~ model(;n)
        y = f(x)
    end
end

@deffun begin
    issue9_lpdf(x::vector[n], n) = 0.
    issue9_rng(n) = rep_vector(0., n)
end

# --- logdensity.jl helpers ---

_stan_std_normal(x) = -0.5*x^2
_stan_normal(x, mu, sigma) = -log(sigma) - 0.5*((x-mu)/sigma)^2
_stan_binomial(k, n, p) = k*log(p) + (n-k)*log(1.0-p)

# --- builtin_shapes.jl helpers ---

function run_model(model)
    problem = instantiate(model)
    theta = zeros(LogDensityProblems.dimension(problem))
    ld = LogDensityProblems.logdensity(problem, theta)
    return isfinite(ld)
end

function check_type(model, var::Symbol, expected_sigtype::AbstractString)
    code = stan_code(model)
    pat = Regex("\\b$(expected_sigtype)(?:\\[[^\\]]*\\])+ (?:\\w+ )?$var\\b")
    occursin(pat, code)
end

# ──────────────────────────────────────────────────────────────────────────────
# Tests
# ──────────────────────────────────────────────────────────────────────────────

# === slic.jl tests ===

@testset "slic: normal(loc,scale)" begin
    @test compiles(@slic (;obs=0.) begin
        loc ~ std_normal()
        scale ~ std_normal(;lower=0.)
        obs ~ normal(loc, scale)
    end)
end

@testset "slic: simple" begin
    @test compiles(@slic (;obs=0.) begin
        loc ~ std_normal()
        obs ~ simple(loc)
    end)
end

@testset "slic: vararg" begin
    @test compiles(@slic (;obs=0.) begin
        loc ~ std_normal()
        obs ~ vararg(loc)
    end)
end

@testset "slic: fof(simple)" begin
    @test compiles(@slic (;obs=0.) begin
        loc ~ std_normal()
        obs ~ fof(simple, loc)
    end)
end

@testset "slic: srs2(vararg)" begin
    @test compiles(@slic (;obs=0.) begin
        loc ~ std_normal()
        obs ~ srs2(vararg, loc)
    end)
end

@testset "slic: srs2(vararg, extra)" begin
    @test compiles(@slic (;obs=0.) begin
        loc ~ std_normal()
        obs ~ srs2(vararg, loc, (1, 2, 3))
    end)
end

@testset "slic: stan_model re-data" begin
    @test compiles(stan_model(@slic (;obs=randn(5)) begin
        loc ~ std_normal()
        scale ~ std_normal(;lower=0.)
        obs ~ normal(loc, scale)
    end)(;obs=randn(10)))
end

@deffun begin
    preconditioned_normal_lpdf(xi::matrix[m, n], loc::vector[m], scale::vector[m], prescale::matrix[m,m], n) = begin
        multi_normal_cholesky_lpdf(eachcol(xi), mdivide_left_tri_low(prescale, loc), mdivide_left_tri_low(prescale, diag_matrix(scale)))
    end
end

@testset "issue9" begin
    @test compiles(@slic (;n=10) begin
        x ~ issue9(n)
    end)
    @test compiles(@slic (;n=10, y=1.) begin
        x ~ issue9(n)
        y ~ vararg(x)
    end)
end

@testset "issue10" begin
    @test compiles(@slic (;n=10) begin
        y ~ std_normal(;n)
        x = issue10a(y)
    end)
    @test compiles(@slic (;n=10) begin
        y ~ std_normal(;n)
        x = issue10b(y)
    end)
end

@testset "issue12" begin
    @test stan_code(sm12a(quote
        return x
    end ; n=10, y=1.)) == stan_code(sm12b(; n=10, y=1.))
    @test compiles(@slic (;n=10, y=1.) begin
        x ~ sm12a(;n)
        y ~ simple(x)
    end)
    @test compiles(@slic (;n=10, y=1.) begin
        x ~ sm12b(;n)
        y ~ simple(x)
    end)
end

@testset "issue15" begin
    @test stan_code(sm15a(quote
        xx = append_row(x, x)
        return xx
    end ; n=10, y=1.)) == stan_code(sm15b(; n=10, y=1.))
    @test compiles(sm15a(;n=10, y=1.))
    @test compiles(sm15b(;n=10, y=1.))
end

@testset "issue17" begin
    @test compiles(@slic (;n=10, y=1.) begin
        x ~ issue17(;n)
        y ~ simple(x)
    end)
end

@testset "issue18" begin
    @test compiles(@slic (;n=10, y=1.) begin
        x ~ issue18(;n)
        y ~ simple(x)
    end)
end

@testset "issue19" begin
    @test compiles(@slic (;n=10, y=1.) begin
        x ~ sm19a(;n)
        y ~ simple(x)
    end)
    @test compiles(@slic (;n=10, y=1.) begin
        x ~ sm19b(;n)
        y ~ simple(x)
    end)
end

@testset "issue20" begin
    @test compiles(m20.model(;n=10))
    @test compiles(m20.modela(;n=10))
    @test compiles(m20.modelb(;n=10))
end

# === logdensity.jl tests ===

@testset "logdensity: unconstrained scalar" begin
    obs_val = 0.5
    problem = instantiate(stan_model(@slic (;obs=obs_val) begin
        mu ~ std_normal()
        obs ~ normal(mu, 1.0)
    end))
    @test LogDensityProblems.dimension(problem) == 1
    for mu_val in [-2.0, 0.0, 1.5]
        expected = _stan_std_normal(mu_val) + _stan_normal(obs_val, mu_val, 1.0)
        @test LogDensityProblems.logdensity(problem, [mu_val]) ≈ expected atol=1e-6
    end
end

@testset "logdensity: lower-bounded scalar" begin
    obs_val = 1.0
    problem = instantiate(stan_model(@slic (;obs=obs_val) begin
        sigma ~ std_normal(;lower=0.)
        obs ~ normal(0., sigma)
    end))
    @test LogDensityProblems.dimension(problem) == 1
    for θ_val in [-1.0, 0.0, 1.0]
        sigma_val = exp(θ_val)
        expected = (
            _stan_std_normal(sigma_val) + θ_val
            + _stan_normal(obs_val, 0.0, sigma_val)
        )
        @test LogDensityProblems.logdensity(problem, [θ_val]) ≈ expected atol=1e-6
    end
end

@testset "logdensity: interval-bounded scalar (beta prior)" begin
    k_val, n_val = 3, 10
    problem = instantiate(stan_model(@slic (;k=k_val, n=n_val) begin
        theta ~ beta(1., 1.)
        k ~ binomial(n, theta)
    end))
    @test LogDensityProblems.dimension(problem) == 1
    for θ_val in [-2.0, 0.0, 1.5]
        p = 1.0 / (1.0 + exp(-θ_val))
        log_jac = log(p) + log(1.0 - p)
        expected = (
            0.0 + log_jac
            + _stan_binomial(k_val, n_val, p)
        )
        @test LogDensityProblems.logdensity(problem, [θ_val]) ≈ expected atol=1e-6
    end
end

@testset "logdensity: unconstrained vector" begin
    obs_val = [0.5, -0.3, 1.2]
    n = length(obs_val)
    problem = instantiate(stan_model(@slic (;obs=obs_val) begin
        mu ~ std_normal(;n=length(obs))
        obs ~ normal(mu, 1.0)
    end))
    @test LogDensityProblems.dimension(problem) == n
    mu_val = [0.1, -0.5, 0.8]
    expected = (
        sum(_stan_std_normal.(mu_val))
        + sum(_stan_normal.(obs_val, mu_val, 1.0))
    )
    @test LogDensityProblems.logdensity(problem, mu_val) ≈ expected atol=1e-6
end

@testset "logdensity: linear regression, flat prior" begin
    x_val = [1.0, 2.0, 3.0]
    y_val = [2.1, 3.9, 6.2]
    problem = instantiate(stan_model(@slic (;y=y_val, x=x_val) begin
        alpha ~ flat()
        beta  ~ flat()
        y ~ normal(alpha + beta * to_vector(x), 1.0)
    end))
    @test LogDensityProblems.dimension(problem) == 2
    for (alpha_val, beta_val) in [(0.0, 2.0), (1.0, 1.5), (-0.5, 2.1)]
        predicted = alpha_val .+ beta_val .* x_val
        expected  = sum(_stan_normal.(y_val, predicted, 1.0))
        @test LogDensityProblems.logdensity(problem, [alpha_val, beta_val]) ≈ expected atol=1e-6
    end
end

@testset "logdensity: hierarchical (normal-normal)" begin
    n = 3
    y_val = [0.2, -0.1, 0.8]
    problem = instantiate(stan_model(@slic (;y=y_val) begin
        mu    ~ std_normal()
        sigma ~ std_normal(;lower=0.)
        theta ~ normal(mu, sigma; n=length(y))
        y     ~ normal(theta, 1.)
    end))
    @test LogDensityProblems.dimension(problem) == n + 2
    mu_unc    = 0.3
    log_sigma = 0.1
    theta_unc = [0.5, -0.2, 0.6]
    θ = [mu_unc; log_sigma; theta_unc]
    sigma_val = exp(log_sigma)
    expected = (
        _stan_std_normal(mu_unc)
        + _stan_std_normal(sigma_val) + log_sigma
        + sum(_stan_normal.(theta_unc, mu_unc, sigma_val))
        + sum(_stan_normal.(y_val, theta_unc, 1.0))
    )
    @test LogDensityProblems.logdensity(problem, θ) ≈ expected atol=1e-6
end

@testset "logdensity: dimension from data" begin
    for n in [1, 5, 20]
        obs = randn(n)
        problem = instantiate(stan_model(@slic (;obs=obs) begin
            mu ~ std_normal(;n=length(obs))
            obs ~ normal(mu, 1.0)
        end))
        @test LogDensityProblems.dimension(problem) == n
    end
end

# === builtin_shapes.jl tests ===

@testset "shapes: vector creation" begin
    @testset "rep_vector(v, n) :: vector[n]" begin
        model = @slic (;n=5, obs=randn(5)) begin
            v   = rep_vector(0., n)
            obs ~ normal(v, 1.)
        end
        @test check_type(model, :v, "vector")
        @test run_model(model)
    end
    @testset "linspaced_vector(n, lo, hi) :: vector[n]" begin
        model = @slic (;n=5, obs=randn(5)) begin
            v   = linspaced_vector(n, 0., 1.)
            obs ~ normal(v, 1.)
        end
        @test check_type(model, :v, "vector")
        @test run_model(model)
    end
end

@testset "shapes: matrix creation" begin
    @testset "rep_matrix(x::real, m, n) :: matrix[m,n]" begin
        model = @slic (;m=3, n=4, obs=randn(3)) begin
            A   = rep_matrix(0., m, n)
            obs ~ normal(A * rep_vector(0., n), 1.)
        end
        @test check_type(model, :A, "matrix")
        @test run_model(model)
    end
    @testset "rep_matrix(v::vector[m], n) :: matrix[m,n]" begin
        model = @slic (;m=3, n=4, obs=randn(3)) begin
            v   = rep_vector(1., m)
            A   = rep_matrix(v, n)
            obs ~ normal(A * rep_vector(0., n), 1.)
        end
        @test check_type(model, :A, "matrix")
        @test run_model(model)
    end
    @testset "diag_matrix(v::vector[n]) :: matrix[n,n]" begin
        model = @slic (;n=4, obs=randn(4)) begin
            v   = rep_vector(2., n)
            D   = diag_matrix(v)
            obs ~ normal(D * rep_vector(1., n), 1.)
        end
        @test check_type(model, :D, "matrix")
        @test run_model(model)
    end
    @testset "to_matrix(v, m, n) :: matrix[m,n]" begin
        model = @slic (;m=2, n=3, obs=randn(2)) begin
            A   = to_matrix(rep_vector(0., m*n), m, n)
            obs ~ normal(A * rep_vector(0., n), 1.)
        end
        @test check_type(model, :A, "matrix")
        @test run_model(model)
    end
end

@testset "shapes: conversions" begin
    @testset "to_vector(x) :: vector[n]" begin
        model = @slic (;x=randn(5), obs=randn(5)) begin
            v   = to_vector(x)
            obs ~ normal(v, 1.)
        end
        @test check_type(model, :v, "vector")
        @test run_model(model)
    end
    @testset "to_row_vector(x) :: row_vector[n]" begin
        model = @slic (;x=randn(5)) begin rv = to_row_vector(x) end
        @test transpiles(model)
        @test check_type(model, :rv, "row_vector")
    end
end

@testset "shapes: append functions" begin
    @testset "append_col(v::vector[n], w::vector[n]) :: matrix[n,2]" begin
        model = @slic (;n=4, obs=randn(4)) begin
            v   = rep_vector(0., n)
            w   = rep_vector(1., n)
            A   = append_col(v, w)
            obs ~ normal(A * rep_vector(0., 2), 1.)
        end
        @test check_type(model, :A, "matrix")
        @test run_model(model)
    end
    @testset "append_col(A::matrix[m,n1], B::matrix[m,n2]) :: matrix[m,n1+n2]" begin
        model = @slic (;m=3, n1=2, n2=3, obs=randn(3)) begin
            A1  = rep_matrix(0., m, n1)
            A2  = rep_matrix(0., m, n2)
            C   = append_col(A1, A2)
            obs ~ normal(C * rep_vector(0., n1+n2), 1.)
        end
        @test run_model(model)
    end
    @testset "append_row(v::vector[m], x::real) :: vector[m+1]" begin
        model = @slic (;n=4, obs=randn(5)) begin
            v   = rep_vector(0., n)
            v2  = append_row(v, 1.)
            obs ~ normal(v2, 1.)
        end
        @test check_type(model, :v2, "vector")
        @test run_model(model)
    end
    @testset "append_row(v1::vector[m], v2::vector[n]) :: vector[m+n]" begin
        model = @slic (;m=3, n=2, obs=randn(5)) begin
            v1  = rep_vector(0., m)
            v2  = rep_vector(1., n)
            v3  = append_row(v1, v2)
            obs ~ normal(v3, 1.)
        end
        @test check_type(model, :v3, "vector")
        @test run_model(model)
    end
end

@testset "shapes: linear algebra" begin
    @testset "dot_product(v, w) :: real" begin
        model = @slic (;n=4, obs=0.) begin
            v   = rep_vector(1., n)
            obs ~ normal(dot_product(v, v), 1.)
        end
        @test run_model(model)
    end
    @testset "rows_dot_product(A, B) :: vector[m]" begin
        let model = @slic (;m=3, n=4) begin
                A = rep_matrix(1., m, n)
                d = rows_dot_product(A, A)
            end
        @test check_type(model, :d, "vector")
        end
        @test run_model(@slic (;m=3, n=4, obs=randn(3)) begin
            A   = rep_matrix(1., m, n)
            obs ~ normal(rows_dot_product(A, A), 1.)
        end)
    end
    @testset "cumulative_sum(v) :: vector[n]" begin
        let model = @slic (;n=5) begin v = rep_vector(1., n); cs = cumulative_sum(v) end
        @test check_type(model, :cs, "vector")
        end
        @test run_model(@slic (;n=5, obs=randn(5)) begin
            v   = rep_vector(1., n)
            obs ~ normal(cumulative_sum(v), 1.)
        end)
    end
    @testset "mdivide_left_tri_low(L, b) :: vector[n]" begin
        model = @slic (;n=3, obs=randn(3)) begin
            L   = diag_matrix(rep_vector(1., n))
            b   = rep_vector(1., n)
            obs ~ normal(mdivide_left_tri_low(L, b), 1.)
        end
        let m2 = @slic (;n=3) begin L = diag_matrix(rep_vector(1., n)); x = mdivide_left_tri_low(L, rep_vector(1., n)) end
        @test check_type(m2, :x, "vector")
        end
        @test run_model(model)
    end
    @testset "diag_pre/post_multiply" begin
        @test run_model(@slic (;m=3, n=4, obs=randn(3)) begin
            v   = rep_vector(1., m)
            A   = rep_matrix(1., m, n)
            obs ~ normal(diag_pre_multiply(v, A) * rep_vector(0., n), 1.)
        end)
        @test run_model(@slic (;m=3, n=4, obs=randn(3)) begin
            v   = rep_vector(1., n)
            A   = rep_matrix(1., m, n)
            obs ~ normal(diag_post_multiply(A, v) * rep_vector(0., n), 1.)
        end)
    end
end

@testset "shapes: introspection" begin
    @testset "dims(v) :: int[1]" begin
        model = @slic (;n=4, obs=randn(4)) begin
            v   = rep_vector(0., n)
            w   = rep_vector(1., dims(v)[1])
            obs ~ normal(w, 1.)
        end
        @test run_model(model)
    end
    @testset "rows / cols" begin
        @test run_model(@slic (;m=3, n=4, obs=0.) begin
            A   = rep_matrix(0., m, n)
            obs ~ normal(rows(A) + cols(A), 1.)
        end)
    end
end

@testset "shapes: scalar math" begin
    @testset "inv_logit / logit" begin
        @test run_model(@slic (;obs=0.) begin
            x   = inv_logit(0.)
            obs ~ normal(x, 1.)
        end)
        @test run_model(@slic (;obs=0.) begin
            x   = logit(0.5)
            obs ~ normal(x, 1.)
        end)
    end
    @testset "log1m" begin
        @test run_model(@slic (;obs=0.) begin
            x   = log1m(0.5)
            obs ~ normal(x, 1.)
        end)
    end
    @testset "log_inv_logit" begin
        @test run_model(@slic (;obs=0.) begin
            x   = log_inv_logit(0.)
            obs ~ normal(x, 1.)
        end)
    end
    @testset "log1m_exp" begin
        @test run_model(@slic (;obs=0.) begin
            x   = log1m_exp(-1.)
            obs ~ normal(x, 1.)
        end)
    end
end

@testset "shapes: arrays" begin
    @testset "rep_array(x::real, n) :: real[n]" begin
        let model = @slic (;n=5) begin a = rep_array(1., n) end
        @test transpiles(model)
        @test check_type(model, :a, "array")
        end
    end
    @testset "to_array_1d" begin
        @test transpiles(@slic (;x=randn(3)) begin
            a = to_array_1d(to_vector(x))
        end)
    end
    @testset "append_array(a, b) :: real[m+n]" begin
        @test transpiles(@slic (;n=3, m=2) begin
            a = rep_array(1., n)
            b = rep_array(2., m)
            c = append_array(a, b)
        end)
    end
end

@testset "shapes: RNG" begin
    @testset "vector_std_normal_rng(n) :: vector[n]" begin
        @test compiles(@slic (;n=5, obs=randn(5)) begin
            mu     ~ std_normal(;n=n)
            obs    ~ normal(mu, 1.)
            y_rep  = vector_std_normal_rng(n)
        end)
    end
end

@testset "shapes: distributions" begin
    @testset "normal" begin
        @test run_model(@slic (;obs=1.5) begin
            mu    ~ std_normal()
            sigma ~ std_normal(;lower=0.)
            obs   ~ normal(mu, sigma)
        end)
        @test run_model(@slic (;obs=randn(10)) begin
            mu    ~ std_normal()
            sigma ~ std_normal(;lower=0.)
            obs   ~ normal(mu, sigma)
        end)
    end
    @testset "student_t" begin
        @test run_model(@slic (;obs=0.) begin
            mu  ~ std_normal()
            obs ~ student_t(3., mu, 1.)
        end)
    end
    @testset "cauchy" begin
        @test run_model(@slic (;obs=0.) begin
            mu  ~ std_normal()
            obs ~ cauchy(mu, 1.)
        end)
    end
    @testset "lognormal" begin
        @test run_model(@slic (;obs=1.) begin
            mu    ~ std_normal()
            sigma ~ std_normal(;lower=0.)
            obs   ~ lognormal(mu, sigma)
        end)
    end
    @testset "gamma" begin
        @test run_model(@slic (;obs=1.) begin
            alpha ~ std_normal(;lower=0.)
            beta  ~ std_normal(;lower=0.)
            obs   ~ gamma(alpha, beta)
        end)
    end
    @testset "beta" begin
        @test run_model(@slic (;obs=0.5) begin
            alpha  ~ std_normal(;lower=0.)
            beta_p ~ std_normal(;lower=0.)
            obs    ~ beta(alpha, beta_p)
        end)
    end
    @testset "exponential" begin
        @test run_model(@slic (;obs=1.) begin
            lambda ~ std_normal(;lower=0.)
            obs    ~ exponential(lambda)
        end)
    end
    @testset "uniform" begin
        @test run_model(@slic (;obs=0.5) begin
            obs ~ uniform(0., 1.)
        end)
    end
    @testset "bernoulli / bernoulli_logit" begin
        @test run_model(@slic (;obs=1) begin
            theta ~ beta(1., 1.)
            obs   ~ bernoulli(theta)
        end)
        @test run_model(@slic (;obs=1) begin
            logit_theta ~ std_normal()
            obs         ~ bernoulli_logit(logit_theta)
        end)
    end
    @testset "binomial" begin
        @test run_model(@slic (;k=3, n=10) begin
            theta ~ beta(1., 1.)
            k     ~ binomial(n, theta)
        end)
    end
    @testset "neg_binomial_2" begin
        @test run_model(@slic (;obs=5) begin
            mu  ~ std_normal(;lower=0.)
            phi ~ std_normal(;lower=0.)
            obs ~ neg_binomial_2(mu, phi)
        end)
    end
    @testset "multi_normal" begin
        @test run_model(@slic (;obs=randn(3)) begin
            mu  ~ std_normal(;n=3)
            cov = diag_matrix(rep_vector(1., 3))
            obs ~ multi_normal(mu, cov)
        end)
    end
end

# === posteriordb.jl tests ===

# Generate per-model test functions for PosteriorDB
let pdb = PosteriorDB.database()
    for posterior_name in PosteriorDB.posterior_names(pdb)
        post = StanBlocks.slic_implementation(PosteriorDB.posterior(pdb, posterior_name))
        isnothing(post) && continue
        fn_name = Symbol("test_pdb_", replace(posterior_name, "-" => "_", "." => "_"))
        @eval function $(fn_name)()
            @testset $posterior_name begin
                post = StanBlocks.slic_implementation(PosteriorDB.posterior(PosteriorDB.database(), $posterior_name))
                @test compiles(post)
            end
        end
    end
end
