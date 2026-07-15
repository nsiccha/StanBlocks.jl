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

# `@lpxf` opts a `_lpdf`-named @deffun into the base-callable + lpxf/rng/likelihood
# triad registration: it binds the base fn (`simple`, `vararg`, `fof`, `srs2`) and
# wires `lpxf_expr`/`rng_expr`/`likelihood_expr` so the name is usable both as a
# `y ~ simple(...)` distribution AND as a `::typeof(simple)` dispatch tag below.
# This registration was NAME-driven until 31772b4 (2026-04-24, "Add @lhs opt-in")
# made it an explicit `@lpxf`/`@lhs` opt-in; a plain `@deffun simple_lpdf` no longer
# binds `simple`, so the `::typeof(simple)` methods here threw `UndefVarError: simple`
# and blocked the whole include. Annotate only the heads actually used as base
# distributions; `my_lpdf` stays plain (`my` is only ever called, never `~ my(...)`).
@deffun begin
    @lpxf simple_lpdf(y, x) = 0.
    simple_lpdfs(y, x) = 0.
    simple_rng(x) = 0.
    @lpxf vararg_lpdf(y, args...) = 0.
    vararg_lpdfs(y, args...) = 0.
    vararg_rng(args...) = 0.

    my_lpdf(y, fargs...) = reject(1)
    my_lpdfs(args...) = reject(1)
    my_rng(args...) = reject(1)
    my_lpdf(y, ::typeof(simple), args...) = simple_lpdf(y, args...)
    my_lpdfs(y, ::typeof(simple), args...) = simple_lpdfs(y, args...)
    my_rng(::typeof(simple), args...) = simple_rng(args...)
    my_lpdf(y, ::typeof(vararg), args...) = vararg_lpdf(y, args...)

    @lpxf fof_lpdf(y, f, args...) = my_lpdf(y, f, args...)
    fof_lpdfs(y, f, args...) = my_lpdfs(y, f, args...)
    fof_rng(f, args...) = my_rng(f, args...)
    @lpxf srs2_lpdf(y, f, args...) = simple_reduce_sum(srs2_helper, rep_array(y, 1), f, args...)
    srs2_helper(y, f, args...) = my_lpdf(y, f, args...)
    srs2_lpdfs(y, f, args...) = 0.
    srs2_rng(f, args...) = 0.
end

# --- issue @deffun definitions (hoisted to module level) ---

@deffun begin
    issue10a(::vector[n]) = 0.
    issue10b(_::vector[n]) = 0.
end

# Determinism regression: an inline UDF whose locals are renamed `<name>__il_<id>`.
@deffun @inline det_polish(x::vector[n])::vector[n] = begin
    z = x * 2
    return z + 1
end

# Determinism regression model — exercises BOTH a session-counter-dependent
# Stan-output site: an inlined UDF (`__il_<id>` local rename) AND a lifted
# closure (`// lifted closure (id <id>)` + `closure_<id>` fn name). Transpiled
# twice in one session it must be byte-identical (per-trace counters, not
# session-global ones).
det_model = @slic (;n=5, ts=collect(1.0:5.0)) begin
    lambda ~ std_normal(;lower=0.)
    mu     ~ std_normal(;n=n)
    obs    = det_polish(mu)
    y      = ode_rk45((t, y_state) -> -lambda * y_state, [1.0], 0.0, to_array_1d(ts))
end

# Built-in math constants (`pi`, `ℯ`) resolve to their Float64 value in a model
# body (user decision 3bbtrv); arbitrary module-level numbers must NOT.
@deffun @inline _scale_by_consts(x::vector[n])::vector[n] = (4. / pi) * x .+ ℯ
consts_model = @slic (;n=5) begin
    x ~ std_normal(;n)
    s = _scale_by_consts(x)
end
CONSTS_TEST_NUM = 2.5
consts_bad_model = @slic (;n=5) begin
    x ~ std_normal(;n)
    s = CONSTS_TEST_NUM .* x
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

@testset "determinism: inline UDF + lifted closure" begin
    a = stan_code(stan_model(det_model))
    b = stan_code(stan_model(det_model))
    # non-vacuous: the model must actually exercise both counter-dependent sites
    @test occursin("__il_", a)
    @test occursin("// lifted closure", a)
    # transpiling twice in one session must be byte-identical
    @test a == b
end

# An in-body `@doc` lowers to a DocumentExpr; `forward!` wraps the docstring
# String into a `StanExpr{String}`, so render must dispatch
# `commentstring(::StanExpr)` (not just `::String`).
doc_model = @slic (;n=5) begin
    y ~ std_normal(;n)
    @doc "documented local declaration" z = y .* 2
end
@testset "in-body @doc docstring renders (StanExpr unwrap)" begin
    @test transpiles(doc_model)
    @test occursin("// documented local declaration", stan_code(stan_model(doc_model)))
end

@testset "built-in constants resolve (pi, ℯ); arbitrary const errors" begin
    @test transpiles(consts_model)
    code = stan_code(stan_model(consts_model))
    @test occursin("3.14159", code)   # pi
    @test occursin("2.71828", code)   # ℯ
    # arbitrary module-level number is NOT a built-in constant → loud failure
    @test !transpiles(consts_bad_model; re=false)
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

# === partly-missing-vector imputation tests ===

@testset "partly-missing: basic scalar-dist transpiles" begin
    @test transpiles(@slic (;y=[1.0, missing, 3.0, missing, 5.0]) begin
        mu    ~ normal(0., 10.)
        sigma ~ gamma(2., 1.)
        y     ~ normal(mu, sigma)
    end)
end

@testset "partly-missing: basic scalar-dist compiles" begin
    @test compiles(@slic (;y=[1.0, missing, 3.0, missing, 5.0]) begin
        mu    ~ normal(0., 10.)
        sigma ~ gamma(2., 1.)
        y     ~ normal(mu, sigma)
    end)
end

@testset "partly-missing: logdensity equals obs-only model" begin
    # y_mis goes to generated_quantities (GQ rng draw), not sampler params.
    # So the sampler dimension is 2: mu (unconstrained) + log_sigma (lower=0).
    # Rigorous check: logdensity must equal an explicit obs-only model at the
    # same unconstrained params — proves missing entries are excluded correctly.
    m_miss = @slic (;y=[1., missing, 3., missing, 5.]) begin
        mu    ~ normal(0., 10.)
        sigma ~ gamma(2., 1.)
        y     ~ normal(mu, sigma)
    end
    m_obs = @slic (;y=[1., 3., 5.]) begin
        mu    ~ normal(0., 10.)
        sigma ~ gamma(2., 1.)
        y     ~ normal(mu, sigma)
    end
    p_miss = instantiate(stan_model(m_miss))
    p_obs  = instantiate(stan_model(m_obs))
    @test LogDensityProblems.dimension(p_miss) == 2
    @test LogDensityProblems.dimension(p_obs)  == 2
    for v in [[-1.0, 0.5], [0.0, 0.0], [1.5, -0.3]]
        @test LogDensityProblems.logdensity(p_miss, v) ≈
              LogDensityProblems.logdensity(p_obs,  v) atol=1e-6
    end
    # y_mis appears in generated quantities (GQ rng draw), not parameters
    sc = stan_code(m_miss)
    @test occursin("generated quantities", sc) && occursin("y_mis", sc)
    @test occursin("merge_missing", sc)
end

@testset "partly-missing: vector dist arg (regression check)" begin
    # This exercises the getindex branch of maybe_index — previously broken
    # because the node was built with Symbol :getindex instead of Function.
    m = @slic (;x=collect(1.:6.), y=[1., missing, 3., missing, 5., missing]) begin
        a     ~ normal(0., 1.)
        b     ~ normal(0., 1.)
        sigma ~ gamma(2., 1.)
        y     ~ normal(a .+ b .* x, sigma)
    end
    @test compiles(m)
    @test LogDensityProblems.dimension(instantiate(stan_model(m))) == 3
    sc = stan_code(m)
    @test occursin("y_ii_obs", sc) && occursin("y_ii_mis", sc)
end

@testset "partly-missing: error on joint dist" begin
    @test_throws Exception stan_model(@slic (;y=[1.0, missing, 3.0]) begin
        mu  ~ std_normal(;n=3)
        cov = diag_matrix(rep_vector(1., 3))
        y   ~ multi_normal(mu, cov)
    end)
end

@testset "partly-missing: error on missing not used as LHS" begin
    @test_throws Exception stan_model(@slic (;y=[1.0, missing, 3.0]) begin
        mu ~ normal(0., 10.)
    end)
end

@testset "partly-missing: regression — all-observed vector unaffected" begin
    @test compiles(@slic (;y=[1.0, 2.0, 3.0]) begin
        mu    ~ normal(0., 10.)
        sigma ~ gamma(2., 1.)
        y     ~ normal(mu, sigma)
    end)
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

@testset "slic: scalar-array elementwise broadcasting (jbroadcasted)" begin
    # Generalised trace-level `jbroadcasted` (f4b601f): elementwise arithmetic on
    # scalar arrays (`array[] int` / `array[] real` — what a `Vector{Int}` /
    # `Vector{Float64}` data input becomes; NOT a native `vector`) has no Stan
    # operator, so it lowers to an element loop whose output CONTAINER is inferred
    # from `f`'s per-element return type — `int` stays `array[] int` (usable as an
    # index), `real` → `vector[n]`. Native `vector` operands keep Stan's built-in
    # scalar-vector ops (no lowering). GATE ON `compiles`, NOT `transpiles`: the
    # pre-generalisation bug transpiled PASS but stanc REJECTed (StanBlocks primer,
    # `4b45a5ac` scalar-array section).
    @testset "int array stays int (reporter: fs = fe - Ks + 1)" begin
        model = @slic (;fe=[3,4,5], Ks=[1,1,1]) begin
            fs = fe - Ks + 1
            mu ~ std_normal()
            mu
        end
        code = stan_code(model)
        # int container preserved — NOT silently coerced to a real `vector`
        @test occursin(r"array\[\w*\] int (?:\w+ )?fs\b", code)
        @test !occursin(r"vector\[\w*\] (?:\w+ )?fs\b", code)
        @test occursin("jbroadcasted_add", code)
        @test occursin("jbroadcasted_sub", code)
        @test compiles(model)
    end
    @testset "real scalar-array → vector" begin
        model = @slic (;n=4) begin
            ra = rep_array(1.5, n)   # array[n] real
            z  = 2.0 .- ra           # scalar-array real → jbroadcasted → vector
            mu ~ std_normal()
            mu
        end
        @test occursin("jbroadcasted_sub", stan_code(model))
        @test check_type(model, :z, "vector")
        @test compiles(model)
    end
    @testset "scalar-first ./ and .^ on a real array lower (any arg position)" begin
        # Previously fell through to the loud reject (no commute/negate identity);
        # the generalised any-position jbroadcasted lowers them directly.
        model = @slic (;n=4) begin
            ra = rep_array(2.0, n)
            q  = 3.0 ./ ra
            p  = 3.0 .^ ra
            mu ~ std_normal()
            mu
        end
        @test occursin("jbroadcasted", stan_code(model))
        @test compiles(model)
    end
    @testset "native vector operand keeps built-in scalar-vector op (no lowering)" begin
        model = @slic (;w=[1.0,2.0,3.0]) begin   # Vector{Float64} → Stan vector
            z = 2.0 .- w
            mu ~ std_normal()
            mu
        end
        @test !occursin("jbroadcasted", stan_code(model))
        @test compiles(model)
    end
    @testset "regression: binomial_lpmfs caller keeps real vector[n]" begin
        # jbroadcasted backs every `*_lpmfs` distribution — the container inference
        # must keep the real-element path a `vector[n]`, not flip it to `array[] int`.
        model = @slic (;y=[0,1,1,0,1], N=[2,2,2,2,2]) begin
            p ~ std_normal(;lower=0., upper=1.)
            y ~ binomial(N, p)
            p
        end
        @test occursin(r"vector\[\w*\] (?:\w+ )?y_likelihood\b", stan_code(model))
        @test compiles(model)
    end
    @testset "reject floor: plain * / ^ on scalar arrays still loudly rejected" begin
        # Matmul/dim-shaped `*` on arrays is NOT elementwise — must stay rejected,
        # never silently miscompiled.
        @test_throws Exception stan_code(@slic (;a=[1,2,3], b=[1,2,3]) begin
            c = a * b
            mu ~ std_normal()
            mu
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

# --- regression: @deffun symbolic-size operand-type propagation through HOFs ---
# A symbolic-size UDF return type (here `gslice`'s `vector[max(0, ends-start+1)]`)
# threaded through a higher-order `size(f(...))` layer used to corrupt the size
# expression's operands: every anonymization level reused `_arg1`, `_arg2`, … so
# an inner `deanon_size` re-substituted an enclosing UDF param's placeholder with
# the wrong arg, aliasing a scalar index to a 2-D value and surfacing as
# `tracetype not defined for (anything - anything)` (and an infinite-loop hang on
# the direct path). Per-call placeholder namespacing (`_arg<tok>_<i>`) fixes it.
# Reproduces with ZERO `@inline` — it is a `@deffun` size-propagation defect.
@deffun begin
    hd_rstart(x, i) = if i == 1; 1; else 1 + x.ends[i-1] end
    hd_rend(x, i) = x.ends[i]
    hd_gslice(x::vector[_], start, ends)::vector[max(0, ends-start+1)] = if ends < start
        rep_vector(0., 0)
    else
        x[start:ends]
    end
    hd_rvec(x, i) = hd_gslice(x.mem, hd_rstart(x, i), hd_rend(x, i))
    # symbolic-size 2-D core (sizes from arg shapes) — the necessary ingredient
    hd_core_sym(a::vector[m], b::vector[n])::int[m, n] = rep_array(0, m, n)
    # concrete-size 2-D core — control: must stay clean (and did pre-fix)
    hd_core_concrete(a::vector[m], b::vector[n])::int[2, 3] = rep_array(0, 2, 3)
    hd_wrap_sym(subject, ut, dt) = to_array_1d(hd_core_sym(hd_rvec(ut, subject), hd_rvec(dt, subject)))
    hd_wrap_concrete(subject, ut, dt) = to_array_1d(hd_core_concrete(hd_rvec(ut, subject), hd_rvec(dt, subject)))
    # double-HOF: f threaded through `size(f(...))` (sub_lens -> sub_len)
    hd_sublen(f, xi, a, b) = size(f(xi, a, b))
    hd_sublens(f, x::anything[n], a, b) = begin
        rv::int[n]
        for i in 1:n
            rv[i] = hd_sublen(f, x[i], a, b)
        end
        rv
    end
end

@testset "regression: @deffun symbolic-size operand propagation through HOFs" begin
    D = (; us=[1, 2], ut=[[0., 1.], [0., 1., 2.]], dt=[[0.], [0., 5.]])
    # double-HOF + symbolic-size core: was `tracetype not defined for (anything - anything)`
    m_hof = @slic D begin
        r = hd_sublens(hd_wrap_sym, us, ut, dt)
        p ~ normal(1.0 * sum(r), 1.0)
    end
    @test transpiles(m_hof)
    # direct (0-HOF) symbolic-size call: was an infinite-loop hang
    m_direct = @slic D begin
        idx = hd_wrap_sym(1, ut, dt)
        p ~ normal(1.0 * sum(idx), 1.0)
    end
    @test transpiles(m_direct)
    # control: concrete-size core stays clean (no regression)
    m_concrete = @slic D begin
        r = hd_sublens(hd_wrap_concrete, us, ut, dt)
        p ~ normal(1.0 * sum(r), 1.0)
    end
    @test transpiles(m_concrete)
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
