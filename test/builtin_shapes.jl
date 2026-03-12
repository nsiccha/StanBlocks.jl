import StanBlocks.stan: stan_model, stan_code, instantiate, compiles, transpiles
using LogDensityProblems

# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

"""
    run_model(model) -> Bool

Compile and run `model`, returning `true` iff no size errors occur.

Two levels of size checking happen:
1. **Instantiation** (`StanProblem` construction) runs the `transformed_data`
   block with the bound data.  Size mismatches there are caught immediately.
2. **Log-density evaluation** runs the `model` block.  Size mismatches between
   data variables and computed results (e.g. `obs ~ normal(v, 1.)` when obs
   and v have different lengths) are caught here.

We run both so that size errors in either block are detected.
"""
function run_model(model)
    problem = try
        instantiate(model)
    catch
        return false
    end
    θ = zeros(LogDensityProblems.dimension(problem))
    ld = LogDensityProblems.logdensity(problem, θ)
    return isfinite(ld)
end

"""
    check_type(model, var, expected_sigtype)

Check that `var` is declared with the expected sigtype keyword in the generated
Stan code (e.g. `"vector"`, `"matrix"`, `"row_vector"`, `"array"`).
Uses the Stan code string rather than internal type API for robustness.
Handles both  `vector[n] v`  and  `array[n] real v` forms.
"""
function check_type(model, var::Symbol, expected_sigtype::AbstractString)
    code = stan_code(model)
    pat = Regex("\\b$(expected_sigtype)(?:\\[[^\\]]*\\])+ (?:\\w+ )?$var\\b")
    occursin(pat, code)
end

# ──────────────────────────────────────────────────────────────────────────────
# Tests
# ──────────────────────────────────────────────────────────────────────────────

@testset "builtin shapes" begin

    # ─────────────────────────────────────────────────────────────────────────
    # Vector creation
    # ─────────────────────────────────────────────────────────────────────────
    @testset "rep_vector(v, n) :: vector[n]" begin
        # obs has exactly n=5 elements; if rep_vector returned vector[k≠n] the
        # normal statement would fail at runtime.
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

    # ─────────────────────────────────────────────────────────────────────────
    # Matrix creation
    # ─────────────────────────────────────────────────────────────────────────
    @testset "rep_matrix(x::real, m, n) :: matrix[m,n]" begin
        # A*v produces vector[m]; obs has m=3 elements.
        # If A were matrix[m,k≠n] then A*rep_vector(0.,n) would fail at runtime.
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
        # D*v is vector[n]; obs has n=4 elements.
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

    # ─────────────────────────────────────────────────────────────────────────
    # Shape conversions
    # ─────────────────────────────────────────────────────────────────────────
    @testset "to_vector(x) :: vector[n]" begin
        # x has n=5 elements; to_vector should preserve length.
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

    # ─────────────────────────────────────────────────────────────────────────
    # Append functions  (size is the critical thing to test here)
    # ─────────────────────────────────────────────────────────────────────────
    @testset "append_col(v::vector[n], w::vector[n]) :: matrix[n,2]" begin
        # A has 2 columns; A * [a,b] is vector[n]; obs has n=4 elements.
        # If A were matrix[n,1], A*rep_vector(0.,2) would fail at runtime.
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
        # C has n1+n2 columns; C * rep_vector(0., n1+n2) is vector[m].
        # obs has m=3 elements; wrong column count → runtime failure.
        model = @slic (;m=3, n1=2, n2=3, obs=randn(3)) begin
            A1  = rep_matrix(0., m, n1)
            A2  = rep_matrix(0., m, n2)
            C   = append_col(A1, A2)
            obs ~ normal(C * rep_vector(0., n1+n2), 1.)
        end
        @test run_model(model)
    end

    @testset "append_row(v::vector[m], x::real) :: vector[m+1]" begin
        # v2 should have m+1=5 elements; obs has 5 elements.
        # If v2 had m=4 elements the normal statement would fail at runtime.
        model = @slic (;n=4, obs=randn(5)) begin
            v   = rep_vector(0., n)
            v2  = append_row(v, 1.)
            obs ~ normal(v2, 1.)
        end
        @test check_type(model, :v2, "vector")
        @test run_model(model)
    end

    @testset "append_row(v1::vector[m], v2::vector[n]) :: vector[m+n]" begin
        # v3 should have m+n=5 elements; obs has 5 elements.
        model = @slic (;m=3, n=2, obs=randn(5)) begin
            v1  = rep_vector(0., m)
            v2  = rep_vector(1., n)
            v3  = append_row(v1, v2)
            obs ~ normal(v3, 1.)
        end
        @test check_type(model, :v3, "vector")
        @test run_model(model)
    end

    # ─────────────────────────────────────────────────────────────────────────
    # Linear algebra utilities
    # ─────────────────────────────────────────────────────────────────────────
    @testset "dot_product(v::vector[n], w::vector[n]) :: real" begin
        # Result is a scalar; obs is scalar.
        model = @slic (;n=4, obs=0.) begin
            v   = rep_vector(1., n)
            obs ~ normal(dot_product(v, v), 1.)
        end
        @test run_model(model)
    end

    @testset "rows_dot_product(A::matrix[m,n], B::matrix[m,n]) :: vector[m]" begin
        # Result is vector[m]; obs has m=3 elements.
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

    @testset "cumulative_sum(v::vector[n]) :: vector[n]" begin
        # Result should have same length as v (n=5); obs has 5 elements.
        let model = @slic (;n=5) begin v = rep_vector(1., n); cs = cumulative_sum(v) end
        @test check_type(model, :cs, "vector")
        end
        @test run_model(@slic (;n=5, obs=randn(5)) begin
            v   = rep_vector(1., n)
            obs ~ normal(cumulative_sum(v), 1.)
        end)
    end

    @testset "mdivide_left_tri_low(L::matrix[n,n], b::vector[n]) :: vector[n]" begin
        # Result is vector[n]; obs has n=3 elements.
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

    @testset "diag_pre_multiply / diag_post_multiply" begin
        # diag_pre_multiply(v::vector[m], M::matrix[m,n]) :: matrix[m,n]
        # Result * rep_vector(0.,n) is vector[m]; obs has m=3 elements.
        @test run_model(@slic (;m=3, n=4, obs=randn(3)) begin
            v   = rep_vector(1., m)
            A   = rep_matrix(1., m, n)
            obs ~ normal(diag_pre_multiply(v, A) * rep_vector(0., n), 1.)
        end)
        # diag_post_multiply(M::matrix[m,n], v::vector[n]) :: matrix[m,n]
        @test run_model(@slic (;m=3, n=4, obs=randn(3)) begin
            v   = rep_vector(1., n)
            A   = rep_matrix(1., m, n)
            obs ~ normal(diag_post_multiply(A, v) * rep_vector(0., n), 1.)
        end)
    end

    # ─────────────────────────────────────────────────────────────────────────
    # Shape introspection
    # ─────────────────────────────────────────────────────────────────────────
    @testset "dims(v::vector[n]) :: int[1] (value = [n])" begin
        # Use dims output as the size of another vector; sizes must agree.
        # obs has n=4 elements; w is built with dims(v)[1] which should be n=4.
        model = @slic (;n=4, obs=randn(4)) begin
            v   = rep_vector(0., n)
            w   = rep_vector(1., dims(v)[1])
            obs ~ normal(w, 1.)
        end
        @test run_model(model)
    end

    @testset "rows / cols return correct scalars" begin
        # rows(A)+cols(A) is a scalar; obs is scalar.
        @test run_model(@slic (;m=3, n=4, obs=0.) begin
            A   = rep_matrix(0., m, n)
            obs ~ normal(rows(A) + cols(A), 1.)
        end)
    end

    # ─────────────────────────────────────────────────────────────────────────
    # Scalar math functions  (run to confirm finite result)
    # ─────────────────────────────────────────────────────────────────────────
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

    # ─────────────────────────────────────────────────────────────────────────
    # Array utilities
    # ─────────────────────────────────────────────────────────────────────────
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

    @testset "append_array(a::real[m], b::real[n]) :: real[m+n]" begin
        @test transpiles(@slic (;n=3, m=2) begin
            a = rep_array(1., n)
            b = rep_array(2., m)
            c = append_array(a, b)
        end)
    end

    # ─────────────────────────────────────────────────────────────────────────
    # Random number generators (generated quantities)
    # ─────────────────────────────────────────────────────────────────────────
    @testset "vector_std_normal_rng(n) :: vector[n]" begin
        @test compiles(@slic (;n=5, obs=randn(5)) begin
            mu     ~ std_normal(;n=n)
            obs    ~ normal(mu, 1.)
            y_rep  = vector_std_normal_rng(n)
        end)
    end

    # ─────────────────────────────────────────────────────────────────────────
    # Distribution families  (compilation + finite log-density at zeros)
    # ─────────────────────────────────────────────────────────────────────────
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
