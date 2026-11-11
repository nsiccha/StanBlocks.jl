@testitem "slic: algebraic solvers trace named and captured residuals" tags=[:slic, :regression, :stanc] setup=[StanBlocksImports, StanBlocksTestSetup] begin
    using .StanBlocksTestSetup: stanc_compiles
    @deffun residual(x::vector[n], theta::vector[n])::vector[n] = x .* x .- exp(theta)
    for solver in (:solve_newton, :solve_powell, :solve_newton_tol, :solve_powell_tol)
        controls = endswith(string(solver), "_tol") ? (1e-12, 1e-12, 100) : ()
        for f in (:residual, :((x, theta) -> x .* x .- exp(theta)),
                  :((x, theta) -> x .* x .- exp(theta) .- exp(extra)))
            m = StanBlocks.SlicModel(quote
                theta::vector[n] ~ normal(0.0, 1.0)
                extra ~ std_normal()
                root = $solver($f, rep_vector(1.0, n), $(controls...), theta)
                y ~ normal(sum(root) + extra, 1.0)
            end, (; n=2, y=0.3), @__MODULE__)
            code = stan_code(m)
            @test occursin("vector[n] root = $solver(", code)
            @test stanc_compiles(m)
        end
    end
    # No variadic inputs; the entire dependency is captured by the residual.
    captured = @slic (; y=0.3) begin
        a ~ std_normal()
        root = solve_powell(x -> x .* x .- exp(a), rep_vector(1.0, 1))
        y ~ normal(root[1], 1.0)
    end
    @test stanc_compiles(captured)
    @test stan_code(captured) == stan_code(captured)
end

@testitem "slic: nested and model-level solver closures share a trace context" tags=[:slic, :regression, :stanc] setup=[StanBlocksImports, StanBlocksTestSetup] begin
    using .StanBlocksTestSetup: stanc_compiles
    @deffun trajectory(a::real)::vector[1, 1] =
        ode_rk45((t, x) -> -a * x, rep_vector(1.0, 1), 0.0,
            to_array_1d(rep_vector(0.5, 1)))
    m = @slic (; ts=[0.5], y=0.3) begin
        a ~ normal(0.0, 1.0)
        nested = trajectory(a)
        direct = ode_rk45((t, x) -> a * x, rep_vector(1.0, 1), 0.0, to_array_1d(ts))
        y ~ normal(nested[1, 1] + direct[1, 1], 1.0)
    end
    @test stanc_compiles(m)
    @test stan_code(m) == stan_code(m)
end

@testitem "slic: implicit roots carry gradients through a parameter bound" tags=[:slic, :regression, :bridgestan] setup=[StanBlocksImports] begin
    @deffun begin
        residual(x::vector[n], a::real, b::real)::vector[n] =
            x .* x .- (exp(a) + exp(b))
        positive_root(a::real, b::real)::real =
            solve_newton_tol(x -> x .* x .- (exp(a) + exp(b)),
                rep_vector(1.0, 1), 1e-12, 1e-12, 100)[1]
    end
    numerical = @slic (; y=0.7) begin
        a ~ std_normal()
        b ~ std_normal()
        z ~ uniform(0.0, positive_root(a, b))
        r_newton = solve_newton(residual, rep_vector(1.0, 1), a, b)[1]
        r_powell = solve_powell(residual, rep_vector(1.0, 1), a, b)[1]
        r_newton_tol = positive_root(a, b)
        r_powell_tol = solve_powell_tol(x -> x .* x .- (exp(a) + exp(b)),
            rep_vector(1.0, 1), 1e-12, 1e-12, 100)[1]
        y ~ normal(z + r_newton + r_powell + r_newton_tol + r_powell_tol, 1.0)
    end
    exact = @slic (; y=0.7) begin
        a ~ std_normal()
        b ~ std_normal()
        z ~ uniform(0.0, sqrt(exp(a) + exp(b)))
        y ~ normal(z + 4 * sqrt(exp(a) + exp(b)), 1.0)
    end
    p = instantiate(numerical)
    reference = instantiate(exact)
    @test BridgeStan.param_unc_names(p.model) == BridgeStan.param_unc_names(reference.model)
    names = BridgeStan.param_names(p.model; include_tp=true, include_gq=false)
    root_names = ["r_newton", "r_powell", "r_newton_tol", "r_powell_tol"]
    indices = [only(findall(==(name), names)) for name in root_names]
    for q in ([0.2, -0.4, 0.3], [-0.7, 0.5, -0.8], [0.0, 0.0, 1.2])
        lp, g = BridgeStan.log_density_gradient(p.model, q; propto=false, jacobian=true)
        exact_lp, exact_g = BridgeStan.log_density_gradient(reference.model, q; propto=false, jacobian=true)
        @test lp ≈ exact_lp atol=2e-6 rtol=2e-7
        @test g ≈ exact_g atol=2e-6 rtol=2e-7
        constrained = BridgeStan.param_constrain(p.model, q; include_tp=true, include_gq=false)
        root = sqrt(exp(q[1]) + exp(q[2]))
        @test constrained[indices] ≈ fill(root, 4) atol=2e-7 rtol=2e-7
        @test 0 < constrained[3] < root
        # An independent finite difference exercises the bound's Jacobian too.
        h = 1e-5
        fd = map(eachindex(q)) do i
            plus, minus = copy(q), copy(q)
            plus[i] += h
            minus[i] -= h
            (BridgeStan.log_density(p.model, plus; propto=false, jacobian=true) -
             BridgeStan.log_density(p.model, minus; propto=false, jacobian=true)) / (2h)
        end
        @test g ≈ fd atol=2e-5 rtol=2e-6
    end
end
