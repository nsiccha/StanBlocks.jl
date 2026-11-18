# Analytic Jacobian UDFs for Stan's ODE solvers (todo `0bkkkf2`): the
# `ode_*(f, jac, y0, …)` overloads and `ode_jacobian_error(f, jac, t, y, …)` of
# the `nsiccha/stanc3` / `nsiccha/math` `dde-solver` line, and `jacobian(f)`,
# which StanBlocks derives with Symbolics.jl (`StanBlocksSymbolicsExt`).
#
# Stock stanc rejects the overloads, so the stanc and BridgeStan items need
# that toolchain: `STANBLOCKS_ODE_JACOBIAN_BRIDGESTAN` names a BridgeStan tree
# whose `bin/stanc` and `stan/lib/stan_math` come from the `dde-solver` forks
# (see `test/symbolics/README.md`). Without it they are reported as skipped.

@testitem "slic: ODE solvers pass a hand-written Jacobian UDF to Stan" tags=[:slic] setup=[StanBlocksImports] begin
    @deffun begin
        hw_rhs(t::real, y::vector[n], k::real)::vector[n] = begin
            dy::vector[n]
            dy[1] = -k * y[1]
            dy[2] = k * y[1] - y[2]
            dy
        end
        hw_jac(t::real, y::vector[n], k::real)::matrix[2, 3] = begin
            J = rep_matrix(0.0, 2, 3)
            J[1, 1] = -k
            J[1, 3] = -y[1]
            J[2, 1] = k
            J[2, 2] = -1.0
            J[2, 3] = y[1]
            J
        end
    end
    m = @slic (; ts=[1.0, 2.0], obs=[0.1, 0.2]) begin
        k ~ lognormal(0.0, 1.0)
        y0 = rep_vector(1.0, 2)
        a = ode_rk45(hw_rhs, hw_jac, y0, 0.0, to_array_1d(ts), k)
        b = ode_bdf_tol(hw_rhs, hw_jac, y0, 0.0, to_array_1d(ts), 1e-8, 1e-8, 10000, k)
        err = ode_jacobian_error(hw_rhs, hw_jac, 0.5, y0, k)
        obs ~ normal(to_vector(a[:, 2]) + to_vector(b[:, 1]) + err, 1.0)
    end
    code = stan_code(m)
    @test occursin("ode_rk45(hw_rhs, hw_jac, y0, 0.0, to_array_1d(ts), k)", code)
    @test occursin("ode_bdf_tol(hw_rhs, hw_jac, y0, 0.0, to_array_1d(ts), 1.0e-8, 1.0e-8, 10000, k)", code)
    @test occursin("ode_jacobian_error(hw_rhs, hw_jac, 0.5, y0, k)", code)
    @test occursin("matrix hw_jac(", code)
    @test occursin("vector hw_rhs(", code)

    # The solution keeps its shape: times × states.
    @test occursin("array[ts_n] vector[2] a = ", code)

    # Without a Jacobian, the call is exactly as before.
    plain = @slic (; ts=[1.0, 2.0], obs=[0.1, 0.2]) begin
        k ~ lognormal(0.0, 1.0)
        a = ode_rk45(hw_rhs, rep_vector(1.0, 2), 0.0, to_array_1d(ts), k)
        obs ~ normal(to_vector(a[:, 2]), 1.0)
    end
    @test occursin("ode_rk45(hw_rhs, rep_vector(1.0, 2), 0.0, to_array_1d(ts), k)", stan_code(plain))

    # A closure `f` threads its captures after the caller's arguments, and Stan
    # passes `jac` the same arguments, so a closure Jacobian may capture exactly
    # what `f` captures, and nothing else.
    captured = @slic (; ts=[1.0, 2.0], obs=[0.1, 0.2]) begin
        k ~ lognormal(0.0, 1.0)
        f = (t, y) -> -k * y
        g = (t, y) -> append_col(rep_matrix(-k, 1, 1), rep_matrix(-y[1], 1, 1))
        a = ode_rk45(f, g, rep_vector(1.0, 1), 0.0, to_array_1d(ts))
        obs ~ normal(to_vector(a[:, 1]), 1.0)
    end
    ccode = stan_code(captured)
    @test occursin(r"ode_rk45\(closure_\d+, closure_\d+, rep_vector\(1\.0, 1\), 0\.0, to_array_1d\(ts\), k\)", ccode)
    other = @slic (; ts=[1.0, 2.0], obs=[0.1, 0.2]) begin
        k ~ lognormal(0.0, 1.0)
        c ~ lognormal(0.0, 1.0)
        f = (t, y) -> -k * y
        g = (t, y) -> rep_matrix(-c, 1, 2)
        a = ode_rk45(f, g, rep_vector(1.0, 1), 0.0, to_array_1d(ts))
        obs ~ normal(to_vector(a[:, 1]), 1.0)
    end
    @test_throws "cannot capture values `f` does not capture" stan_code(other)
end

@testitem "slic: jacobian(f) names the Symbolics.jl requirement when it is not loaded" tags=[:slic] setup=[StanBlocksImports] begin
    if isnothing(Base.get_extension(StanBlocks, :StanBlocksSymbolicsExt))
        @deffun nosym_rhs(t::real, y::vector[n], k::real)::vector[n] = -k * y
        m = @slic (; ts=[1.0, 2.0], obs=[0.1, 0.2]) begin
            k ~ lognormal(0.0, 1.0)
            a = ode_rk45(nosym_rhs, jacobian(nosym_rhs), rep_vector(1.0, 1), 0.0, to_array_1d(ts), k)
            obs ~ normal(to_vector(a[:, 1]), 1.0)
        end
        @test_throws "load it (`using Symbolics`)" stan_code(m)
    else
        @test_skip "Symbolics.jl is loaded in this process"
    end
end

@testsnippet OdeJacobianModels begin
    using Symbolics
    import BridgeStan

    # Right-hand sides covering what the derivation evaluates: indexed
    # assignments into a declared local, a loop over the states, a vector
    # argument, a nested `@deffun`, a condition on the time, and Stan builtins
    # that stay opaque (`inv_logit`, `log1p_exp`) or are expanded (`square`).
    @deffun begin
        jtest_three(t::real, y::vector[n], ka::real, ke::real, k12::real)::vector[n] = begin
            dy::vector[n]
            dy[1] = -ka * y[1]
            dy[2] = ka * y[1] - ke * y[2] - k12 * y[2]
            dy[3] = k12 * y[2] - exp(-t) * square(y[3])
            dy
        end
        jtest_rate(x::real, vmax::real, km::real)::real = vmax * x / (km + x)
        jtest_chain(t::real, y::vector[n], theta::vector[m], tlag::real)::vector[n] = begin
            dy::vector[n]
            input = t < tlag ? 0.0 : theta[1] * inv_logit(t - tlag)
            dy[1] = input - jtest_rate(y[1], theta[2], theta[3])
            for i in 2:n
                dy[i] = jtest_rate(y[i - 1], theta[2], theta[3]) - log1p_exp(theta[4]) * y[i]
            end
            dy
        end
        jtest_zero_jac(t::real, y::vector[n], theta::vector[m], tlag::real)::matrix[4, 9] =
            rep_matrix(0.0, 4, 9)
        # Matrix and vector arguments (Stan counts a matrix's entries
        # column-major), and vectorised builtins on the state.
        jtest_linear(t::real, y::vector[n], A::matrix[n, n], b::vector[n])::vector[n] =
            A * y + exp(-t) * b
        jtest_vectorised(t::real, y::vector[n], k::real, cap::real[n])::vector[n] =
            k * square(y) - fmax(y, 0.5) + to_vector(cap) .* inv_logit(y)
    end
end

@testitem "slic: jacobian(f) derives the Jacobian UDF of an ODE right-hand side" tags=[:slic, :symbolics] setup=[StanBlocksImports, OdeJacobianModels] begin
    m = @slic (; ts=[0.5, 1.0, 2.0], obs=[1.0, 0.8, 0.5]) begin
        ka ~ lognormal(0.0, 0.5)
        ke ~ lognormal(0.0, 0.5)
        k12 ~ lognormal(0.0, 0.5)
        y0 = rep_vector(1.0, 3)
        a = ode_rk45(jtest_three, jacobian(jtest_three), y0, 0.0, to_array_1d(ts), ka, ke, k12)
        obs ~ normal(to_vector(a[:, 2]), 0.5)
    end
    code = stan_code(m)
    @test occursin("ode_rk45(jtest_three, jtest_three_jacobian_3, y0, 0.0, to_array_1d(ts), ka, ke, k12)", code)
    @test occursin("matrix jtest_three_jacobian_3(", code)
    # 3 states and 3 real arguments: a 3×6 matrix, nonzero entries only.
    @test occursin("rep_matrix(0.0, 3, 6)", code)
    @test occursin("jac[1, 1] = (-ka);", code)
    @test occursin("jac[1, 4] = (-y[1]);", code)
    @test !occursin("jac[1, 2]", code)

    # A body that writes three entries cannot serve four states.
    two = @slic (; ts=[0.5, 1.0], obs=[1.0, 0.8]) begin
        ka ~ lognormal(0.0, 0.5)
        a = ode_bdf(jtest_three, jacobian(jtest_three), rep_vector(1.0, 3), 0.0, to_array_1d(ts), ka, ka, ka)
        b = ode_rk45(jtest_three, jacobian(jtest_three), rep_vector(1.0, 4), 0.0, to_array_1d(ts), ka, ka, ka)
        obs ~ normal(to_vector(a[:, 2]) + to_vector(b[:, 2]), 0.5)
    end
    @test_throws "leaves an entry of its returned vector unassigned" stan_code(two)

    # Loops, a vector argument, a nested `@deffun`, a time condition, opaque builtins.
    chain = @slic (; ts=[0.5, 1.0, 2.0], obs=[1.0, 0.8, 0.5]) begin
        theta::vector[4] ~ normal(0.0, 1.0)
        tlag ~ lognormal(0.0, 0.5)
        a = ode_rk45_tol(jtest_chain, jacobian(jtest_chain), rep_vector(0.0, 4), 0.0, to_array_1d(ts),
            1e-8, 1e-8, 100000, theta, tlag)
        obs ~ normal(to_vector(a[:, 4]), 0.5)
    end
    ccode = stan_code(chain)
    @test occursin(r"ode_rk45_tol\(\s*jtest_chain,\s*jtest_chain_jacobian_4_4,", ccode)
    # 4 states, then 4 + 1 real arguments: a 4×9 matrix.
    @test occursin("rep_matrix(0.0, 4, 9)", ccode)
    # `log1p_exp`'s derivative is Stan's own `inv_logit`; `t < tlag` stays a condition.
    @test occursin("inv_logit(theta[4])", ccode)
    @test occursin(r"\? 0 : inv_logit\(", ccode)

    # The same right-hand side at two state counts is two functions.
    sizes = @slic (; ts=[0.5, 1.0], obs=[1.0, 0.8]) begin
        theta::vector[4] ~ normal(0.0, 1.0)
        a = ode_rk45(jtest_chain, jacobian(jtest_chain), rep_vector(0.0, 3), 0.0, to_array_1d(ts), theta, 0.1)
        b = ode_rk45(jtest_chain, jacobian(jtest_chain), rep_vector(0.0, 4), 0.0, to_array_1d(ts), theta, 0.1)
        obs ~ normal(to_vector(a[:, 3]) + to_vector(b[:, 4]), 0.5)
    end
    scode = stan_code(sizes)
    @test occursin("matrix jtest_chain_jacobian_3_4(", scode)
    @test occursin("matrix jtest_chain_jacobian_4_4(", scode)

    # Matrix arguments are differentiated column by column, after the state.
    lin = @slic (; ts=[0.5, 1.0], obs=[1.0, 0.8]) begin
        A::matrix[2, 2] ~ normal(0.0, 0.3)
        b::vector[2] ~ normal(0.0, 1.0)
        a = ode_rk45(jtest_linear, jacobian(jtest_linear), rep_vector(1.0, 2), 0.0, to_array_1d(ts), A, b)
        obs ~ normal(to_vector(a[:, 2]), 0.5)
    end
    lcode = stan_code(lin)
    @test occursin("matrix jtest_linear_jacobian_2_2x2_2(", lcode)
    @test occursin("jac[1, 2] = A[1, 2];", lcode)
    @test occursin("jac[1, 5] = y[2];", lcode)   # d f[1] / d A[1, 2]
    @test occursin("jac[2, 4] = y[1];", lcode)   # d f[2] / d A[2, 1]
    # Vectorised builtins apply elementwise, as in Stan.
    vec = @slic (; ts=[0.5, 1.0], obs=[1.0, 0.8], cap=[1.0, 2.0, 3.0]) begin
        k ~ normal(0.0, 1.0)
        a = ode_rk45(jtest_vectorised, jacobian(jtest_vectorised), rep_vector(1.0, 3), 0.0, to_array_1d(ts), k, cap)
        obs ~ normal(to_vector(a[:, 2]), 0.5)
    end
    @test_throws "every size must be a literal" stan_code(vec)
    # A closure right-hand side: its captures are trailing arguments.
    clo = @slic (; ts=[0.5, 1.0], obs=[1.0, 0.8]) begin
        k ~ lognormal(0.0, 1.0)
        c ~ lognormal(0.0, 1.0)
        f = (t, y) -> [-k * y[1] + c * y[2], k * y[1] - c * y[2]]
        a = ode_bdf(f, jacobian(f), rep_vector(1.0, 2), 0.0, to_array_1d(ts))
        obs ~ normal(to_vector(a[:, 2]), 0.5)
    end
    kcode = stan_code(clo)
    @test occursin(r"ode_bdf\(closure_\d+, closure_\d+_jacobian_2, rep_vector\(1\.0, 2\), 0\.0, to_array_1d\(ts\), k, c\)", kcode)
    @test occursin("jac[1, 4] = y[2];", kcode)

    # Refusals name what the derivation needs.
    unsized = @slic (; ts=[0.5, 1.0], obs=[1.0, 0.8], y0=[1.0, 1.0, 1.0]) begin
        ka ~ lognormal(0.0, 0.5)
        a = ode_rk45(jtest_three, jacobian(jtest_three), y0, 0.0, to_array_1d(ts), ka, ka, ka)
        obs ~ normal(to_vector(a[:, 2]), 0.5)
    end
    @test_throws "every size must be a literal" stan_code(unsized)
    @deffun jtest_other(t::real, y::vector[n], k::real)::vector[n] = -k * y
    mismatched = @slic (; ts=[0.5, 1.0], obs=[1.0, 0.8]) begin
        k ~ lognormal(0.0, 0.5)
        a = ode_rk45(jtest_other, jacobian(jtest_three), rep_vector(1.0, 1), 0.0, to_array_1d(ts), k)
        obs ~ normal(to_vector(a[:, 1]), 0.5)
    end
    @test_throws "must name the solver's own right-hand side" stan_code(mismatched)
end

@testitem "slic: derived ODE Jacobians match autodiff in Stan" tags=[:slic, :symbolics, :bridgestan] setup=[StanBlocksImports, OdeJacobianModels] begin
    tree = get(ENV, "STANBLOCKS_ODE_JACOBIAN_BRIDGESTAN", "")
    if isempty(tree)
        @test_skip "Set STANBLOCKS_ODE_JACOBIAN_BRIDGESTAN to a BridgeStan tree built from the dde-solver forks"
    else
        withenv("BRIDGESTAN" => tree, "STANC_PATH" => joinpath(tree, "bin", "stanc")) do
            # The same model with and without `jac`, plus the pointwise check of
            # every derived Jacobian against Stan's autodiff Jacobian of `f`.
            source(jac) = quote
                theta::vector[4] ~ lognormal(0.0, 0.5)
                tlag ~ lognormal(-1.0, 0.5)
                ka ~ lognormal(0.0, 0.5)
                ke ~ lognormal(-1.0, 0.5)
                k = 0.3
                y0 = rep_vector(0.5, 4)
                chain = $(jac ? :(ode_rk45_tol(jtest_chain, jacobian(jtest_chain), y0, 0.0, to_array_1d(ts),
                    1e-10, 1e-10, 1000000, theta, tlag)) : :(ode_rk45_tol(jtest_chain, y0, 0.0, to_array_1d(ts),
                    1e-10, 1e-10, 1000000, theta, tlag)))
                three = $(jac ? :(ode_bdf_tol(jtest_three, jacobian(jtest_three), rep_vector(1.0, 3), 0.0,
                    to_array_1d(ts), 1e-10, 1e-10, 1000000, ka, ke, k)) : :(ode_bdf_tol(jtest_three,
                    rep_vector(1.0, 3), 0.0, to_array_1d(ts), 1e-10, 1e-10, 1000000, ka, ke, k)))
                obs ~ normal(to_vector(chain[:, 4]) + to_vector(three[:, 2]), 0.5)
            end
            data = (; ts=[0.5, 1.0, 2.0, 4.0], obs=[0.9, 0.8, 0.6, 0.4])
            with_jac = StanBlocks.SlicModel(source(true), data, @__MODULE__)
            without = StanBlocks.SlicModel(source(false), data, @__MODULE__)
            @test occursin("jtest_chain_jacobian_4_4", stan_code(with_jac))
            p = instantiate(with_jac)
            q = instantiate(without)
            @test BridgeStan.param_unc_names(p.model) == BridgeStan.param_unc_names(q.model)
            for x in ([0.1, -0.2, 0.3, 0.0, -0.5, 0.2, -0.1], [-0.4, 0.5, 0.0, 0.3, 0.6, -0.3, 0.4],
                      [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
                lp, g = BridgeStan.log_density_gradient(p.model, x)
                lq, h = BridgeStan.log_density_gradient(q.model, x)
                @test lp ≈ lq rtol=1e-7
                @test g ≈ h rtol=1e-6
            end

            check = StanBlocks.SlicModel(quote
                theta::vector[4] ~ lognormal(0.0, 0.5)
                tlag ~ lognormal(-1.0, 0.5)
                ka ~ lognormal(0.0, 0.5)
                ke ~ lognormal(-1.0, 0.5)
                y4::vector[4] ~ normal(0.0, 2.0)
                y3::vector[3] ~ normal(0.0, 2.0)
                tt ~ uniform(0.0, 3.0)
                err_chain = ode_jacobian_error(jtest_chain, jacobian(jtest_chain), tt, y4, theta, tlag)
                err_three = ode_jacobian_error(jtest_three, jacobian(jtest_three), tt, y3, ka, ke, 0.3)
                A::matrix[3, 3] ~ normal(0.0, 1.0)
                b::vector[3] ~ normal(0.0, 1.0)
                err_linear = ode_jacobian_error(jtest_linear, jacobian(jtest_linear), tt, y3, A, b)
                cap = to_array_1d([1.0, 2.0, 3.0])
                err_vectorised = ode_jacobian_error(jtest_vectorised, jacobian(jtest_vectorised), tt, y3, ka, cap)
                f = (t, y) -> [-ka * y[1] + ke * y[2] * y[3], ka * y[1] - ke * y[2], log1p_exp(y[3]) * tlag]
                err_closure = ode_jacobian_error(f, jacobian(f), tt, y3)
                # Positive control: a wrong Jacobian is detected.
                err_zero = ode_jacobian_error(jtest_chain, jtest_zero_jac, tt, y4, theta, tlag)
            end, (;), @__MODULE__)
            c = instantiate(check)
            names = BridgeStan.param_names(c.model; include_tp=true, include_gq=true)
            checked = ("err_chain", "err_three", "err_linear", "err_vectorised", "err_closure")
            idx = [only(findall(==(n), names)) for n in (checked..., "err_zero")]
            for seed in 1:5
                x = 0.7 .* randn(MersenneTwister(seed), BridgeStan.param_unc_num(c.model))
                vals = BridgeStan.param_constrain(c.model, x; include_tp=true, include_gq=true,
                    rng=BridgeStan.StanRNG(c.model, seed))
                @test all(<(1e-10), vals[idx[1:end-1]])
                @test vals[idx[end]] > 1e-2
            end
        end
    end
end
