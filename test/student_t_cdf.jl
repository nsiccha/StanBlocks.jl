"""
Stan Math 5.3.0/5.4.0 `student_t_lcdf`/`student_t_lccdf` have a NaN gradient
where the bound equals the location: they work with q = ν/t², and at t = 0 the
Jacobian 2r²q/t is 0·∞/0, although the value log(1/2) is right. A censored
Student-t observation whose limit equals the location therefore had a NaN
gradient (snag `native-student-t-b7023f91`). The same kernels lose the tail mass
to `1 - x` cancellation and underflow. `censored`/`truncated`/`interval_censored`
(`student_t`, …) and `truncated_student_t` now take their tail masses from
`student_t_lcdf_stable`/`student_t_lccdf_stable`. Values and gradients in the
location, log scale and log degrees of freedom are checked against a 256-bit
reference at t = 0, in the series, middle and far branches, and in tails whose
mass underflows a double.
"""
@testitem "slic: Student-t censoring tail masses keep exact gradients at the location and in deep tails" tags=[:slic, :regression, :stanc, :bridgestan] setup=[StanBlocksImports, StanBlocksTestSetup] begin
    using .StanBlocksTestSetup: stanc_compiles, stan_block
    const SF = StanBlocks.SpecialFunctions
    setprecision(BigFloat, 256) do
        # Regularized incomplete beta by Lentz's continued fraction, at 256 bits.
        function betacf(a, b, x)
            local c, d, h
            tiny = big"1e-300"
            fix(v) = abs(v) < tiny ? tiny : v
            c = one(x); d = inv(fix(1 - (a + b) * x / (a + 1))); h = d
            for m in 1:200000
                e = m * (b - m) * x / ((a - 1 + 2m) * (a + 2m))
                d = inv(fix(1 + e * d)); c = fix(1 + e / c); h *= d * c
                e = -(a + m) * (a + b + m) * x / ((a + 2m) * (a + 1 + 2m))
                d = inv(fix(1 + e * d)); c = fix(1 + e / c); h *= d * c
                abs(d * c - 1) < eps(x) && return h
            end
            error("betacf did not converge")
        end
        pre(a, b, x) = exp(a * log(x) + b * log1p(-x) - SF.logbeta(a, b))
        ibeta(a, b, x) = x == 1 ? one(x) :
            x < (a + 1) / (a + b + 2) ? pre(a, b, x) * betacf(a, b, x) / a : 1 - pre(a, b, x) * betacf(b, a, 1 - x) / b
        # log P(T_ν ≤ t): the one-sided tail I_x(ν/2, 1/2)/2, x = ν/(ν+t²).
        tail(t, nu) = ibeta(nu / 2, big(1) / 2, nu / (nu + t^2)) / 2
        lcdf(t, nu) = t < 0 ? log(tail(t, nu)) : log1p(-tail(t, nu))
        lpdf(y, nu, mu, s) = (z = (y - mu) / s; SF.loggamma((nu + 1) / 2) - SF.loggamma(nu / 2) -
            log(nu * big(pi)) / 2 - log(s) - (nu + 1) / 2 * log1p(z^2 / nu))
        step = big"1e-30"
        grad(f, x) = [(f(x .+ step .* (1:3 .== i)) - f(x .- step .* (1:3 .== i))) / 2step for i in 1:3]
        check(problem, f, x) = begin
            value, gradient = BridgeStan.log_density_gradient(problem.model, x; propto = false, jacobian = false)
            reference, dreference = f(big.(x)), grad(f, big.(x))
            @test isfinite(value) && all(isfinite, gradient)
            @test value ≈ reference rtol = 1e-12 atol = 1e-12
            @test gradient ≈ dreference rtol = 1e-11 atol = 1e-15
            gradient
        end

        # One atom per instantiation: at `lo` it takes the lcdf, at `hi` the
        # lccdf. Parameters are the location, log scale and log dof.
        lo, hi = -0.5, 0.7
        censored_t = @slic (; y = [lo], lo, hi) begin
            mu ~ flat(); ls ~ flat(); lnu ~ flat()
            y ~ censored(student_t(exp(lnu), mu, exp(ls)); lower = lo, upper = hi)
        end
        functions = stan_block(stan_code(censored_t), "functions")
        @test occursin("student_t_lcdf_stable(", functions)
        @test occursin("student_t_lccdf_stable(", functions)
        @test !occursin(r"(?<!\w)student_t_lc?cdf\(", functions)
        @test stanc_compiles(censored_t)
        lower = instantiate(stan_model(censored_t))
        upper = instantiate(stan_model(censored_t(; y = [hi])))
        lower_ref(x) = lcdf((lo - x[1]) / exp(x[2]), exp(x[3]))
        upper_ref(x) = lcdf((x[1] - hi) / exp(x[2]), exp(x[3]))

        # The reported case: limit = location, ν = 4, σ = 0.5. log(1/2), and
        # d/dμ = -density/CDF = -1.5; the scale and dof leave F(0) = 1/2.
        g = check(lower, lower_ref, [lo, log(0.5), log(4.0)])
        @test g ≈ [-1.5, 0.0, 0.0] atol = 1e-14
        g = check(upper, upper_ref, [hi, log(0.5), log(4.0)])
        @test g ≈ [1.5, 0.0, 0.0] atol = 1e-14

        # t = 0, the series (|t| ≲ 1e-4), the middle branch (t² ≤ 4), the far
        # branch, and tails whose mass is below a double's range (log space).
        s = 0.5
        points = [(nu, t) for nu in (0.5, 4.0, 30.0) for t in (0.0, 1e-5, -1e-5, 1e-3, -1e-3, 1.5, -1.5, 3.0, -3.0, 50.0, -50.0, 1e4, -1e4)]
        append!(points, [(30.0, -1e12), (600.0, -100.0), (2000.0, -1000.0), (2000.0, 1000.0)])
        for (nu, t) in points
            check(lower, lower_ref, [lo - t * s, log(s), log(nu)])
            check(upper, upper_ref, [hi + t * s, log(s), log(nu)])
        end

        # The other selectors and the legacy LOQ family, each with its limit at
        # the location: truncation divides by the lccdf at `lo`, the interval
        # (-∞, whi] takes the lcdf at `whi` (and -∞ at its lower end), and
        # `truncated_student_t` takes the lcdf at its lower LOQ.
        mixed = @slic (; x = 0.3, lo, w = 0.1, wlo = -Inf, whi = 0.9, obs = [lo], lloq = [lo], uloq = [10.0]) begin
            mu ~ flat(); ls ~ flat(); lnu ~ flat()
            x ~ truncated(student_t(exp(lnu), mu, exp(ls)); lower = lo)
            w ~ interval_censored(student_t, wlo, whi, exp(lnu), mu, exp(ls))
            obs ~ truncated_student_t(rep_vector(exp(lnu), 1), rep_vector(mu, 1), rep_vector(exp(ls), 1), lloq, uloq)
        end
        @test !occursin(r"(?<!\w)student_t_lc?cdf\(", stan_block(stan_code(mixed), "functions"))
        @test stanc_compiles(mixed)
        mixed_problem = instantiate(stan_model(mixed))
        mixed_ref(z) = begin
            mu, sig, nu = z[1], exp(z[2]), exp(z[3])
            lpdf(big(0.3), nu, mu, sig) - lcdf((mu - lo) / sig, nu) + lcdf((big(0.9) - mu) / sig, nu) +
                lcdf((lo - mu) / sig, nu)
        end
        for mu in (lo, 0.9, 0.2), nu in (0.5, 4.0, 30.0)
            check(mixed_problem, mixed_ref, [mu, log(s), log(nu)])
        end
    end
end
