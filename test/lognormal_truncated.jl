"""
`lognormal_truncated_mean(mu, sigma, lower)` and
`lognormal_truncated_variance(mu, sigma, lower)` are the normalized moments of
X | X ≥ lower, X ~ LogNormal(mu, sigma), and `lognormal_lccdf_stable(y, mu,
sigma)` is log P(X ≥ y) (snag `bounded-lognorma-c225e536`). The textbook forms
cancel exactly where truncation matters: Φ(-x) underflows past x ≈ 38, and the
variance is a second difference that loses ~1/σ² of relative precision for a
narrow scale. Values and gradients in the location, log scale and log bound are
checked against a 256-bit reference in every branch (continued fraction in
divided-difference arithmetic, Taylor series about the midpoint, direct
difference), at the two reported points, and where Stan's own `lognormal_lccdf`
returns -Inf. The `lognormal` family's censoring companions select the stable
pair, and a bound ≤ 0 gives the untruncated moments.
"""
@testitem "slic: lower-truncated LogNormal moments and survival keep exact values and gradients in deep tails and narrow scales" tags=[:slic, :regression, :stanc, :bridgestan] setup=[StanBlocksImports, StanBlocksTestSetup] begin
    using .StanBlocksTestSetup: stanc_compiles, stan_block
    const SF = StanBlocks.SpecialFunctions
    # One observation family returning one quantity selected by data `k`:
    # 1 mean, 2 variance, 3 the stable log survival, 4 Stan's own lognormal_lccdf.
    @deffun begin
        @lpxf lntprobe_lpdf(y::vector[n], k::int, mu::real, s::real, lb::real)::real =
            k == 1 ? lognormal_truncated_mean(mu, s, lb) :
                (k == 2 ? lognormal_truncated_variance(mu, s, lb) :
                    (k == 3 ? lognormal_lccdf_stable(lb, mu, s) :
                    (k == 4 ? lognormal_lccdf(lb, mu, s) : lognormal_lcdf_stable(lb, mu, s))))
        lntprobe_lpdfs(y::vector[n], k::int, mu::real, s::real, lb::real)::vector[n] = begin
            out::vector[n]
            for i in 1:n
                out[i] = lntprobe_lpdf(y[i:i], k, mu, s, lb)
            end
            out
        end
        lntprobe_rng(vector[n], k::int, mu::real, s::real, lb::real)::vector[n] = begin
            out::vector[n]
            for i in 1:n
                out[i] = 0.0
            end
            out
        end
    end
    setprecision(BigFloat, 256) do
        Φ(t) = SF.erfc(-t / sqrt(big(2))) / 2
        # Parameters: location, log scale, log bound.
        ref(k, x) = begin
            mu, s, lb = x[1], exp(x[2]), exp(x[3])
            a = (log(lb) - mu) / s
            # log1p of the small side: log Φ(20) = -2.8e-89 rounds to 0 at 256 bits.
            k == 3 && return a < 0 ? log1p(-Φ(a)) : log(Φ(-a))
            k == 5 && return a > 0 ? log1p(-Φ(-a)) : log(Φ(a))
            m = exp(mu + s^2 / 2) * Φ(s - a) / Φ(-a)
            k == 1 ? m : exp(2mu + 2s^2) * Φ(2s - a) / Φ(-a) - m^2
        end
        step = big"1e-30"
        grad(f, x) = [(f(x .+ step .* (1:3 .== i)) - f(x .- step .* (1:3 .== i))) / 2step for i in 1:3]

        probe = @slic (; y = [0.0], k = 1) begin
            mu ~ flat(); ls ~ flat(); llb ~ flat()
            y ~ lntprobe(k, mu, exp(ls), exp(llb))
        end
        @test stanc_compiles(probe)
        problems = [instantiate(stan_model(probe(; k))) for k in 1:5]
        evaluate(k, x) = BridgeStan.log_density_gradient(problems[k].model, x; propto = false, jacobian = false)
        # The absolute floor only admits a mass within 1e-300 of 1, whose log
        # is 0 in a double.
        check(k, mu, s, lb; rtol = 1e-12, grtol = 1e-11) = begin
            x = [mu, log(s), log(lb)]
            value, gradient = evaluate(k, x)
            reference, dreference = ref(k, big.(x)), grad(z -> ref(k, z), big.(x))
            @test isfinite(value) && all(isfinite, gradient)
            @test value ≈ reference rtol = rtol atol = 1e-300
            @test gradient ≈ dreference rtol = grtol atol = 1e-300
            value
        end

        # The reported points: deep lower tails, x = (log L - μ)/σ = 15 and 40.
        # Reference values from the reporter's independent 80-digit script.
        @test check(1, -3.0, 0.2, 1.0) ≈ 1.0133936271158251321640 rtol = 1e-14
        @test check(2, -3.0, 0.2, 1.0) ≈ 0.0001826210272836557383737 rtol = 1e-13
        @test check(3, -3.0, 0.2, 1.0) ≈ -116.1313848457116952359 rtol = 1e-14
        @test check(1, -40.0, 1.0, 1.0) ≈ 1.0256078378620389719011 rtol = 1e-14
        @test check(2, -40.0, 1.0, 1.0) ≈ 0.0006893027166535609095551 rtol = 1e-13
        @test check(3, -40.0, 1.0, 1.0) ≈ -804.6084420137537881666 rtol = 1e-14
        # Stan's own lognormal_lccdf has lost that mass: -Inf, gradient NaN.
        native, dnative = evaluate(4, [-40.0, 0.0, 0.0])
        @test native == -Inf
        @test !all(isfinite, dnative)

        # Every branch: continued fraction (x - 2σ ≥ 2), including σ far below
        # the distance to the bound; Taylor series (σ < 0.5) on both sides of
        # the bound, down to σ = 1e-6; direct differences (σ ≥ 0.5); a bound
        # deep in the lower tail of the distribution (x ≪ 0), where the log
        # survival is a tiny negative number and must stay accurate relative to
        # its size (an event model takes log1m_exp of it).
        points = [
            (0.0, 0.2, 1.0), (3.0, 0.8, 1.0), (-0.03, 0.01, 1.0), (-3.0, 1e-4, 1.0),
            (-0.3, 0.2, 1.0), (0.5, 0.1, 1.0), (-1.5e-6, 1e-6, 1.0), (-1.0, 0.8, 1.0),
            (-4.0, 1.5, 1.0), (0.0, 0.3, 1e-10), (-2.0, 0.3, 0.5), (1.0, 0.05, 3.5),
            (-0.2, 0.49, 1.0), (-0.2, 0.51, 1.0), (-1.2, 0.5, 2.0), (-10.0, 2.0, 5.0),
            (-1e3, 3.0, 2.0), (2.0, 5.0, 7.0), (2.0, 0.1, 1.0), (8.0, 0.2, 1.0),
        ]
        for (mu, s, lb) in points, k in (1, 2, 3, 5)
            check(k, mu, s, lb)
        end

        # A bound ≤ 0 truncates nothing: the plain LogNormal moments.
        untruncated = @slic (; y = [0.0], k = 1, lb = 0.0) begin
            mu ~ flat(); ls ~ flat()
            y ~ lntprobe(k, mu, exp(ls), lb)
        end
        for k in 1:3, lb in (0.0, -2.0)
            value, gradient = BridgeStan.log_density_gradient(
                instantiate(stan_model(untruncated(; k, lb))).model, [0.3, log(0.7)];
                propto = false, jacobian = false)
            @test value ≈ (k == 1 ? exp(0.3 + 0.245) : k == 2 ? expm1(0.49) * exp(0.6 + 0.49) : 0.0) rtol = 1e-14
            @test all(isfinite, gradient)
        end

        # The family's censoring/truncation companions select the stable pair.
        lo, hi = 0.5, 2.0
        censored_ln = @slic (; y = [lo], lo, hi) begin
            mu ~ flat(); ls ~ flat()
            y ~ censored(lognormal(mu, exp(ls)); lower = lo, upper = hi)
        end
        functions = stan_block(stan_code(censored_ln), "functions")
        @test occursin("lognormal_lcdf_stable(", functions)
        @test occursin("lognormal_lccdf_stable(", functions)
        @test !occursin(r"(?<!\w)lognormal_lc?cdf\(", functions)
        @test stanc_compiles(censored_ln)
        # A truncated observation whose bound sits 40σ above the location keeps
        # a finite log density: lpdf - log P(X ≥ L).
        truncated_ln = @slic (; y = [1.2], lb = 1.0) begin
            mu ~ flat(); ls ~ flat()
            y ~ truncated(lognormal(mu, exp(ls)); lower = lb)
        end
        value, gradient = BridgeStan.log_density_gradient(
            instantiate(stan_model(truncated_ln)).model, [-40.0, 0.0]; propto = false, jacobian = false)
        lpdf(x) = -log(big(1.2)) - x[2] - log(2big(pi)) / 2 - (log(big(1.2)) - x[1])^2 / (2exp(2x[2]))
        tref(x) = lpdf(x) - ref(3, [x[1], x[2], big(0.0)])
        @test value ≈ tref(big.([-40.0, 0.0])) rtol = 1e-13
        @test gradient ≈ [(tref(big.([-40.0, 0.0]) .+ step .* (1:2 .== i)) - tref(big.([-40.0, 0.0]) .- step .* (1:2 .== i))) / 2step for i in 1:2] rtol = 1e-11
    end
end
