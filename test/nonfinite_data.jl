using TestItemRunner

"""
A real datum may be non-finite: an observation-shaped `censored` bound holds
`-Inf` or `Inf` for a row with no bound on that side. Writing the data for
BridgeStan raised JSON.jl's `-Inf not allowed to be written in JSON spec`
before Stan saw it (snag `stan-json-infini-725445b0`). The data file now
carries Stan's bare `Infinity`/`-Infinity`/`NaN` tokens. Below, lower, upper and
two-sided bounds mix censored atoms with ordinary rows for a normal and a
Student-t family, and the log density and gradient match an independent closed
form. The `yflag` mean adds `is_inf`/`is_nan` of five data entries, a matrix
cell included, so the match also proves each one arrives exact rather than as
a large finite stand-in.
"""
@testitem "slic: non-finite real data reaches BridgeStan exactly" tags=[:slic, :regression, :bridgestan] setup=[StanBlocksImports] begin
    using StanBlocks.SpecialFunctions: erfc, loggamma, beta_inc

    s, nu = 0.5, 4.0
    logPhi(z) = log(erfc(-z / sqrt(2)) / 2)
    n_lpdf(y, mu) = -0.5 * ((y - mu) / s)^2 - log(s * sqrt(2pi))
    n_lcdf(y, mu) = logPhi((y - mu) / s)
    n_lccdf(y, mu) = logPhi((mu - y) / s)
    t_lpdf(y, mu) = loggamma((nu + 1) / 2) - loggamma(nu / 2) - 0.5 * log(nu * pi) - log(s) -
        (nu + 1) / 2 * log1p(((y - mu) / s)^2 / nu)
    # Mass of the Student-t beyond |y - mu| on one side.
    t_tail(y, mu) = beta_inc(nu / 2, 0.5, nu / (nu + ((y - mu) / s)^2))[1] / 2
    t_lcdf(y, mu) = y < mu ? log(t_tail(y, mu)) : log1p(-t_tail(y, mu))
    t_lccdf(y, mu) = y > mu ? log(t_tail(y, mu)) : log1p(-t_tail(y, mu))
    # Each row's term follows its known evidence category, not a comparison with
    # its bound: an atom at a finite bound takes the tail mass, and every other
    # row, a row whose bound is infinite included, takes the density.
    family(mu, lpdf, lcdf, lccdf) =
        lcdf(-0.5, mu) + lpdf(0.3, mu) +                       # lower
        lccdf(0.7, mu) + lpdf(0.3, mu) +                       # upper
        lcdf(-0.5, mu) + lpdf(0.3, mu) + lccdf(0.7, mu) +      # two-sided
        lpdf(0.1, mu)                                          # scalar bound Inf
    exact(mu) = -0.5 * mu^2 - 0.5 * log(2pi) + n_lpdf(0.4, mu) +
        family(mu, n_lpdf, n_lcdf, n_lccdf) + family(mu, t_lpdf, t_lcdf, t_lccdf)

    data = (; yl=[-0.5, 0.3], lo=[-0.5, -Inf], yu=[0.7, 0.3], hi=[0.7, Inf],
              yb=[-0.5, 0.3, 0.7], blo=[-0.5, -Inf, -Inf], bhi=[Inf, Inf, 0.7],
              yc=0.1, cap=Inf, w=NaN, A=[1.0 -Inf; 0.0 2.0], yflag=5.4)
    saved = deepcopy(data)
    model = @slic data begin
        mu ~ normal(0.0, 1.0)
        yl ~ censored(normal(mu, 0.5); lower=lo)
        yu ~ censored(normal(mu, 0.5); upper=hi)
        yb ~ censored(normal(mu, 0.5); lower=blo, upper=bhi)
        yc ~ censored(normal(mu, 0.5); upper=cap)
        tl ~ censored(student_t(4.0, mu, 0.5); lower=lo)
        tu ~ censored(student_t(4.0, mu, 0.5); upper=hi)
        tb ~ censored(student_t(4.0, mu, 0.5); lower=blo, upper=bhi)
        tc ~ censored(student_t(4.0, mu, 0.5); upper=cap)
        yflag ~ normal(mu + is_inf(cap) + is_inf(lo[2]) + is_inf(bhi[1]) + is_nan(w) + is_inf(A[1, 2]), 0.5)
    end
    model = model(; tl=data.yl, tu=data.yu, tb=data.yb, tc=data.yc)

    json = stan_data_json(model)
    @test occursin("-Infinity", json) && occursin("NaN", json)
    parsed = StanBlocks.JSON.parse(json; allownan=true)
    @test isequal(Float64.(parsed["lo"]), data.lo)
    @test isequal(Float64.(parsed["bhi"]), data.bhi)
    @test parsed["cap"] == Inf && isnan(parsed["w"])
    # Stan nests a matrix by its first index: row 1 is `[1.0, -Inf]`.
    @test isequal(Float64.(parsed["A"][1]), [1.0, -Inf])

    problem = instantiate(model)
    @test BridgeStan.param_unc_names(problem.model) == ["mu"]
    for mu in (-0.4, 0.0, 0.6)
        value, gradient = BridgeStan.log_density_gradient(problem.model, [mu]; propto=false, jacobian=false)
        @test value ≈ exact(mu) atol = 1e-12 rtol = 1e-12
        h = 1e-5
        @test only(gradient) ≈ (exact(mu + h) - exact(mu - h)) / 2h atol = 1e-7 rtol = 1e-8
    end
    @test isequal(data, saved)
end
