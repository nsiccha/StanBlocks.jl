"""
A body can hold a Julia function VALUE where it would usually name a builtin: a
producer that splices the evaluated callable into the body (BRM's model scope
builds `Expr(:call, f, args...)`) hands StanBlocks `Statistics.mean`, not the
name `mean`. The builtin `mean` is a separate function, so the value matched no
tracetype, and `u - mean(u)` failed with `tracetype not defined for
mean(::vector)::anything` (snag `mean-vector-trac-306ef1b4`). Each Julia function
in the `_slic_callable` table now traces exactly like the builtin's name, and the
keyword forms of the `Statistics` functions, which Stan does not have, are refused
instead of dropped from the emitted call.
"""
@testitem "slic: a Julia function value traces as the builtin of the same computation" tags=[:slic, :regression, :stanc] setup=[StanBlocksImports, StanBlocksTestSetup] begin
    using .StanBlocksTestSetup: stan_block, stanc_compiles
    SF = StanBlocks.SpecialFunctions
    data = (; m=3, n=2, y=0.3, p=0.25, z=[0.1, 0.4, 0.2])
    model(rhs, use=:out) = StanBlocks.stan.SlicModel(quote
        v::vector[n] ~ normal(0.0, 1.0)
        X::matrix[m, n] ~ normal(0.0, 1.0)
        s ~ normal(0.0, 1.0)
        out = $rhs
        y ~ normal($use + sum(v) + sum(X) + s, 1.0)
    end, data, @__MODULE__)
    errmsg(m) = try (stan_code(m); "") catch e; sprint(showerror, e) end

    # The reported shape: centring a vector by its mean.
    reported = model(Expr(:call, -, :v, Expr(:call, Statistics.mean, :v)), :(sum(out)))
    @test occursin("vector[n] out = (v - mean(v));",
                   stan_block(stan_code(reported), "transformed parameters"))
    @test stanc_compiles(reported)

    # Every pair: the value emits exactly what the builtin's name emits.
    cases = [
        (Statistics.mean, :mean, (:X,), :out),
        (Statistics.std, :sd, (:v,), :out),
        (Statistics.var, :variance, (:X,), :out),
        (Statistics.quantile, :quantile, (:z, :p), :out),
        (SF.erf, :erf, (:s,), :out),
        (SF.erfc, :erfc, (:s,), :out),
        (SF.digamma, :digamma, (:s,), :out),
        (SF.trigamma, :trigamma, (:s,), :out),
        (Base.log2, :log2, (:s,), :out),
        (Base.cbrt, :cbrt, (:s,), :out),
        (Base.fma, :fma, (:s, :s, :s), :out),
        (Base.hcat, :hcat, (:v, :v), :(sum(out))),
        (Base.reshape, :reshape, (:v, 1, 2), :(sum(out))),
    ]
    for (f, name, args, use) in cases
        by_value = model(Expr(:call, f, args...), use)
        @test stan_code(by_value) == stan_code(model(Expr(:call, name, args...), use))
        @test stanc_compiles(by_value)
    end
    # So do Julia's `Statistics` names written in a body that has them in scope.
    @test stan_code(model(:(std(v)))) == stan_code(model(:(sd(v))))
    @test stan_code(model(:(var(X)))) == stan_code(model(:(variance(X))))

    # refused: Stan's `mean`/`sd`/`variance`/`quantile` take no keywords, and the
    # positional signature used to drop them, emitting `mean(X)` for
    # `mean(X; dims=1)` (a silent miscompile, `dev` §1).
    for rhs in (
        Expr(:call, Statistics.mean, Expr(:parameters, Expr(:kw, :dims, 1)), :X),
        :(mean(X; dims=1)),
        Expr(:call, Statistics.std, Expr(:parameters, Expr(:kw, :corrected, false)), :v),
        :(sd(v; corrected=false)),
        :(quantile(z, p; sorted=true)),
    )
        @test occursin("takes no keyword arguments in Stan", errmsg(model(rhs)))
    end
end
