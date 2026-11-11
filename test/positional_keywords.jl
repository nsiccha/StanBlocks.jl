"""
A Stan function takes positional arguments only, and the emitter prints only
those, so a keyword on any call that became a plain Stan call vanished:
`sum(v; dims=1)` emitted `sum(v)`, `exp(s; foo=1)` emitted `exp(s)`, and an
`@deffun` declared without keywords dropped one silently (todo 18nseyb, found
under snag mean-vector-trac-306ef1b4, which refused only the `Statistics` twins).
Such a keyword is refused where the call is typed. A keyword still reaches every
consumer that reads it: a distribution call's bounds and shape, an `@deffun`'s
declared keywords, a sub-model's inputs and `plate`'s.
"""
@testitem "slic: a keyword on a positional-only call is refused, not dropped" tags=[:slic, :regression, :stanc] setup=[StanBlocksImports, StanBlocksTestSetup] begin
    using .StanBlocksTestSetup: stan_block, stanc_compiles
    @deffun begin
        pk_positional(x::real)::real = x + 1.0
        pk_keyword(x::real; k::real = 1.0)::real = x + k
    end
    data = (; n = 2, y = 0.3)
    model(rhs) = StanBlocks.stan.SlicModel(quote
        v::vector[n] ~ normal(0.0, 1.0)
        s ~ normal(0.0, 1.0)
        out = $rhs
        y ~ normal(sum(out) + s, 1.0)
    end, data, @__MODULE__)
    errmsg(m) = try (stan_code(m); "") catch e; sprint(showerror, e) end
    refused(rhs, callee, key) = begin
        msg = errmsg(model(rhs))
        ok = occursin("`$(callee)` has no keyword argument `$(key)`", msg)
        ok || println("unexpected outcome for `", rhs, "`: ", first(msg, 600))
        ok
    end

    # Each emitted the call without its keyword.
    @test refused(:(sum(v; dims = 1)), :sum, :dims)
    @test refused(:(exp(s; foo = 1)), :exp, :foo)
    @test refused(:(log1p(exp(s); foo = 1)), :log1p, :foo)
    @test refused(:(square(s; foo = 1)), :square, :foo)
    @test refused(:(rep_vector(s, n; foo = 1)), :rep_vector, :foo)
    @test refused(:(dot_product(v, v; foo = 1)), :dot_product, :foo)
    @test refused(:(log_sum_exp(v; foo = 1)), :log_sum_exp, :foo)
    @test refused(:(+(s, 1.0; foo = 1)), :+, :foo)
    @test refused(:(normal_lpdf(0.3, s, 1.0; foo = 1)), :normal_lpdf, :foo)
    @test refused(:(pk_positional(s; foo = 1)), :pk_positional, :foo)
    # The `Statistics` twins keep their own message.
    @test occursin("takes no keyword arguments in Stan", errmsg(model(:(mean(v; dims = 1)))))

    # Keyword consumers keep working: an `@deffun`'s declared keyword, a
    # distribution's bounds and shape, a sub-model's inputs and `plate`'s.
    declared = @slic (; y = 0.3) begin
        s ~ normal(0.0, 1.0)
        y ~ normal(pk_keyword(s; k = 2.0), 1.0)
    end
    @test occursin("kwcall_pk_keyword(2.0, s)", stan_code(declared))
    @test stanc_compiles(declared)
    inner = @slic begin
        m ~ normal(x, 1.0)
        return m
    end
    consumers = @slic (; y = [0.3, 0.2], hi = 2.0) begin
        z ~ std_normal(; n = 2)
        sig ~ normal(0.0, 1.0; lower = 0.0)
        c ~ inner(; x = 1.0)
        p ~ plate(; outer = (2,)) do g
            e ~ normal(c, sig); e
        end
        y ~ normal(z .+ p, sig; upper = hi)
    end
    @test stanc_compiles(consumers)
end
