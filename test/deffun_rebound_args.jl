"""
Stan passes function arguments and loop variables read-only, so a `@deffun`
body that rebinds an argument, `mg = mg ./ fmax(nfit, 1.0)`, transpiled to
`mg = (mg ./ fmax(nfit, 1.0));`, which stanc rejects ("Cannot assign to function
argument or loop identifier", todo 1d4vme2). In Julia that rebinds a local
name, so the body now reads a rebound argument through a local copy. A write
into an argument, which in Julia would change the caller's array, and a rebound
loop variable are refused at definition time, naming the fix.
"""
@testitem "slic: a @deffun that rebinds an argument reads a local copy" tags=[:slic, :regression, :stanc] setup=[StanBlocksImports, StanBlocksTestSetup] begin
    using .StanBlocksTestSetup: stan_block, stanc_compiles
    @deffun rb_norm(mg::vector[k], nfit::vector[k])::vector[k] = begin
        mg = mg ./ fmax(nfit, 1.0)
        mg
    end
    @deffun rb_accumulate(x::real, n::int)::real = begin
        for i in 1:n
            x += i
        end
        x
    end
    m = @slic (; y = [0.3, 0.1], f = [2.0, 0.5]) begin
        s ~ normal(0.0, 1.0)
        z = rb_norm(rep_vector(s, 2), f)
        y ~ normal(z .+ rb_accumulate(s, 3), 1.0)
    end
    code = stan_code(m)
    functions = stan_block(code, "functions")
    @test occursin(r"vector\[.+\] mg__local = mg;", functions)
    @test occursin("mg__local = (mg__local ./ fmax(nfit, 1.0));", functions)
    @test occursin("real x__local = x;", functions)
    @test stanc_compiles(m)

    # refused where the Stan definition is built: todo 1d4vme2.
    @deffun @juliacompat rb_write(v::vector[k])::vector[k] = begin
        v[1] = 0.0
        v
    end
    @deffun rb_bcast(v::vector[k])::vector[k] = begin
        v .= v .* 2.0
        v
    end
    @deffun rb_loopvar(n::int)::int = begin
        acc = 0
        for i in 1:n
            i = i + 1
            acc += i
        end
        acc
    end
    errmsg(m) = try (stan_code(m); "") catch e; sprint(showerror, e) end
    uses(call) = StanBlocks.stan.SlicModel(quote
        s ~ normal(0.0, 1.0)
        y ~ normal(sum($call) + s, 1.0)
    end, Dict{Symbol,Any}(:y => 0.3, :w => [1.0, 2.0], :n => 3), @__MODULE__)
    @test occursin("writes into its argument `v`", errmsg(uses(:(rb_write(w)))))
    @test occursin("writes into its argument `v`", errmsg(uses(:(rb_bcast(w)))))
    @test occursin("assigns its loop variable `i`", errmsg(uses(:(rb_loopvar(n)))))
    # The Julia method keeps Julia's semantics: the write reaches the caller's array.
    w = [1.0, 2.0]
    @test rb_write(w) == [0.0, 2.0] && w == [0.0, 2.0]
end
