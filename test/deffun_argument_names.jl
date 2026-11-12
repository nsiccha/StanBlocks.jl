"""
A `@deffun` argument, or a signature dimension, may have any name a Julia local
can have (snag `deffun-argument-03fb3f25`). The tracing methods `@deffun` and
`@defsig` emit are evaluated in the author's module, keep locals of their own
(`x`, `info`, `context`) and call Base functions (`values`, `collect`, `vcat`,
`merge`, `pairs`). An argument named `values` used to become a Julia local
beside them, so `values(...)` called the argument and transpiling failed with
`objects of type StanExpr… are not callable`. A module whose own globals are
named like those Base functions broke every `@deffun` in it the same way.

Each colliding definition below has a twin whose argument is plainly named
`v`; the emitted Stan must be identical up to that renaming.
"""
@testmodule DeffunArgumentNameFixtures begin
    module Names
        using StanBlocks
        # Base functions the emitted tracing code calls, and its own locals.
        const NAMES = (:values, :collect, :vcat, :merge, :pairs, :error, :keys,
                       :info, :x, :context, :v)
        for name in NAMES
            declared = Symbol(:declared_, name)
            inferred = Symbol(:inferred_, name)
            looped = Symbol(:looped_, name)
            @eval @deffun $declared($name::vector[n], k::real)::vector[n] = begin
                $name .* k
            end
            @eval @deffun $inferred($name::vector[n], k::real) = begin
                $name .* k
            end
            # Reads its signature dimension, so the `int n = dims(…)[1];`
            # binding is emitted from the colliding argument.
            @eval @deffun $looped($name::vector[n], from::int)::real = begin
                acc = 0.0
                for i in from:n
                    acc += $name[i]
                end
                acc
            end
        end

        function sig_x end
        function sig_context end
        function sig_plain end
        @defsig typeof(sig_x) => begin
            (vector[x], vector[x]) => vector[x]
        end
        @defsig typeof(sig_context) => begin
            (vector[context], vector[context]) => vector[context]
        end
        @defsig typeof(sig_plain) => begin
            (vector[n], vector[n]) => vector[n]
        end

        # The Julia method's dimension preamble reads `size(<argument>, i)`.
        @deffun @juliacompat jc_size(size::vector[n], k::real)::vector[n] = begin
            out::vector[n] = size .* k
            out
        end
    end

    module Shadowing
        using StanBlocks
        # This module's OWN functions, not Base's. The code `@deffun` emits is
        # evaluated here and must not call them.
        values(_...) = throw(ArgumentError("Shadowing.values was called"))
        collect(_...) = throw(ArgumentError("Shadowing.collect was called"))
        vcat(_...) = throw(ArgumentError("Shadowing.vcat was called"))
        merge(_...) = throw(ArgumentError("Shadowing.merge was called"))
        pairs(_...) = throw(ArgumentError("Shadowing.pairs was called"))
        size(_...) = throw(ArgumentError("Shadowing.size was called"))
        @deffun shadowed_scale(a::vector[n], k::real)::vector[n] = begin
            a .* k
        end
        @deffun @juliacompat shadowed_jc(a::vector[n], k::real)::vector[n] = begin
            out::vector[n] = a .* k
            out
        end
        const model = @slic (; y = [0.3, 0.1]) begin
            s ~ normal(0.0, 1.0)
            y ~ normal(shadowed_scale(rep_vector(s, 2), 2.0) .+ shadowed_jc(rep_vector(s, 2), 0.5), 1.0)
        end
    end
end

@testitem "slic: @deffun arguments and @defsig dimensions may take any name" tags=[:slic, :regression, :stanc] setup=[StanBlocksImports, StanBlocksTestSetup, DeffunArgumentNameFixtures] begin
    using .StanBlocksTestSetup: stanc_compiles
    using .DeffunArgumentNameFixtures: Names, Shadowing
    uses(call) = StanBlocks.stan.SlicModel(quote
        s ~ normal(0.0, 1.0)
        y ~ normal($call, 1.0)
    end, Dict{Symbol,Any}(:y => [0.3, 0.1], :w => [1.0, 2.0, 3.0]), Names)
    word(s) = Regex("\\b" * s * "\\b")
    # The definition of `f` alone: builtins in the same program may use the
    # colliding name for their own parameters (`broadcasted_getindex(vector x, …)`).
    definition(code, f) = match(Regex("(?ms)^[^\\n]*\\b" * f * "\\(.*?^}\$"), code).match
    # A longer name can wrap a function header over several lines.
    tokens(code) = replace(replace(code, r"\s+" => " "), "( " => "(", " )" => ")")
    calls = (
        declared = f -> :($f(rep_vector(s, 2), 2.0)),
        inferred = f -> :($f(rep_vector(s, 2), 2.0)),
        looped = f -> :(rep_vector($f(w, 2) * s, 2)),
    )
    for name in Names.NAMES, (kind, call) in pairs(calls)
        name === :v && continue
        @testset "$kind $name" begin
            f, twin = Symbol(kind, :_, name), Symbol(kind, :_v)
            m = uses(call(f))
            @test stanc_compiles(m)
            code = stan_code(m)
            def = definition(code, string(f))
            renamed_def = replace(def, word(string(f)) => string(twin), word(string(name)) => "v")
            @test tokens(replace(replace(code, def => renamed_def), word(string(f)) => string(twin))) ==
                tokens(stan_code(uses(call(twin))))
        end
    end

    # `@defsig`: a dimension named like the tracing method's own `x` / `context`.
    defsig_code(f) = stan_code(uses(:($f(rep_vector(s, 2), rep_vector(s, 2)))))
    for f in (:sig_x, :sig_context)
        @test replace(defsig_code(f), word(string(f)) => "sig_plain") == defsig_code(:sig_plain)
    end

    # `@juliacompat`: an argument named `size`, both targets.
    @test Names.jc_size([1.0, 2.0], 3.0) == [3.0, 6.0]
    @test stanc_compiles(uses(:(jc_size(rep_vector(s, 2), 2.0))))

    # A module's own `values`/`merge`/`pairs`/`collect`/`vcat`/`size` globals.
    @test stanc_compiles(Shadowing.model)
    @test Shadowing.shadowed_jc([1.0, 2.0], 3.0) == [3.0, 6.0]
end
