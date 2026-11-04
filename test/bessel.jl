"""
The four integer-order Bessel builtins traced in Stan but had no scalar Julia
twins, so a downstream `@juliacompat` helper looked up an undefined name in
its defining module (snag `juliacompat-modi-62673aa9`). Exercise the public
macro from a fresh module without caller-provided Bessel bindings.
"""
@testitem "slic: @juliacompat integer-order Bessel functions" tags=[:slic, :regression] begin
    import StanBlocks
    const SF = StanBlocks.SpecialFunctions
    workspace = Module(gensym(:BesselJulia))
    Core.eval(workspace, :(using StanBlocks))
    pairs = ((:bessel_first_kind, SF.besselj), (:bessel_second_kind, SF.bessely),
        (:modified_bessel_first_kind, SF.besseli), (:modified_bessel_second_kind, SF.besselk))
    for (name, reference) in pairs
        @test !isdefined(workspace, name)
        scalar, vector, orders, matrix, row, array = Symbol.(string(name) .* ["_scalar", "_vector", "_orders", "_matrix", "_row", "_array"])
        Core.eval(workspace, quote
            @deffun @juliacompat begin
                $scalar(n::int, x::real)::real = $name(n, x)
                $vector(n::int, x::vector[m])::vector[m] = $name(n, x)
                $orders(n::int[m], x::vector[m])::vector[m] = $name(n, x)
                $matrix(n::int, x::matrix[m, k])::matrix[m, k] = $name(n, x)
                $row(n::int, x::row_vector[m])::row_vector[m] = $name(n, x)
                $array(n::int[m], x::real)::real[m] = $name(n, x)
            end
        end)
        call(f, args...) = Base.invokelatest(getglobal(workspace, f), args...)
        for n in (-3, -1, 0, 1, 2, 5), x in (0.01, 0.75, 2.5, 20.0)
            @test call(scalar, n, x) ≈ reference(n, x) rtol=2e-14
        end
        @test call(scalar, 1, 2) ≈ reference(1, 2.0)
        @test Base.return_types(getglobal(workspace, scalar), (Int, Float64)) == [Float64]
        xs, ns = [0.25, 1.5, 4.0], [-1, 0, 2]
        @test call(vector, 2, xs) ≈ reference.(2, xs)
        @test call(orders, ns, xs) ≈ reference.(ns, xs)
        @test call(array, ns, 1.5) ≈ reference.(ns, 1.5)
        @test call(matrix, 1, [0.5 1.0; 2.0 3.0]) ≈ reference.(1, [0.5 1.0; 2.0 3.0])
        @test call(row, 1, xs) ≈ reference.(1, xs)
        @test isempty(call(vector, 1, Float64[]))
        @test_throws DimensionMismatch call(orders, [0, 1], xs)
        @test_throws DimensionMismatch StanBlocks.jcall(Val(name), workspace, [0, 1], xs)
        # The descriptor evaluator uses a Module scope, while macro-generated
        # calls carry _JuliaCallScope{workspace}. Both reach the same twin.
        @test StanBlocks.jcall(Val(name), workspace, 1, 0.75) == call(scalar, 1, 0.75)
        if name in (:bessel_first_kind, :modified_bessel_first_kind)
            for n in (-2, -1, 0, 1, 2), x in (-2.0, 0.0)
                @test call(scalar, n, x) ≈ reference(n, x) atol=1e-15
            end
            @test_throws DomainError call(scalar, 1, NaN)
        else
            for x in (-2.0, -0.0, 0.0, -Inf)
                @test_throws DomainError call(scalar, 1, x)
            end
            @test isnan(call(scalar, 1, NaN))
            @test call(scalar, 1, Inf) == 0.0
        end
    end
end

"""
One compiled Stan program is the independent numerical oracle for all four
functions, including integer orders of either sign, zero, negative arguments
for J/I, NaNs and the positive-domain boundary of Y/K. Selecting the family
with data lets invalid calls be checked without affecting the other families.
"""
@testitem "slic: @juliacompat Bessel values agree with Stan" tags=[:slic, :regression, :bridgestan] setup=[StanBlocksImports] begin
    @deffun @juliacompat bessel_value(kind::int, n::int, x::real)::real = begin
        if kind == 1
            return bessel_first_kind(n, x)
        elseif kind == 2
            return bessel_second_kind(n, x)
        elseif kind == 3
            return modified_bessel_first_kind(n, x)
        else
            return modified_bessel_second_kind(n, x)
        end
    end
    @deffun finite_location(x::real)::real = is_nan(x) || is_inf(x) ? 0.0 : x
    model(kind, n) = @slic (; kind, n, observed = 0.2) begin
        x ~ normal(0.0, 1.0)
        # Keep x as a parameter without the observation's finite-location
        # requirement masking the Bessel functions' non-finite behavior.
        observed ~ normal(finite_location(x), 1.0)
        value = bessel_value(kind, n, x)
    end
    code = stan_code(model(1, 0))
    for name in ("bessel_first_kind", "bessel_second_kind", "modified_bessel_first_kind", "modified_bessel_second_kind")
        @test occursin(name * "(n, x)", code)
    end
    for kind in 1:4, n in (-3, -1, 0, 1, 2, 5)
        p = instantiate(stan_model(model(kind, n)))
        @test BridgeStan.param_unc_num(p.model) == 1
        names = BridgeStan.param_names(p.model; include_tp=true, include_gq=true)
        index = only(findall(==("value"), names))
        stan(x) = BridgeStan.param_constrain(p.model, [x]; include_tp=true, include_gq=true,
            rng=BridgeStan.StanRNG(p.model, 1))[index]
        xs = kind in (1, 3) ? [-2.0, 0.0, 0.01, 0.75, 2.5, 20.0, 700.0] : [0.01, 0.75, 2.5, 20.0, 700.0, 1000.0]
        for x in xs
            @test bessel_value(kind, n, x) ≈ stan(x) rtol=2e-12 atol=1e-300
        end
        if kind in (1, 3)
            @test_throws DomainError bessel_value(kind, n, NaN)
            @test_throws Exception stan(NaN)
            for x in (-Inf, Inf)
                if kind == 1
                    @test stan(x) == bessel_value(kind, n, x) == 0.0
                elseif n in (0, 1)
                    @test isnan(stan(x)) && isnan(bessel_value(kind, n, x))
                else
                    @test_throws Exception stan(x)
                    @test_throws Exception bessel_value(kind, n, x)
                end
            end
            if kind == 3
                @test_throws Exception stan(1000.0)
                @test_throws Exception bessel_value(kind, n, 1000.0)
            end
        else
            for x in (-2.0, 0.0)
                @test_throws DomainError bessel_value(kind, n, x)
                @test_throws Exception stan(x)
            end
            @test isnan(stan(NaN)) && isnan(bessel_value(kind, n, NaN))
            @test stan(Inf) == bessel_value(kind, n, Inf) == 0.0
        end
    end
end
