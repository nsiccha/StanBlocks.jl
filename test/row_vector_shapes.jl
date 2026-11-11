using TestItemRunner

@testsnippet RowVectorShapeHelpers begin
    using StanBlocks
    import BridgeStan, LogDensityProblems

    @deffun begin
        row_literal(x::row_vector[5])::vector[5] = begin
            column = x'
            shifted = column + 0.25
            shifted
        end
        row_named(x::row_vector[n])::vector[n] = begin
            column::vector[n] = x'
            column * 2.0
        end
        row_inferred(x) = begin
            column = x'
            column * 3.0
        end
        row_generic(x::any_vector[n])::vector[n] = begin
            column::vector[n] = to_vector(x)
            column * 2.0
        end
        row_pair(x::row_vector[n], y::vector[n])::vector[n] = begin
            column = x'
            column + y
        end
        row_array(x::row_vector[m, n])::vector[n] = begin
            column = x[1]'
            column * 4.0
        end
        row_array_call(x::row_vector[n])::vector[n] = begin
            rows::row_vector[2, n]
            rows[1] = x
            rows[2] = x
            row_array(rows)
        end
        row_tuple(x::ntup) = begin
            column = x.row'
            column * x.scale
        end
        row_lazy(x::vector[n])::vector[n] = begin
            row = x[1:num_elements(x)]'
            column = row'
            column
        end
    end
    @deffun @inline row_inline(x::any_vector[n])::vector[n] = begin
        column::vector[n]
        for i in 1:n
            column[i] = x[i]
        end
        column
    end
    const row_value = ValueUDF(:row_value,
        :(row_value(x::any_vector[n])::vector[n] = begin
            column::vector[n] = to_vector(x)
            column * 2.0
        end), @__MODULE__)

    data = (; x=[-0.8, 0.3, 1.1, -0.2, 0.6], y=zeros(5))
    model = @slic data begin
        a ~ std_normal()
        scaled = x * a
        row = scaled'
        literal = row_literal(row)
        named = row_named(row)
        inferred = row_inferred(row)
        generic = row_generic(row)
        generic_column = row_generic(scaled)
        value_column = row_value(row)
        paired = row_pair(row, scaled)
        array_column = row_array_call(row)
        tuple_column = row_tuple((; row, scale=5.0))
        inlined = row_inline(row)
        lazy = row_lazy(scaled)
        total = literal + named + inferred + generic + generic_column + value_column + paired + array_column + tuple_column + inlined + lazy
        y ~ normal(total, 1.0)
    end
end

@testitem "slic: row-vector dimensions survive function boundaries" tags=[:slic, :shapes, :bridgestan] setup=[RowVectorShapeHelpers] begin
    saved = deepcopy(data)
    source = stan_code(model)
    @test stanc_check(source; warn_pedantic=false).ok
    p = StanBlocks.instantiate(model; nan_on_error=false)
    baseline = LogDensityProblems.logdensity(p, [0.0])
    for a in (-0.4, 0.0, 0.7)
        # Every helper preserves the same five entries; their scales sum to 25.
        expected = 25.0 .* data.x .* a .+ 0.25
        lp, gradient = LogDensityProblems.logdensity_and_gradient(p, [a])
        @test isfinite(lp)
        @test lp - baseline ≈ -0.5 * a^2 - 0.5 * sum(abs2, expected) + 0.5 * 5 * 0.25^2 atol=1e-10
        @test gradient ≈ [-a - sum(expected .* (25.0 .* data.x))] atol=1e-10
        names = BridgeStan.param_names(p.model; include_tp=true, include_gq=true)
        values = BridgeStan.param_constrain(p.model, [a]; include_tp=true, include_gq=true,
            rng=BridgeStan.StanRNG(p.model, 123))
        draw = Dict(zip(names, values))
        @test [draw["total.$i"] for i in eachindex(data.x)] ≈ expected
    end
    @test isequal(data, saved)
end
