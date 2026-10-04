using TestItemRunner

@testsnippet CallerShapeHelpers begin
    using StanBlocks
    import LogDensityProblems, BridgeStan
    import StanBlocks.stan: instantiate

    @deffun begin
        shift_values(x::vector[n], displacement::real) = x + displacement
        scale_values(x::vector[n], scale::real) = x * scale
        explicit_values(x::vector[n], scale::real)::vector[n] = x * scale
        nested_values(x::vector[n], scale::real) = scale_values(shift_values(x, scale), scale)
        local_shadow(z::vector[n]) = begin
            x_n = 2
            z + x_n
        end
        pair_values(x::vector[n]) = (; first=x + 1.0, second=x * 2.0)
        consume_pair(x::ntup) = scale_values(x.first, 2.0)
        nested_pair(x::vector[n]) = begin
            pair = pair_values(x)
            consume_pair(pair)
        end
    end

    data = (; x=[[-0.4, 0.2], Float64[], [0.7]],
        y=[[0.1, -0.2], Float64[], [0.3]])
    model = @slic data begin
        a ~ normal(0.0, 0.7)
        b ~ normal(0.0, 0.4)
        loc ~ plate(x; outer=length(x)) do xx
            shifted = shift_values(xx, a)
            scale_values(shifted, b)
        end
        y ~ normal(loc, 0.8)
    end
end

@testitem "slic: caller shapes survive ragged helper chains" tags=[:slic, :plate, :stanc] setup=[CallerShapeHelpers] begin
    saved = deepcopy(data)
    traced = stan_model(model)
    source = stan_code(traced)
    @test stanc_check(source; warn_pedantic=false).ok
    @test !occursin("getfield(", source)
    @test !occursin("CallSizeRef", source)
    @test !occursin("_arg", source)
    outputs = collect(stan_descriptor(traced).outputs)
    for stem in ("loc", "loc_shifted")
        carrier = only(filter(o -> startswith(string(o.name), stem * "__pl_mem_"), outputs))
        @test carrier.segments == [2, 2, 3]
    end
    @test isequal(data, saved)

    for xs in ([Float64[], Float64[], Float64[]],
               [Float64[], [0.2, -0.6, 0.4], [0.1]])
        ys = [zeros(length(x)) for x in xs]
        rebound = traced(; x=xs, y=ys)
        @test stanc_check(stan_code(rebound); warn_pedantic=false).ok
        rebound_outputs = collect(stan_descriptor(rebound).outputs)
        ends = cumsum(length.(xs))
        for stem in ("loc", "loc_shifted")
            carrier = only(filter(o -> startswith(string(o.name), stem * "__pl_mem_"), rebound_outputs))
            @test carrier.segments == ends
        end
    end
end

@testitem "slic: caller shapes survive locals and composite helper arguments" tags=[:slic, :shapes, :stanc] setup=[CallerShapeHelpers] begin
    dense = @slic (; x=[0.1, 0.2, 0.3], y=zeros(3)) begin
        z = local_shadow(x)
        y ~ normal(z, 1.0)
    end
    dense_source = stan_code(dense)
    @test stanc_check(dense_source; warn_pedantic=false).ok
    @test occursin("vector[x_n] z", dense_source)

    for helper in (explicit_values, nested_values)
        m = @slic (; data..., helper) begin
            a ~ normal(0.0, 0.7)
            loc ~ plate(x; outer=length(x)) do xx
                helper(xx, a)
            end
            y ~ normal(loc, 0.8)
        end
        @test stanc_check(stan_code(m); warn_pedantic=false).ok
        @test !occursin("CallSizeRef", stan_code(m))
    end
    composite = @slic data begin
        loc ~ plate(x; outer=length(x)) do xx
            nested_pair(xx)
        end
        y ~ normal(loc, 0.8)
    end
    @test stanc_check(stan_code(composite); warn_pedantic=false).ok
    @test !occursin("_arg", stan_code(composite))
end

@testitem "slic: ragged helper chains preserve native density and values" tags=[:slic, :plate, :bridgestan] setup=[CallerShapeHelpers] begin
    p = instantiate(stan_model(model))
    @test LogDensityProblems.dimension(p) == 2
    flat_x, flat_y = vcat(data.x...), vcat(data.y...)
    oracle(a, b) = -0.5 * (a / 0.7)^2 - 0.5 * (b / 0.4)^2 -
        0.5 * sum(((flat_y .- (flat_x .+ a) .* b) ./ 0.8).^2)
    baseline = LogDensityProblems.logdensity(p, [0.0, 0.0])
    for (a, b) in ((-0.3, 0.5), (0.2, -0.4), (0.0, 0.0))
        lp, grad = LogDensityProblems.logdensity_and_gradient(p, [a, b])
        residual = flat_y .- (flat_x .+ a) .* b
        expected_grad = [-a / 0.7^2 + sum(b .* residual) / 0.8^2,
            -b / 0.4^2 + sum((flat_x .+ a) .* residual) / 0.8^2]
        @test lp - baseline ≈ oracle(a, b) - oracle(0.0, 0.0) atol=1e-10
        @test grad ≈ expected_grad atol=1e-10

        names = BridgeStan.param_names(p.model; include_tp=true, include_gq=true)
        values = BridgeStan.param_constrain(p.model, [a, b]; include_tp=true,
            include_gq=true, rng=BridgeStan.StanRNG(p.model, 1234))
        draw = Dict(zip(names, values))
        for i in eachindex(flat_x)
            @test draw["loc_shifted__pl_mem_1.$i"] ≈ flat_x[i] + a
            @test draw["loc__pl_mem_1.$i"] ≈ (flat_x[i] + a) * b
        end
    end
end
