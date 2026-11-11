"""
Ragged broadcasting slices ragged carriers and shares dense arguments. A dense
array of totals is not a scalar total for one multinomial group. Todo 1m9t9qt
authorizes a precise diagnostic and explicit-plate guidance; ambiguous implicit
indexing would require a separate design decision.
"""
@testitem "slic: ragged composition totals name the argument mismatch" tags=[:slic, :regression, :ragged, :plate, :stanc, :bridgestan] setup=[StanBlocksImports, StanBlocksTestSetup] begin
    using .StanBlocksTestSetup: stanc_compiles
    ys = [[2, 3, 1], [4, 0]]
    ps = [[0.2, 0.5, 0.3], [0.6, 0.4]]
    ns = [6, 4]
    data = Dict{Symbol,Any}(:ys=>ys, :ps=>ps, :ns=>ns, :G=>2)
    for family in (:multinomial, :dirichlet_multinomial)
        @testset "$family" begin
            bad = StanBlocks.SlicModel(:(begin ys ~ $family(ps, ns) end), data, @__MODULE__)
            # Diagnostic boundary accepted by 1m9t9qt / user GO 00u5h5g:
            # dense per-group values need explicit indexing, not a guessed axis.
            message = try
                stan_code(bad)
                ""
            catch err
                sprint(showerror, err)
            end
            @test occursin("family `$family` needs a scalar integer total", message)
            @test occursin("array[] int", message)
            @test occursin("dense arguments are passed whole", message)
            @test occursin("@plate for k in 1:G", message)
            @test !occursin("has no SIZED predictive companion", message)

            # Positive control: both existing sized companions draw every
            # group at its own total when the caller indexes that total.
            parameters = family === :multinomial ?
                :(softmax(phi .* log(ps[k]))) : :(phi .* ps[k])
            body = quote
                phi ~ lognormal(0.0, 1.0)
                @plate for k in 1:G
                    ys[k] ~ $family($parameters, ns[k])
                end
            end
            good = StanBlocks.SlicModel(body, data, @__MODULE__)
            @test stanc_compiles(good)
            p = instantiate(good)
            names = BridgeStan.param_names(p.model; include_tp=true, include_gq=true)
            draw = BridgeStan.param_constrain(p.model, [0.0]; include_tp=true, include_gq=true,
                rng=BridgeStan.StanRNG(p.model, 11))
            generated = draw[findall(name -> startswith(name, "ys_gen"), names)]
            @test length(generated) == 5
            @test all(x -> isinteger(x) && x >= 0, generated)
            @test [sum(generated[1:3]), sum(generated[4:5])] == ns
            @test ys == [[2, 3, 1], [4, 0]] && ns == [6, 4]
        end
    end
end
