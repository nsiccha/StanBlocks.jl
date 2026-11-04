@testitem "slic: plate submodel keywords retain their input names" tags=[:slic, :plate, :regression, :stanc] setup=[StanBlocksImports, StanBlocksTestSetup] begin
    using .StanBlocksTestSetup: stanc_compiles, stan_block

    sub = @slic begin
        a ~ normal(0, s)
        return a + x
    end
    data = Dict{Symbol,Any}(:x => [0.1, 0.2], :y => [0.3, 0.4])
    function model(param, call)
        StanBlocks.SlicModel(quote
            s ~ exponential(1)
            loc ~ plate(x, y) do $param, yy
                cell ~ $call
                yy ~ normal(cell, 1)
                cell
            end
        end, data, @__MODULE__)
    end

    control = model(:cell_x, :($sub(; x=cell_x, s=s)))
    code = stan_code(control)
    @test occursin("loc_cell_a;", stan_block(code, "parameters"))
    @test occursin("x[plate_i__pl_1]", code)
    @test stanc_compiles(control)
    for call in (:($sub(; x=x, s=s)), :($sub(x=x, s=s)), :($sub(; x, s)))
        same_name = model(:x, call)
        @test stan_code(same_name) == code
        @test stanc_compiles(same_name)
    end
end

@testitem "slic: plate local renaming preserves keyword labels and bounds" tags=[:slic, :plate, :regression, :stanc] setup=[StanBlocksImports, StanBlocksTestSetup] begin
    using .StanBlocksTestSetup: stanc_compiles, stan_block

    sub = @slic begin
        a ~ normal(0, s)
        return a + x
    end
    alternate = @slic begin
        a ~ normal(0, scale)
        return a + value
    end
    data = Dict{Symbol,Any}(:x => [0.1, 0.2], :y => [0.3, 0.4])
    function model(call)
        StanBlocks.SlicModel(quote
            loc ~ plate(x, y) do xi, yy
                s ~ exponential(1)
                cell ~ $call
                yy ~ normal(cell, 1)
                cell
            end
        end, data, @__MODULE__)
    end
    control = stan_code(model(:($alternate(; value=xi, scale=s))))
    @test occursin("loc_s;", stan_block(control, "parameters"))
    for call in (:($sub(; x=xi, s=s)), :($sub(; x=xi, s)))
        local_input = model(call)
        @test stan_code(local_input) == control
        @test stanc_compiles(local_input)
    end

    # A bound's label is syntax too and must survive do-parameter renaming.
    function bounded(param)
        StanBlocks.SlicModel(quote
            loc ~ plate(bounds, y) do $param, yy
                a ~ normal(0, 1; lower=$param)
                yy ~ normal(a, 1)
                a
            end
        end, Dict{Symbol,Any}(:bounds => [-0.5, 0.0], :y => [0.3, 0.4]), @__MODULE__)
    end
    code = stan_code(bounded(:lo))
    @test occursin("lower=", stan_block(code, "parameters"))
    @test stan_code(bounded(:lower)) == code
    @test stanc_compiles(bounded(:lower))
end

@testitem "slic: plate named tuple fields retain their labels" tags=[:slic, :plate, :regression, :stanc] setup=[StanBlocksImports, StanBlocksTestSetup] begin
    using .StanBlocksTestSetup: stanc_compiles

    function model(param, fields)
        StanBlocks.SlicModel(quote
            loc ~ plate(x, y) do $param, yy
                yy ~ normal(($fields).x + ($fields).s, 1)
                ($fields).x
            end
        end, Dict{Symbol,Any}(:x => [0.1, 0.2], :y => [0.3, 0.4], :s => 0.5), @__MODULE__)
    end
    code = stan_code(model(:cell_x, :((; x=cell_x, s=s))))
    for fields in (:((; x=x, s=s)), :((; x, s)))
        same_name = model(:x, fields)
        @test stan_code(same_name) == code
        @test stanc_compiles(same_name)
    end
end
