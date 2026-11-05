@testitem "slic: submodels return whole plate side outputs" tags=[:slic, :plate, :regression, :stanc] setup=[StanBlocksImports, StanBlocksTestSetup] begin
    using .StanBlocksTestSetup: stanc_compiles, stan_block

    # Unequal cell width and cell count catch a retained single-cell type even
    # when the emitted declaration already has the correct collected shape.
    data = Dict{Symbol,Any}(:x => [-0.5, 0.2, 0.8], :y => 0.4)
    for local_name in (:channels, :value), resultless in (false, true), depth in (1, 2)
        inner = StanBlocks.SlicModel(quote
            spare ~ plate(x) do xi
                z ~ normal(0, scale)
                $local_name = [xi + z, xi - z]
                $(resultless ? nothing : :z)
            end
            alias = $(Symbol(:spare_, local_name))
            result = alias
            return EachCol(result)
        end, Dict{Symbol,Any}(), @__MODULE__)
        if depth == 2
            inner = StanBlocks.SlicModel(quote
                middle ~ $inner(; x=x, scale=scale)
                return middle
            end, Dict{Symbol,Any}(), @__MODULE__)
        end
        model = StanBlocks.SlicModel(quote
            scale ~ exponential(1)
            field ~ $inner(; x=x, scale=scale)
            y ~ normal(field[3][2], 0.4)
        end, data, @__MODULE__)
        code = stan_code(model)
        prefix = depth == 1 ? "field_" : "field_middle_"
        @test occursin("matrix[2, num_elements(x)] " * prefix * "result", stan_block(code, "transformed parameters"))
        @test occursin("col(" * prefix * "result, 3)[2]", stan_block(code, "model"))
        @test stanc_compiles(model)
    end
end

@testitem "slic: plate side collections keep scalar vector and matrix shapes" tags=[:slic, :plate, :regression, :stanc] setup=[StanBlocksImports, StanBlocksTestSetup] begin
    using .StanBlocksTestSetup: stanc_compiles, stan_block

    @slic side_collections(x::vector[n], scale::real) = begin
        spare ~ plate(x) do xi
            z ~ normal(0, scale)
            channels = [xi + z, xi - z]
            covariance = diag_matrix(channels .* channels)
            shifted = xi + z
            fixed = [xi, -xi]
            nothing
        end
        return (; samples=spare_z, vectors=spare_channels,
            matrices=spare_covariance, scalars=spare_shifted, data_vectors=spare_fixed)
    end
    model = @slic (; x=[-0.5, 0.2, 0.8], y=0.4) begin
        scale ~ exponential(1)
        field ~ side_collections(x, scale)
        y ~ normal(field.samples[3] + field.vectors[2, 3] +
            field.matrices[3][2, 2] + field.scalars[3] + field.data_vectors[2, 3], 0.4)
    end
    code = stan_code(model)
    @test occursin("vector[num_elements(x)] field_spare_z;", stan_block(code, "parameters"))
    @test occursin("matrix[2, num_elements(x)] field_spare_channels;", stan_block(code, "transformed parameters"))
    @test occursin("array[num_elements(x)] matrix[2, 2] field_spare_covariance;", stan_block(code, "transformed parameters"))
    @test occursin("vector[num_elements(x)] field_spare_shifted;", stan_block(code, "transformed parameters"))
    @test occursin("matrix[2, num_elements(x)] field_spare_fixed;", stan_block(code, "transformed data"))
    @test stanc_compiles(model)
end

@testitem "slic: nested plate side output density matches collected return" tags=[:slic, :plate, :regression, :bridgestan] setup=[StanBlocksImports, StanBlocksTestSetup] begin
    side = @slic begin
        spare ~ plate(x) do xi
            z ~ normal(0, scale)
            channels = [xi + z, xi - z]
            z
        end
        alias = spare_channels
        return EachCol(alias)
    end
    collected = @slic begin
        spare ~ plate(x) do xi
            z ~ normal(0, scale)
            [xi + z, xi - z]
        end
        return EachCol(spare)
    end
    data = Dict{Symbol,Any}(:x => [-0.5, 0.2, 0.8], :y => 0.4)
    function model(inner)
        StanBlocks.SlicModel(quote
            scale ~ exponential(1)
            field ~ $inner(; x=x, scale=scale)
            y ~ normal(field[3][2], 0.4)
        end, data, @__MODULE__)
    end
    actual, reference = instantiate(model(side)), instantiate(model(collected))
    @test LogDensityProblems.dimension(actual) == LogDensityProblems.dimension(reference) == 4
    for q in ([0.0, -0.2, 0.1, 0.4], [0.3, 0.6, -0.1, -0.5])
        la, ga = LogDensityProblems.logdensity_and_gradient(actual, q)
        lr, gr = LogDensityProblems.logdensity_and_gradient(reference, q)
        @test la ≈ lr atol=1e-12 rtol=1e-12
        @test ga ≈ gr atol=1e-12 rtol=1e-12
    end
end
