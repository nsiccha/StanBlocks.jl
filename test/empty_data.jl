using TestItemRunner

"""
An empty index computation can produce a bottom-eltype vector rather than Int[].
It must ingest as the same integer array, including through views, tuple data,
model rebinding and observations, without changing explicitly typed empties.
"""
@testitem "slic: bottom-eltype empty data preserves the integer-array contract" tags=[:slic, :regression] begin
    using StanBlocks

    function empty_model(x)
        @slic (; x, y=[0.2, -0.4]) begin
            mu ~ normal(0.0, 1.0)
            y ~ normal(mu + sum(x), 1.0)
            mu
        end
    end

    reference = stan_model(empty_model(Int[]))
    for x in (Union{}[], view(Union{}[], :))
        model = stan_model(empty_model(x))
        @test stan_code(model) == stan_code(reference)
        @test StanBlocks.stan_data(model) == StanBlocks.stan_data(reference)
        @test StanBlocks.center_type(model.vars[:x]) === StanBlocks.types.int
        @test model.vars[:x].type.info.value === x

        # Rebinding an already traced model uses the same ingest boundary.
        rebound = reference(; x)
        @test stan_code(rebound) == stan_code(reference)
        @test StanBlocks.stan_data(rebound) == StanBlocks.stan_data(reference)
    end

    for (x, expected) in ((Float64[], StanBlocks.types.vector),
                          (Int[], StanBlocks.types.int),
                          (Vector{Float64}[], StanBlocks.RaggedVector))
        @test StanBlocks.center_type(StanBlocks.stan_type(:x, x; qual=:data)) === expected
    end
    # Refused: a nonempty bottom-eltype array cannot contain defined data.
    @test_throws ArgumentError StanBlocks.stan_type(:x, Vector{Union{}}(undef, 1); qual=:data)

    function tuple_model(x)
        @slic (; payload=(; x, scale=1.0), y=[0.2]) begin
            y ~ normal(sum(payload.x), payload.scale)
        end
    end
    tuple_reference = stan_model(tuple_model(Int[]))
    tuple_bottom = stan_model(tuple_model(Union{}[]))
    @test stan_code(tuple_bottom) == stan_code(tuple_reference)
    @test StanBlocks.stan_data(tuple_bottom) == StanBlocks.stan_data(tuple_reference)

    function observed_model(y)
        @slic (; y) begin
            mu ~ normal(0.0, 1.0)
            y ~ normal(mu, 1.0)
            mu
        end
    end
    @test stan_code(observed_model(Union{}[])) == stan_code(observed_model(Int[]))

    # Fully observed completion still needs an empty integer index array.
    function completion_model(ii_mis)
        @slic (; values=[0.2, -0.4], missing_values=Float64[],
                 ii_obs=[1, 2], ii_mis, y=[0.3]) begin
            completed = merge_missing(values, missing_values, ii_obs, ii_mis)
            y ~ normal(sum(completed), 1.0)
        end
    end
    completion_reference = stan_model(completion_model(Int[]))
    completion_bottom = stan_model(completion_model(Union{}[]))
    @test stan_code(completion_bottom) == stan_code(completion_reference)
    @test StanBlocks.stan_data(completion_bottom) == StanBlocks.stan_data(completion_reference)
end

"""
Bottom-eltype empty input and Int[] emit the same Stan program, which the
Stan compiler must accept. Numeric, integer and ragged empty controls stay valid.
"""
@testitem "slic: empty data programs compile with stanc" tags=[:slic, :regression, :stanc] begin
    using StanBlocks

    for x in (Union{}[], Float64[], Int[], Vector{Float64}[])
        model = @slic (; x, y=[0.2]) begin
            mu ~ normal(0.0, 1.0)
            y ~ normal(mu + length(x), 1.0)
        end
        result = stanc_check(stan_code(model); warn_pedantic=false)
        result.ok || @error "stanc rejected empty input" input_type=typeof(x) output=result.output
        @test result.ok
    end

    model = @slic (; values=[0.2, -0.4], missing_values=Float64[],
                     ii_obs=[1, 2], ii_mis=Union{}[], y=[0.3]) begin
        completed = merge_missing(values, missing_values, ii_obs, ii_mis)
        y ~ normal(sum(completed), 1.0)
    end
    result = stanc_check(stan_code(model); warn_pedantic=false)
    result.ok || @error "stanc rejected fully observed completion" output=result.output
    @test result.ok
end
