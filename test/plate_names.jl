"""
A plate's emitted Stan names follow from each value's own name, so a consumer
outside Julia can rely on them (todo `19ijy0g`). In a `@plate for` loop an
array filled at the loop index (`mu[i] = …`) is declared under its own name,
also when its cells have different lengths: its flat memory is `mu`, with
transformed-data group lengths `mu__pl_len` and inclusive ends `mu__pl_end`. A
per-iteration local is stored as `loop_<i>_<name>`, named after the loop
variable. Neither name depends on the other statements of the loop or on how
many plates came before it; both used to (`mu__pl_mem_1`, locals prefixed by
the alphabetically first array).
"""
@testitem "slic: plate names do not depend on unrelated statements" tags=[:slic, :plate, :ragged, :descriptor, :regression, :stanc] setup=[StanBlocksImports, StanBlocksTestSetup] begin
    using .StanBlocksTestSetup: stanc_compiles, stan_block
    data = (; b0 = 0.1, omega = 0.3,
        y = [[0.1, 0.2, 0.15, 0.3], [0.2, 0.1]],
        x = [[0.0, 1.0, 2.0, 0.0, 0.5, 0.0], [1.0, 0.0, 0.0]],
        idx = [[2, 3, 4, 6], [1, 3]])
    base = @slic data begin
        b ~ normal(b0, 1); a ~ normal(0, 1); sigma ~ exponential(1)
        @plate for i in eachindex(y)
            W = cumulative_sum(x[i])
            effect[i] = a .* W[idx[i]]
            mu[i] = b .+ effect[i]
            scale = sqrt.(sigma ^ 2 .+ (omega .* effect[i]) .^ 2)
            y[i] ~ normal(mu[i], scale)
        end
    end
    # One more output, as the loop's last statement.
    extended = @slic data begin
        b ~ normal(b0, 1); a ~ normal(0, 1); sigma ~ exponential(1)
        @plate for i in eachindex(y)
            W = cumulative_sum(x[i])
            effect[i] = a .* W[idx[i]]
            mu[i] = b .+ effect[i]
            scale = sqrt.(sigma ^ 2 .+ (omega .* effect[i]) .^ 2)
            y[i] ~ normal(mu[i], scale)
            amp[i] = 2.0 .* effect[i]
        end
    end
    # The same statements in another order, after an unrelated plate.
    reordered = @slic data begin
        b ~ normal(b0, 1); a ~ normal(0, 1); sigma ~ exponential(1)
        u ~ plate(; outer = 2) do g
            t ~ std_normal()
            t
        end
        @plate for i in eachindex(y)
            W = cumulative_sum(x[i])
            effect[i] = a .* W[idx[i]]
            scale = sqrt.(sigma ^ 2 .+ (omega .* effect[i]) .^ 2)
            mu[i] = b .+ effect[i]
            y[i] ~ normal(mu[i], scale)
        end
    end
    described(m) = Dict(o.name => (o.kind, o.segments) for o in stan_descriptor(m).outputs)
    expected = Dict(
        :effect => (:transformed_parameter, [4, 6]),
        :mu => (:transformed_parameter, [4, 6]),
        :loop_i_scale => (:transformed_parameter, [4, 6]),
        :y_gen => (:generated_quantity, [4, 6]),
        :y_likelihood => (:generated_quantity, [4, 6]),
    )
    for m in (base, extended, reordered)
        @test stanc_compiles(m)
        d = described(m)
        @test all(d[k] == v for (k, v) in expected)
    end
    @test setdiff(keys(described(extended)), keys(described(base))) == Set([:amp])
    @test setdiff(keys(described(reordered)), keys(described(base))) == Set([:u, :u_t])

    tp = stan_block(stan_code(base), "transformed parameters")
    @test occursin("vector[sum(effect__pl_len)] effect;", tp)
    @test occursin("vector[sum(mu__pl_len)] mu;", tp)
    @test occursin("mu[ragged_start(mu__pl_end, i):ragged_end(mu__pl_end, i)] = (b + effect[", tp)
    # A data-only local is transformed data; its name follows the same rule.
    @test occursin("vector[sum(loop_i_W__pl_len)] loop_i_W;", stan_block(stan_code(base), "transformed data"))

    # Rebinding the data re-derives the layout under the same names.
    rebound = StanBlocks.stan_model(base)(; y = [[0.3], [0.1, 0.2]], x = [[1.0], [0.5, 0.5]], idx = [[1], [1, 2]])
    @test only(o.segments for o in stan_descriptor(rebound).outputs if o.name === :mu) == [1, 3]

    # After the loop, `mu` reads as the ragged value: `mu[g]` is group `g`.
    downstream = @slic data begin
        b ~ normal(b0, 1); a ~ normal(0, 1); sigma ~ exponential(1)
        @plate for i in eachindex(y)
            W = cumulative_sum(x[i])
            mu[i] = b .+ a .* W[idx[i]]
            y[i] ~ normal(mu[i], sigma)
        end
        first_total = sum(mu[1])
    end
    @test stanc_compiles(downstream)
    @test occursin("sum(mu[ragged_start(mu__pl_end, 1):ragged_end(mu__pl_end, 1)])", stan_code(downstream))
end

"""
A do-form plate names its carriers after its result: the collected result `rv`
(also as ragged flat memory) and `rv_<name>` for each cell binding. Those names
do not depend on the cell's other statements either.
"""
@testitem "slic: do-form plate carriers are named after the result" tags=[:slic, :plate, :ragged, :descriptor, :regression, :stanc] setup=[StanBlocksImports, StanBlocksTestSetup] begin
    using .StanBlocksTestSetup: stanc_compiles
    data = (; y = [[0.1, 0.2, 0.15], [0.2, 0.1]], x = [[0.0, 1.0, 2.0], [1.0, 0.0]], G = 2)
    base = @slic data begin
        a ~ normal(0, 1); sigma ~ exponential(1)
        res ~ plate(y, x; outer = (G,)) do yi, xi
            effect = a .* xi
            yi ~ normal(effect, sigma)
            effect
        end
    end
    extended = @slic data begin
        a ~ normal(0, 1); sigma ~ exponential(1)
        res ~ plate(y, x; outer = (G,)) do yi, xi
            effect = a .* xi
            yi ~ normal(effect, sigma)
            twice = 2.0 .* effect
            effect
        end
    end
    names(m) = Set(o.name for o in stan_descriptor(m).outputs)
    @test stanc_compiles(base) && stanc_compiles(extended)
    @test names(base) == Set([:a, :sigma, :res_effect, :res, :y_gen, :y_likelihood])
    @test setdiff(names(extended), names(base)) == Set([:res_twice])
end

"""
The compiler-made names of a ragged carrier never take a bound name: one that
is taken gets a numbered suffix (`_plate_fresh_name!`). An array the loop fills
must not be bound before it.
"""
@testitem "slic: a ragged plate carrier's lengths and ends avoid bound names" tags=[:slic, :plate, :ragged, :regression, :stanc] setup=[StanBlocksImports, StanBlocksTestSetup] begin
    using .StanBlocksTestSetup: stanc_compiles
    data = (; y = [[0.1, 0.2], [0.3]], x = [[1.0, 2.0], [3.0]], mu__pl_len = 7)
    m = @slic data begin
        a ~ normal(0, 1)
        @plate for i in eachindex(y)
            mu[i] = a .* x[i]
            y[i] ~ normal(mu[i], 1)
        end
    end
    @test stanc_compiles(m)
    @test occursin("vector[sum(mu__pl_len_2)] mu;", stan_code(m))
    # refused: an indexed LHS declares a new model-scope array, and a model
    # never rebinds a name, a data input included (`@plate for` contract,
    # stanblocks-use §35). It used to fail with an untyped-`tracetype` error.
    @test_throws "writes to the data input `mu`" StanBlocks.stan_code(@slic (; y = [[0.1], [0.2]], mu = 1.0) begin
        @plate for i in eachindex(y)
            mu[i] = mu .+ 0.0 .* y[i]
            y[i] ~ normal(mu[i], 1)
        end
    end)
end
