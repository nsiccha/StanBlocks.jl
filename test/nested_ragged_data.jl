"""
Vector data nested three deep (`Vector{Vector{Vector{<:Real}}}`, e.g. readings per
subject per endpoint) is a vector of RaggedVectors: it ingests as a
`NestedRaggedVector`, one Stan data tuple `(mem, ends, groups)` with every innermost
vector in `mem`, their inclusive ends in `ends`, and each outer group's last vector
in `groups`. `x[e]` is group `e`'s RaggedVector wherever the model reads it (the
model body, a plate positional, a submodel keyword, a `@deffun` argument),
`x[e][s]` one vector, `length(x)` the outer count, and `x[idx]` selects outer
groups. Such data used to be refused at ingest with "Do not know how to handle
`stan_type(x, …)`" (snag `emit-stan-code-f-2c7ed92c`, follow-up F1).
"""
@testitem "slic: three-level vector data is a vector of RaggedVectors" tags=[:slic, :ragged, :plate, :submodel, :regression, :descriptor, :stanc] setup=[StanBlocksImports, StanBlocksTestSetup] begin
    using .StanBlocksTestSetup: stanc_compiles, stan_block
    import StanBlocks.stan: stan_descriptor
    to_ragged = StanBlocks.stan.to_ragged
    # An empty innermost vector and an empty outer group.
    ys = [[[0.1, 0.2], Float64[]], [[0.3], [0.4, 0.5]], Vector{Float64}[]]
    at = [[[1, 2], Int[]], [[2], [1, 3]], Vector{Int}[]]
    grid = [[1.0, 2.0], [3.0, 4.0, 5.0]]

    # Ingest: integer memory stays integer.
    @test to_ragged(ys) == (; mem=[0.1, 0.2, 0.3, 0.4, 0.5], ends=[2, 2, 3, 5], groups=[2, 4, 4])
    @test to_ragged(at) == (; mem=[1, 2, 2, 1, 3], ends=[2, 2, 3, 5], groups=[2, 4, 4])
    @test to_ragged(at).mem isa Vector{Int}
    @test to_ragged(Vector{Vector{Float64}}[]) == (; mem=Float64[], ends=Int[], groups=Int[])

    # The model body: group, inner vector, lengths, and an integer inner vector
    # indexing other data.
    body = @slic (; ys, at, grid) begin
        mu ~ normal(0, 1)
        n = length(ys)
        k = length(ys[2])
        j = length(ys[2][2])
        ys[2][2] ~ normal(mu + n + k + j + sum(grid[2][at[2][2]]), 1)
    end
    code = stan_code(body)
    data_block = stan_block(code, "data")
    @test occursin("tuple(vector[ys_mem_n], array[ys_ends_n] int, array[ys_groups_n] int) ys;", data_block)
    @test occursin("tuple(array[at_mem_n] int, array[at_ends_n] int, array[at_groups_n] int) at;", data_block)
    td = stan_block(code, "transformed data")
    @test occursin("int n = num_elements_NestedRaggedVector(ys);", td)
    @test occursin("int k = ragged_length(ys.3, 2);", td)
    @test occursin("int j = ragged_length(ys.2, (ragged_start(ys.3, 2) + 1));", td)
    @test occursin("ys.1[ragged_start(ys.2, (ragged_start(ys.3, 2) + 1)):ragged_end(ys.2, (ragged_start(ys.3, 2) + 1))] ~ normal(", stan_block(code, "model"))
    @test stanc_compiles(body)

    # The data file and the descriptor carry both levels.
    data = StanBlocks.JSON.parse(stan_data_json(body))
    @test data["ys"] == Dict("1" => [0.1, 0.2, 0.3, 0.4, 0.5], "2" => [2, 2, 3, 5], "3" => [2, 4, 4])
    @test data["at"] == Dict("1" => [1, 2, 2, 1, 3], "2" => [2, 2, 3, 5], "3" => [2, 4, 4])
    inputs = stan_descriptor(body).inputs
    input = only(i for i in inputs if i.name === :ys)
    @test input.type === :NestedRaggedVector
    @test keys(input.fields) == (:mem, :ends, :groups)
    @test input.fields.ends.value == [2, 2, 3, 5]
    @test input.fields.groups.value == [2, 4, 4]
    @test only(i for i in inputs if i.name === :at).fields.mem.type === :int

    # A plate positional: each cell is one outer group's RaggedVector, read through
    # the outer tables (no per-cell copy of shifted ends).
    positional = @slic (; ys=ys[1:2], at=at[1:2]) begin
        mu ~ normal(0, 1)
        counts ~ plate(ys, at) do y, a
            y[1] ~ normal(mu, 1)
            length(y) + length(a[2])
        end
    end
    code = stan_code(positional)
    @test occursin(r"counts\[(plate_i__pl_\d+)\] = \(ragged_length\(ys\.3, \1\) \+ ragged_length\(at\.2, \(ragged_start\(at\.3, \1\) \+ 1\)\)\);",
        stan_block(code, "transformed data"))
    @test occursin(r"ys\.1\[\s*ragged_start\(ys\.2, ragged_start\(ys\.3, plate_i__pl_\d+\)\):ragged_end\(ys\.2, ragged_start\(ys\.3, plate_i__pl_\d+\)\)\s*\] ~ normal\(mu, 1\);",
        stan_block(code, "model"))
    @test !occursin("nested_ragged_ends", code)
    @test stanc_compiles(positional)
    # The empty outer group is a cell with no vectors.
    @test stanc_compiles(@slic (; ys, at) begin
        mu ~ normal(0, 1)
        counts ~ plate(ys, at) do y, a
            length(y) + length(a)
        end
        q ~ normal(mu + sum(counts), 1)
    end)
    # Positionals of different outer counts are refused, counting outer groups.
    err = try
        stan_code(@slic (; ys, grid) begin
            mu ~ normal(0, 1)
            tot ~ plate(ys, grid) do y, g
                length(y) + sum(g)
            end
        end)
        nothing
    catch e
        sprint(showerror, e)
    end
    @test err !== nothing && occursin("positional `ys` (param `y`) has 3 groups, but positional `grid` (param `g`) has 2 groups", err)

    # Selecting outer groups (`ys[idx]`) is a NestedRaggedVector of those groups.
    selected = @slic (; ys, idx=[2, 1]) begin
        mu ~ normal(0, 1)
        sel = ys[idx]
        q ~ normal(mu + sum(sel[1][2]) + length(sel) + length(sel[2]), 1)
    end
    code = stan_code(selected)
    @test occursin("ragged_select_ends(ys.3, idx)", code)
    @test occursin("ragged_select_index(ys.3, idx)", code)
    @test stanc_compiles(selected)

    # A submodel keyword and a `@deffun` argument; the whole value or one group.
    sub = @slic begin
        b ~ normal(0, 1)
        return b * sum(Z[2][2]) + length(Z)
    end
    composed = @slic (; ys) begin
        r ~ sub(; Z=ys)
        q ~ normal(r, 1.0)
    end
    @test stanc_compiles(composed)
    # A submodel keyword named like the data it reads (`at = at[1]`): the cell
    # sizes read the caller's `at`, not the keyword.
    component = @slic begin
        shift ~ normal(0, 1)
        loc ~ plate(y, at, signal) do ys, picks, xs
            fitted = shift .+ xs[picks]
            ys ~ normal(fitted, 1)
            fitted
        end
        return loc
    end
    keyed = @slic (; grid, ys=ys[1:2], at=at[1:2]) begin
        gain ~ normal(0, 1)
        signal ~ plate(grid) do xs
            gain .* xs
        end
        f1 ~ component(; y=ys[1], at=at[1], signal=signal)
    end
    @test occursin(r"f1_loc_fitted__pl_len\[(f1_plate_i__pl_\d+)\] = \(\s*1 \+\s*\(\s*ragged_end\(at\.2, \(ragged_start\(at\.3, 1\) \+ \(\1 - 1\)\)\)",
        stan_block(stan_code(keyed), "transformed data"))
    @test stanc_compiles(keyed)
    @deffun nested_inner_total(x::NestedRaggedVector, e::int, s::int)::real = sum(x[e][s])
    called = @slic (; ys) begin
        b ~ normal(0, 1)
        q ~ normal(b + nested_inner_total(ys, 2, 2), 1.0)
    end
    @test occursin("nested_inner_total_NestedRaggedVector(ys, 2, 2)", stan_code(called))
    @test stanc_compiles(called)
end

@testitem "slic: a component called with nested ragged groups has the density of separate data" tags=[:slic, :ragged, :plate, :submodel, :regression, :bridgestan] setup=[StanBlocksImports, StanBlocksTestSetup] begin
    using LogDensityProblems
    component = @slic begin
        shift ~ normal(0, 1)
        loc ~ plate(y, at, signal) do ys, picks, xs
            fitted = shift .+ xs[picks]
            ys ~ normal(fitted, 1)
            fitted
        end
        return loc
    end
    nested = @slic begin
        gain ~ normal(0, 1)
        signal ~ plate(grid) do xs
            gain .* xs
        end
        f1 ~ component(; y=ys[1], at=at[1], signal=signal)
        f2 ~ component(; y=ys[2], at=at[2], signal=signal)
    end
    separate = @slic begin
        gain ~ normal(0, 1)
        signal ~ plate(grid) do xs
            gain .* xs
        end
        f1 ~ component(; y=y1, at=at1, signal=signal)
        f2 ~ component(; y=y2, at=at2, signal=signal)
    end
    # One subject without readings in the first group.
    ys = [[[0.1, 0.2], Float64[]], [[0.3], [0.4, 0.5]]]
    at = [[[1, 2], Int[]], [[2], [1, 3]]]
    grid = [[1.0, 2.0], [3.0, 4.0, 5.0]]
    pn = instantiate(stan_model(nested(; grid, ys, at)))
    ps = instantiate(stan_model(separate(; grid, y1=ys[1], at1=at[1], y2=ys[2], at2=at[2])))
    @test LogDensityProblems.dimension(pn) == LogDensityProblems.dimension(ps) == 3
    for x in ([0.3, -0.2, 0.5], [1.1, 0.4, -0.7])
        ln, gn = LogDensityProblems.logdensity_and_gradient(pn, x)
        ls, gs = LogDensityProblems.logdensity_and_gradient(ps, x)
        @test ln == ls
        @test gn == gs
    end

    # Selected groups observed through a plate: the density of the plain-Julia sum.
    selected = @slic (; ys, idx=[2, 1]) begin
        mu ~ normal(0, 1)
        tot ~ plate(ys[idx]) do y
            y[1] ~ normal(mu, 1)
            length(y)
        end
    end
    reference(mu) = -mu^2 / 2 - sum(abs2, ys[2][1] .- mu) / 2 - sum(abs2, ys[1][1] .- mu) / 2
    p = instantiate(stan_model(selected))
    l0, _ = LogDensityProblems.logdensity_and_gradient(p, [0.0])
    l1, g1 = LogDensityProblems.logdensity_and_gradient(p, [0.7])
    @test l1 - l0 ≈ reference(0.7) - reference(0.0) atol = 1e-12
    @test only(g1) ≈ -0.7 + sum(ys[2][1] .- 0.7) + sum(ys[1][1] .- 0.7) atol = 1e-12
end
