"""
Verify `slic: descriptor describes tuple inputs and outputs by their components`.

Snag stan-descriptor-9bbea147 (from a downstream regression package): a ragged
data input is emitted as `tuple(vector[y_mem_n], array[y_ends_n] int) y`, but
`stan_descriptor` reported its component sizes `y_mem_n`/`y_ends_n` as required
inputs and nothing linked them to `y`, so a consumer encoding CmdStan JSON had
to reproduce the names `stan_type` gives them. Tables and `NamedTuple`/`Tuple`
data had the same gap, and a tuple output described no components either. Each
tuple entry now carries `fields`, one row per component in Stan tuple order,
and every component size is `derived` (decision `18oq2xa`, option 1).
"""
@testitem "slic: descriptor describes tuple inputs and outputs by their components" tags=[:slic, :ragged, :descriptor, :regression] setup=[StanBlocksImports, DescriptorJSON] begin
    input(d, name) = only(i for i in d.inputs if i.name === name)
    output(d, name) = only(o for o in d.outputs if o.name === name)
    StanJSON = StanBlocks.stan.JSON

    ragged = @slic (; y = [[0.1, 0.2], [0.3]], x = [1.0, 2.0]) begin
        mu ~ std_normal()
        y ~ normal(mu, 1.0)
        x ~ normal(mu, 1.0)
    end
    d = stan_descriptor(ragged)
    @test StanBlocks.required_inputs(d) == (:y, :x)
    @test all(input(d, n).derived for n in (:y_mem_n, :y_ends_n, :x_n))
    y = input(d, :y)
    # The outer entry keeps its meaning: the tuple itself, no array dims.
    @test y.type == :RaggedVector && y.size == ()
    @test keys(y.fields) == (:mem, :ends)
    @test y.fields.mem == (type = :vector, size = (:y_mem_n,), value = [0.1, 0.2, 0.3], fields = (;))
    @test y.fields.ends == (type = :int, size = (:y_ends_n,), value = [2, 3], fields = (;))
    @test input(d, :x).fields == (;)
    @test occursin("tuple(vector[y_mem_n], array[y_ends_n] int) y;", stan_code(ragged))

    # An integer ragged vector differs from a real one only in its `mem` row.
    ints = stan_descriptor(@slic (; g = [[1, 2], [2]], z = [0.1, 0.2, 0.3]) begin
        mu ~ std_normal()
        z ~ normal(mu + g.mem[1], 1.0)
    end)
    @test StanBlocks.required_inputs(ints) == (:z, :g)
    @test input(ints, :g).type == :RaggedVector
    @test input(ints, :g).fields.mem == (type = :int, size = (:g_mem_n,), value = [1, 2, 2], fields = (;))

    # A table: one row per column, all sized by the shared row count.
    table = stan_descriptor(@slic (; df = [(age = 20.0, grp = 1), (age = 45.0, grp = 2), (age = 31.0, grp = 1)]) begin
        beta ~ std_normal()
        df.age ~ normal(beta * df.grp, 1.0)
    end)
    @test StanBlocks.required_inputs(table) == (:df,)
    @test input(table, :df_nrow).derived
    df = input(table, :df)
    @test keys(df.fields) == (:age, :grp)
    @test df.fields.age.size == df.fields.grp.size == (:df_nrow,)
    @test (df.fields.age.type, df.fields.grp.type) == (:vector, :int)
    @test df.fields.grp.value == [1, 2, 1]

    # `NamedTuple` data nests: a ragged field has components of its own. Its
    # value is the ingested `(mem, ends)` pair, which is also what reaches the
    # JSON; the caller's vector-of-vectors used to reach `prepare_for_stan` raw
    # and could not be encoded at all.
    nested = @slic (; tab = (; a = [1.0, 2.0], r = [[1.0], [2.0, 3.0]])) begin
        mu ~ std_normal()
        tab.a ~ normal(mu, 1.0)
    end
    nd = stan_descriptor(nested)
    @test StanBlocks.required_inputs(nd) == (:tab,)
    tab = input(nd, :tab)
    @test tab.type == :ntup && keys(tab.fields) == (:a, :r)
    @test tab.fields.a == (type = :vector, size = (:tab_a_n,), value = [1.0, 2.0], fields = (;))
    @test tab.fields.r.type == :RaggedVector
    @test tab.fields.r.value == (mem = [1.0, 2.0, 3.0], ends = [1, 3])
    @test tab.fields.r.fields.mem == (type = :vector, size = (:tab_r_mem_n,), value = [1.0, 2.0, 3.0], fields = (;))
    @test tab.fields.r.fields.ends == (type = :int, size = (:tab_r_ends_n,), value = [1, 3], fields = (;))
    @test StanJSON.parse(StanBlocks.stan.bridgestan_data(StanBlocks.stan.stan_data(nested)))["tab"] ==
        Dict("1" => [1.0, 2.0], "2" => Dict("1" => [1.0, 2.0, 3.0], "2" => [1, 3]))

    # A positional `Tuple`'s components are keyed by position.
    positional = stan_descriptor(@slic (; t = (1.0, [1.0, 2.0])) begin
        mu ~ std_normal()
        t[2] ~ normal(mu + t[1], 1.0)
    end)
    @test StanBlocks.required_inputs(positional) == (:t,)
    @test keys(input(positional, :t).fields) == (Symbol("1"), Symbol("2"))
    @test input(positional, :t).fields[Symbol("2")].size == (:t_2_n,)

    # A record a `@deffun` returns: components with resolved `dims`.
    @deffun tuple_record(a::real, v::vector[n]) = (; s = a, w = v)
    record = stan_descriptor(@slic (; z = [0.1, 0.2, 0.4]) begin
        mu ~ std_normal()
        rec = tuple_record(mu, z)
        z ~ normal(rec.s, 1.0)
    end)
    rec = output(record, :rec)
    @test rec.type == :ntup && rec.dims == ()
    @test rec.fields == (
        s = (type = :real, size = (), dims = (), fields = (;)),
        w = (type = :vector, size = (:z_n,), dims = (3,), fields = (;)),
    )
    @test output(record, :mu).fields == (;)

    # Re-binding reports the rebound components, as it does the rebound sizes.
    rebound = stan_descriptor(stan_model(ragged)(; y = [[0.1, 0.2, 0.5], [0.3], [0.7]]))
    @test StanBlocks.required_inputs(rebound) == (:y, :x)
    @test input(rebound, :y).fields.mem.value == [0.1, 0.2, 0.5, 0.3, 0.7]
    @test input(rebound, :y).fields.ends.value == [3, 4, 5]
    @test (input(rebound, :y_mem_n).value, input(rebound, :y_ends_n).value) == (5, 3)

    # A consumer that knows only the descriptor builds the JSON StanBlocks
    # writes: each required input, a tuple as an object keyed by component
    # position, and each derived size read from the container it sizes.
    for m in (ragged, nested)
        @test StanJSON.parse(StanJSON.json(descriptor_json(stan_descriptor(m)))) ==
            StanJSON.parse(StanBlocks.stan.bridgestan_data(StanBlocks.stan.stan_data(m)))
    end
end

"""
Verify `slic: a ragged input encoded from its descriptor fits through BridgeStan`.

The CmdStan JSON a consumer builds from `stan_descriptor` alone (required
inputs, tuple `fields`, derived sizes) loads into the compiled model and gives
the log density StanBlocks' own data gives through the same library (snag
stan-descriptor-9bbea147).
"""
@testitem "slic: a ragged input encoded from its descriptor fits through BridgeStan" tags=[:slic, :ragged, :descriptor, :bridgestan] setup=[StanBlocksImports, DescriptorJSON] begin
    m = @slic (; y = [[0.1, 0.2], [0.3]], x = [1.0, 2.0]) begin
        mu ~ std_normal()
        y ~ normal(mu, 1.0)
        x ~ normal(mu, 1.0)
    end
    library = stan_library(stan_code(m))
    from_descriptor = BridgeStan.StanModel(library,
        StanBlocks.stan.JSON.json(descriptor_json(stan_descriptor(m))))
    from_stanblocks = BridgeStan.StanModel(library,
        StanBlocks.stan.bridgestan_data(StanBlocks.stan.stan_data(m)))
    for theta in ([0.0], [0.7], [-1.3])
        lp = BridgeStan.log_density(from_descriptor, theta)
        @test isfinite(lp)
        @test lp == BridgeStan.log_density(from_stanblocks, theta)
    end
end

# The CmdStan JSON for a descriptor's model, built from the descriptor alone: no
# input name is parsed for a `_n`/`_mem_n`/`_ends_n`/`_nrow` suffix.
@testsnippet DescriptorJSON begin
    encoded(entry) = isempty(entry.fields) ? entry.value :
        Dict(string(i) => encoded(f) for (i, f) in enumerate(values(entry.fields)))
    record_sizes!(sizes, entry) = begin
        for (axis, s) in enumerate(entry.size)
            s isa Symbol && (sizes[s] = size(entry.value, axis))
        end
        foreach(f -> record_sizes!(sizes, f), values(entry.fields))
        sizes
    end
    descriptor_json(d) = begin
        required = [i for i in d.inputs if i.name in StanBlocks.required_inputs(d)]
        sizes = Dict{Symbol,Int}()
        foreach(i -> record_sizes!(sizes, i), required)
        merge(Dict(string(i.name) => encoded(i) for i in required),
            Dict(string(k) => v for (k, v) in sizes))
    end
end
