"""
`stan_interface(model)` describes a traced program for a consumer that runs its
Stan code without Julia (todo `0gr4e02`): each data variable with its encoding
and whether it is supplied or computed, the parameter order, and each output
with its block, shape and the source of a ragged output's group ends.
`stan_data_json(model)` is the data file the program reads. A consumer that
knows only the interface and its own raw values writes that same file.
"""
@testitem "slic: stan_interface describes data, parameters and outputs" tags=[:slic, :descriptor, :ragged, :interface, :regression] setup=[StanBlocksImports, InterfaceJSON] begin
    raw = Dict{String,Any}("b0" => 0.1, "omega" => 0.3,
        "y" => [[0.1, 0.2, 0.15, 0.3], [0.2, 0.1]],
        "x" => [[0.0, 1.0, 2.0, 0.0, 0.5, 0.0], [1.0, 0.0, 0.0]],
        "idx" => [[2, 3, 4, 6], [1, 3]], "K" => [2, 3],
        "M" => [1.0 2.0 3.0; 4.0 5.0 6.0],
        "tab" => Dict{String,Any}("a" => [1.0, 2.0], "r" => [[1.0], [2.0, 3.0]]))
    m = @slic (; b0 = 0.1, omega = 0.3, y = raw["y"], x = raw["x"], idx = raw["idx"],
            K = raw["K"], M = raw["M"], tab = (; a = [1.0, 2.0], r = [[1.0], [2.0, 3.0]])) begin
        b ~ normal(b0, 1); a ~ normal(0, 1); sigma ~ exponential(1)
        @plate for i in eachindex(y)
            W = cumulative_sum(x[i])
            effect[i] = a .* W[idx[i]]
            mu[i] = b .+ effect[i]
            scale = sqrt.(sigma ^ 2 .+ (omega .* effect[i]) .^ 2)
            y[i] ~ normal(mu[i], scale)
        end
        r ~ plate(; outer = length(K)) do g
            z::vector[K[g]] ~ std_normal()
            z
        end
        M ~ normal(b, sigma)
        tab.a ~ normal(a, sigma)
    end
    iface = stan_interface(m)
    @test iface["format"] == "stanblocks-interface/1"
    @test iface["program_id"] == stan_descriptor(m).id
    @test iface["parameters"] == ["b", "a", "sigma"]
    # The interface is plain JSON data.
    @test StanBlocks.JSON.parse(StanBlocks.JSON.json(iface)) isa AbstractDict

    data = Dict(e["name"] => e for e in iface["data"])
    @test [e["name"] for e in iface["data"] if e["supplied"]] ==
        ["b0", "y", "idx", "x", "omega", "K", "M", "tab"]
    @test data["y"]["encoding"] == "ragged" && data["y"]["ragged"]["element"] == "real"
    @test data["idx"]["ragged"] == Dict("element" => "int", "values" => "1", "ends" => "2")
    @test data["y_ends_n"]["computed"] == Dict("rule" => "size", "of" => "y", "component" => ["ends"], "axis" => 1)
    @test data["M_n"]["computed"] == Dict("rule" => "size", "of" => "M", "component" => String[], "axis" => 2)
    @test data["M"]["size"] == ["M_m", "M_n"]
    @test data["tab_r_mem_n"]["computed"]["component"] == ["r", "mem"]
    @test [(f["key"], f["name"], f["encoding"]) for f in data["tab"]["fields"]] ==
        [("1", "a", "value"), ("2", "r", "ragged")]
    @test data["y"]["observed"] && !data["x"]["observed"]

    outputs = Dict(o["name"] => o for o in iface["outputs"])
    @test outputs["mu"]["block"] == "transformed_parameters"
    @test outputs["mu"]["dims"] == [6] && outputs["mu"]["segments"] == [4, 6]
    @test outputs["mu"]["segments_rule"] == Dict("rule" => "groups_of", "of" => "idx")
    @test outputs["effect"]["segments_rule"] == outputs["loop_i_scale"]["segments_rule"]
    @test outputs["y_gen"]["generative"] == "draw" && outputs["y_gen"]["source"] == "y"
    @test outputs["y_likelihood"]["segments_rule"] == Dict("rule" => "groups_of", "of" => "y")
    @test outputs["r"]["segments_rule"] == Dict("rule" => "lengths", "of" => "K")
    @test outputs["b"]["segments_rule"] === nothing && outputs["M_gen"]["dims"] == [2, 3]

    # The writer is the payload BridgeStan receives, and a consumer that knows
    # only the interface writes the same file from its raw values.
    @test stan_data_json(m) == StanBlocks.stan.bridgestan_data(StanBlocks.stan.stan_data(m))
    @test stan_data_json(stan_descriptor(m)) == stan_data_json(m)
    @test StanBlocks.JSON.parse(StanBlocks.JSON.json(interface_json(iface, raw))) ==
        StanBlocks.JSON.parse(stan_data_json(m))

    # Rebinding the data re-derives every size and layout.
    newraw = merge(raw, Dict("y" => [[0.3], [0.1, 0.2, 0.4]], "idx" => [[1], [1, 2, 3]],
        "x" => [[0.5], [1.0, 2.0, 3.0]]))
    rebound = StanBlocks.stan_model(m)(; y = newraw["y"], idx = newraw["idx"], x = newraw["x"])
    riface = stan_interface(rebound)
    @test only(o["segments"] for o in riface["outputs"] if o["name"] == "mu") == [1, 4]
    @test StanBlocks.JSON.parse(StanBlocks.JSON.json(interface_json(riface, newraw))) ==
        StanBlocks.JSON.parse(stan_data_json(rebound))
end

"""
The data a consumer encodes from `stan_interface` loads into the compiled
program, which then evaluates the same log density as with StanBlocks' own data;
the interface's parameter order and output `dims` match BridgeStan's names.
"""
@testitem "slic: data encoded from stan_interface fits through BridgeStan" tags=[:slic, :descriptor, :ragged, :interface, :bridgestan] setup=[StanBlocksImports, InterfaceJSON] begin
    raw = Dict{String,Any}("y" => [[0.1, 0.2, 0.15], [0.2, 0.1]], "x" => [[1.0, 2.0, 0.5], [0.0, 1.0]])
    m = @slic (; y = raw["y"], x = raw["x"]) begin
        b ~ normal(0, 1); sigma ~ exponential(1)
        @plate for i in eachindex(y)
            mu[i] = b .+ x[i]
            y[i] ~ normal(mu[i], sigma)
        end
    end
    iface = stan_interface(m)
    library = stan_library(stan_code(m))
    from_interface = BridgeStan.StanModel(library, StanBlocks.JSON.json(interface_json(iface, raw)))
    from_stanblocks = BridgeStan.StanModel(library, stan_data_json(m))
    for theta in ([0.0, 0.0], [0.4, -0.3], [-1.1, 0.8])
        lp = BridgeStan.log_density(from_interface, theta)
        @test isfinite(lp) && lp == BridgeStan.log_density(from_stanblocks, theta)
    end
    stem(n) = first(split(n, '.'))
    @test unique(stem.(BridgeStan.param_unc_names(from_interface))) == iface["parameters"]
    names = BridgeStan.param_names(from_interface; include_tp = true, include_gq = true)
    for o in iface["outputs"]
        @test count(n -> stem(n) == o["name"], names) == prod(o["dims"]; init = 1)
    end
end

# A consumer that knows only `stan_interface` and its raw values: each supplied
# value encoded as the interface says, each computed size read off the value it
# sizes.
@testsnippet InterfaceJSON begin
    interface_encode(entry, v) =
        entry["encoding"] == "ragged" ? Dict(
            entry["ragged"]["values"] => reduce(vcat, v;
                init = entry["ragged"]["element"] == "int" ? Int[] : Float64[]),
            entry["ragged"]["ends"] => cumsum(length.(v))) :
        entry["encoding"] == "tuple" ?
            Dict(f["key"] => interface_encode(f, v[f["name"]]) for f in entry["fields"]) :
        v isa AbstractMatrix ? [collect(row) for row in eachrow(v)] : v
    interface_component(v, entry, path) = isempty(path) ? v : begin
        f = only(f for f in entry["fields"] if f["name"] == path[1])
        sub = entry["encoding"] == "ragged" ? interface_encode(entry, v)[f["key"]] : v[f["name"]]
        interface_component(sub, f, path[2:end])
    end
    interface_json(iface, raw) = begin
        entries = Dict(e["name"] => e for e in iface["data"])
        Dict(e["name"] => if e["supplied"]
                interface_encode(e, raw[e["name"]])
            else
                rule = e["computed"]
                size(interface_component(raw[rule["of"]], entries[rule["of"]], rule["component"]), rule["axis"])
            end
            for e in iface["data"])
    end
end
