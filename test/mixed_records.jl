using TestItemRunner

"""
Mixed records retain integer fields at the Stan boundary. Likelihood-reaching
records and their consumers need model-local storage plus generated outputs;
data-only, prior-only, and real-only records retain their usual block placement.
"""
@testitem "slic: mixed records route their dependent transforms" tags=[:slic, :stanc] setup=[StanBlocksImports, StanBlocksTestSetup] begin
    using .StanBlocksTestSetup: stanc_compiles, stan_block

    @deffun read_record(r, input::vector[n])::vector[n] =
        r.offset .+ r.scale .* input
    const cell = @slic begin
        offset ~ normal(0, 1)
        record = (; offset, scale=2.0, input, weight=exp(offset), count=0)
        alias = record
        return read_record(alias, input)
    end
    composed = @slic (; input=[0.1, 0.4, 0.8], y=[0.2, 0.5, 0.9]) begin
        mu ~ cell(; input)
        y ~ normal(mu, 1)
    end
    code = stan_code(composed)
    @test stanc_compiles(composed)
    @test !occursin("mu_record", stan_block(code, "transformed parameters"))
    @test !occursin("mu =", stan_block(code, "transformed parameters"))
    for b in ("model", "generated quantities")
        block = stan_block(code, b)
        @test occursin("tuple(real, real, vector[input_n], real, int) mu_record", block)
        @test occursin("mu_alias = mu_record", block)
        @test occursin("mu = read_record(mu_alias, input)", block)
    end
    outputs = Dict(o.name => o.kind for o in stan_descriptor(composed).outputs)
    @test outputs[:mu_record] == :generated_quantity
    @test outputs[:mu_alias] == :generated_quantity
    @test outputs[:mu] == :generated_quantity
    @test outputs[:mu_offset] == :parameter

    @deffun make_record(location::real, indices::int[n]) =
        (; inner=(; location, count=2), indices)
    @deffun read_nested(r, input::vector[n])::real =
        r.inner.location + input[r.indices[1]]
    nested = @slic (; input=[0.1, 0.4], indices=[2, 1], y=0.3) begin
        location ~ normal(0, 1)
        independent = 2 * location
        record = make_record(location, indices)
        mu = read_nested(record, input)
        y ~ normal(mu + independent, 1)
    end
    code = stan_code(nested)
    @test stanc_compiles(nested)
    @test occursin("real independent", stan_block(code, "transformed parameters"))
    @test !occursin("record =", stan_block(code, "transformed parameters"))
    @test !occursin("mu =", stan_block(code, "transformed parameters"))
    @test occursin("tuple(tuple(real, int), array[indices_n] int) record", stan_block(code, "model"))
    @test occursin("record = make_record", stan_block(code, "generated quantities"))

    @deffun @inline fill_record(r, input::vector[n])::vector[n] = begin
        out::vector[n]
        for i in 1:n
            out[i] = r.location + input[i]
        end
        return out
    end
    filled = @slic (; input=[0.1, 0.4], y=[0.2, 0.5]) begin
        location ~ normal(0, 1)
        record = (; location, count=2)
        mu = fill_record(record, input)
        y ~ normal(mu, 1)
    end
    code = stan_code(filled)
    @test stanc_compiles(filled)
    @test !occursin("for(", stan_block(code, "transformed parameters"))
    @test occursin("for(", stan_block(code, "model"))
    @test occursin("for(", stan_block(code, "generated quantities"))

    @deffun select_indices(location::real)::int[2] = begin
        out::int[2]
        out[1] = location > 0 ? 1 : 2
        out[2] = 2
        return out
    end
    indexed = @slic (; input=[0.1, 0.4], y=0.3) begin
        location ~ normal(0, 1)
        indices = select_indices(location)
        mu = input[indices[1]]
        y ~ normal(mu, 1)
    end
    code = stan_code(indexed)
    @test stanc_compiles(indexed)
    @test occursin("array[2] int indices", stan_block(code, "model"))
    @test occursin("array[2] int indices", stan_block(code, "generated quantities"))
    @test !occursin("indices", stan_block(code, "transformed parameters"))

    @deffun read_pair(r)::real = r.location + r.count
    data_only = @slic (; input=0.1, y=0.3) begin
        record = (; location=input, count=2)
        mu = read_pair(record)
        y ~ normal(mu, 1)
    end
    @test stanc_compiles(data_only)
    @test occursin("tuple(real, int) record", stan_block(stan_code(data_only), "transformed data"))

    prior_only = @slic begin
        location ~ normal(0, 1)
        record = (; location, count=2)
        mu = read_pair(record)
    end
    @test stanc_compiles(prior_only)
    @test !occursin("record", stan_block(stan_code(prior_only), "model"))
    @test occursin("tuple(real, int) record", stan_block(stan_code(prior_only), "generated quantities"))

    real_only = @slic (; y=0.3) begin
        location ~ normal(0, 1)
        record = (; location, count=2.0)
        mu = read_pair(record)
        y ~ normal(mu, 1)
    end
    @test stanc_compiles(real_only)
    @test occursin("tuple(real, real) record", stan_block(stan_code(real_only), "transformed parameters"))
end

"""
Relocating a mixed record changes storage, not the density or sampler dimension.
The generated record keeps its integer field and remains a named output.
"""
@testitem "slic: mixed records preserve density and generated values" tags=[:slic, :stanc, :bridgestan] setup=[StanBlocksImports] begin
    @deffun record_mean(r)::real = r.location + r.count
    mixed = @slic (; y=0.3) begin
        location ~ normal(0, 1)
        record = (; location, count=2)
        mu = record_mean(record)
        y ~ normal(mu, 1)
    end
    reference = @slic (; y=0.3) begin
        location ~ normal(0, 1)
        mu = location + 2
        y ~ normal(mu, 1)
    end
    p, ref = instantiate(mixed), instantiate(reference)
    @test LogDensityProblems.dimension(p) == LogDensityProblems.dimension(ref) == 1
    for location in (-0.7, 0.0, 0.3)
        lp, grad = LogDensityProblems.logdensity_and_gradient(p, [location])
        expected_lp, expected_grad = LogDensityProblems.logdensity_and_gradient(ref, [location])
        @test lp ≈ expected_lp atol=1e-10
        @test grad ≈ expected_grad atol=1e-10
    end
    names = BridgeStan.param_names(p.model; include_tp=true, include_gq=true)
    values = BridgeStan.param_constrain(p.model, [0.3]; include_tp=true, include_gq=true,
        rng=BridgeStan.StanRNG(p.model, 1234))
    draw = Dict(zip(names, values))
    @test draw["record:1"] ≈ 0.3
    @test draw["record:2"] == 2
    @test draw["mu"] ≈ 2.3
end
