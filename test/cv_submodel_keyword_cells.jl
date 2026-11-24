"""
A `maybecv` mark holds out an observation inside a called sub-model whose
keyword is the caller's plate cell (`member(; ys = rows)` in a plate over
`ys`), as it does for the same observation written at model scope: what the
mark taints is re-drawn in generated quantities, the observation's likelihood
leaves the model block while its `ys_gen` / `ys_likelihood` twins read the
re-drawn values, and every untainted parameter (`sigma`, the per-cell `shift`)
stays a parameter that an earlier fit's draws supply.

Before, the held-out observation re-typed the keyword's local alias by minting
a new model-scope binding (`<lhs>_ys`), which the enclosing plate then promoted
to per-cell storage (snag `transpile-a-nest-9789d814`). A plate or annotated
loop inside the sub-model failed ("Sampling statement `read ~ ...` has LHS
bound to a parameter-qualified value" or "Could not find …_s in model"), and a
sub-model observing its keyword directly transpiled silently as a prior draw:
`sigma` and `shift` re-drawn and the observed data replaced by a fresh draw,
with no `ys_likelihood`.
"""
@testitem "slic: a cv mark holds out an observation of a sub-model keyword bound to a plate cell" tags=[:slic, :plate, :submodel, :regression, :stanc] setup=[StanBlocksImports, StanBlocksTestSetup] begin
    using .StanBlocksTestSetup: stanc_compiles, stan_block
    import StanBlocks.stan: stan_descriptor, maybecv

    do_member = @slic begin
        shift ~ normal(0.0, 1.0)
        result ~ plate(ys) do read
            loc = m + shift
            read ~ normal(loc, 1.0)
            loc
        end
        return result
    end
    loop_member = @slic begin
        shift ~ normal(0.0, 1.0)
        loc = m + shift
        @plate for s in eachindex(ys)
            ys[s] ~ normal(loc, 1.0)
        end
        return loc
    end
    direct_member = @slic begin
        shift ~ normal(0.0, 1.0)
        loc = m + shift
        ys ~ normal(loc, 1.0)
        return loc
    end
    do_plate = @slic begin
        sigma ~ normal(0, 1; lower=0)
        n = maximum(groups)
        z::vector[n] ~ std_normal()
        m = sigma * z[1]
        result ~ plate(ys) do rows
            value ~ do_member(; ys=rows, m=m)
            value
        end
    end
    do_plate_loop = @slic begin
        sigma ~ normal(0, 1; lower=0)
        n = maximum(groups)
        z::vector[n] ~ std_normal()
        m = sigma * z[1]
        result ~ plate(ys) do rows
            value ~ loop_member(; ys=rows, m=m)
            value
        end
    end
    do_plate_direct = @slic begin
        sigma ~ normal(0, 1; lower=0)
        n = maximum(groups)
        z::vector[n] ~ std_normal()
        m = sigma * z[1]
        result ~ plate(ys) do rows
            value ~ direct_member(; ys=rows, m=m)
            value
        end
    end
    loop_plate = @slic begin
        sigma ~ normal(0, 1; lower=0)
        n = maximum(groups)
        z::vector[n] ~ std_normal()
        m = sigma * z[1]
        @plate for e in eachindex(ys)
            value ~ do_member(; ys=ys[e], m=m)
        end
    end
    nested = [[[0.4], [-0.5, 0.1]], [[0.8, 0.3], [0.2]]]
    flat = [[0.4, -0.5], [0.1, 0.8, 0.3]]
    cases = (
        ("do-plate in the sub-model", do_plate, nested),
        ("annotated loop in the sub-model", do_plate_loop, nested),
        ("the sub-model observes its keyword", do_plate_direct, flat),
        ("annotated outer loop", loop_plate, nested),
    )
    for (label, model, ys) in cases
        @testset "$label" begin
            fit = model(; groups=[1, 2], ys)
            fit_code = stan_code(fit)
            @test occursin("vector[n] z;", stan_block(fit_code, "parameters"))
            @test occursin("ys.1[", stan_block(fit_code, "model"))
            @test :fit in (o.name for o in stan_descriptor(fit).operations)

            cv = model(; groups=maybecv(:groups, [1, 2]), ys)
            code = stan_code(cv)
            @test stanc_compiles(cv)
            params = stan_block(code, "parameters")
            @test occursin("real<lower=0> sigma;", params)
            @test occursin("_shift;", params)
            @test !occursin(" z;", params)
            model_block = stan_block(code, "model")
            @test !occursin("ys.1[", model_block)
            @test occursin("_shift", model_block)
            gq = stan_block(code, "generated quantities")
            @test occursin("vector[n] z = std_normal_vector_rng(n);", gq)
            @test occursin("ys_gen", gq)
            @test occursin("ys_likelihood", gq)
            d = stan_descriptor(cv)
            ops = [o.name for o in d.operations]
            @test :predict in ops && :pointwise_loglik in ops && !(:fit in ops)
            kinds = Dict(o.name => o.kind for o in d.outputs)
            @test kinds[:sigma] === :parameter
            @test kinds[:z] === :generated_quantity
            @test any(k -> endswith(String(k), "_shift") && kinds[k] === :parameter, keys(kinds))
        end
    end
end
