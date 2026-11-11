"""
`/` on two integers is real division, as in Julia (decision 12ehvh9, todo
14vbs5d). It used to emit Stan's truncating `k / 2`, so a `@deffun` body,
a model-scope binding or a bound meant `1` where its source reads `1.5`, even
with a declared `::real` result. The quotient is now emitted with a promoted
numerator, `(k * 1.0) / 2`, and the Julia layer agrees. Integer division is `÷`
(Stan's `%/%`). Where an integer is needed from `/`, the program fails at trace
time naming `÷`: a size, `m = m / 2` and `m /= 2` on an integer local.
"""
@testitem "slic: two integers divide as in Julia, and ÷ is integer division" tags=[:slic, :regression, :stanc, :bridgestan] setup=[StanBlocksImports, StanBlocksTestSetup] begin
    using .StanBlocksTestSetup: stan_block, stanc_compiles
    @deffun @juliacompat iq_half(k::int)::real = k / 2
    @deffun iq_quot(k::int)::int = k ÷ 2
    @deffun iq_rebind(k::int)::int = begin
        m = k
        m = m / 2
        m
    end
    @deffun iq_update(k::int)::int = begin
        m = k
        m /= 2
        m
    end
    errmsg(m) = try (stan_code(m); "") catch e; sprint(showerror, e) end

    m = @slic (; k = 3, y = 0.3) begin
        h = k / 2
        s ~ normal(0.0, 1.0)
        y ~ normal(s + h + iq_half(k) + iq_quot(k), 1.0)
    end
    code = stan_code(m)
    @test occursin("real h = ((k * 1.0) / 2);", code)
    @test occursin("return ((k * 1.0) / 2);", stan_block(code, "functions"))
    @test occursin("return (k %/% 2);", stan_block(code, "functions"))
    @test stanc_compiles(m)
    # The Julia method agrees with the emitted Stan.
    @test Base.invokelatest(iq_half, 3) === 1.5
    # BridgeStan reads the transformed data back: 1.5, not 1.
    p = instantiate(stan_model(m))
    names = BridgeStan.param_names(p.model; include_tp=true, include_gq=true)
    @test !("h" in names) || BridgeStan.param_constrain(p.model, [0.0];
        include_tp=true, include_gq=true)[findfirst(==("h"), names)] == 1.5

    # Refused where an integer is needed from `/`.
    @test occursin("stores a real into the integer `m`", errmsg(@slic((; k = 3, y = 0.3), begin
        y ~ normal(iq_rebind(k), 1.0)
    end)))
    @test occursin("`m /= …` divides the integer `m`", errmsg(@slic((; k = 3, y = 0.3), begin
        y ~ normal(iq_update(k), 1.0)
    end)))
    @test occursin("a size must be an integer", errmsg(@slic((; k = 4, y = [0.3, 0.1]), begin
        v ~ normal(0.0, 1.0; n = k / 2); y ~ normal(v, 1.0)
    end)))
    @test occursin("a size must be an integer", errmsg(@slic((; k = 4, y = [0.3, 0.1]), begin
        v::vector[k / 2] ~ normal(0.0, 1.0); y ~ normal(v, 1.0)
    end)))
    halved = @slic (; k = 4, y = [0.3, 0.1]) begin
        v::vector[k ÷ 2] ~ normal(0.0, 1.0); y ~ normal(v, 1.0)
    end
    @test occursin("vector[(k %/% 2)] v;", stan_code(halved))
    @test stanc_compiles(halved)
end
