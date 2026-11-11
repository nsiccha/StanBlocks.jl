"""
`y ~ student_t(nu, 0.0, sg)` with a scalar dof and location and a vector scale
failed to trace its `y_gen` draw: the native `student_t_rng` table had no
`(real, real, vector)` row, although Stan Math vectorises every argument (todo
0hv2zix). The row exists now, for `skew_normal`, `exp_mod_normal`,
`skew_double_exponential` and `pareto_type_2` too.
"""
@testitem "slic: a vector scale alone draws a student_t predictive" tags=[:slic, :regression, :stanc, :bridgestan] setup=[StanBlocksImports, StanBlocksTestSetup] begin
    using .StanBlocksTestSetup: stan_block, stanc_compiles
    m = @slic (; y = [0.1, -0.4, 0.7, 1.2], sg = [0.6, 1.0, 1.4, 0.8]) begin
        nu ~ exponential(1.0)
        y ~ student_t(nu, 0.0, sg)
    end
    gq = stan_block(stan_code(m), "generated quantities")
    @test occursin("y_gen", gq)
    @test stanc_compiles(m)
    p = instantiate(stan_model(m))
    names = BridgeStan.param_names(p.model; include_tp=true, include_gq=true)
    draw = BridgeStan.param_constrain(p.model, [0.3]; include_tp=true, include_gq=true,
        rng=BridgeStan.StanRNG(p.model, 3))
    gen = [draw[i] for (i, n) in enumerate(names) if startswith(n, "y_gen")]
    @test length(gen) == 4 && all(isfinite, gen)
end

@testitem "slic: three-argument RNGs accept either vector orientation" tags=[:slic, :regression, :stanc] setup=[StanBlocksImports, StanBlocksTestSetup] begin
    using .StanBlocksTestSetup: stanc_compiles
    # Check the native signatures against stanc, including all 27 combinations
    # of scalar, column and row arguments. Each family is one small program.
    for family in (:student_t, :skew_normal, :exp_mod_normal,
                   :skew_double_exponential, :pareto_type_2)
        rng = Symbol(family, :_rng)
        choices = (1.0, :v, :(D[1, :]))
        draws = [Expr(:(=), Symbol(:draw_, i), Expr(:call, rng, args...))
                 for (i, args) in enumerate(Iterators.product(choices, choices, choices))]
        m = StanBlocks.SlicModel(quote
            z ~ std_normal()
            $(draws...)
        end, Dict{Symbol,Any}(:v => [0.5, 0.7, 0.9], :D => [0.5 0.7 0.9]), @__MODULE__)
        @test stanc_compiles(m)

        # The observed-family path also needs sized RNG overloads when its
        # first argument is a row, plus the scalar/scalar/row native signature.
        observed = StanBlocks.SlicModel(quote
            z ~ std_normal()
            y1 ~ $family(1.0, 1.0, D[1, :])
            y2 ~ $family(D[1, :], 1.0, D[1, :])
            y3 ~ $family(D[1, :], D[1, :], 1.0)
        end, Dict{Symbol,Any}(:D => [0.5 0.7 0.9],
            :y1 => [1.0, 2.0, 3.0], :y2 => [1.0, 2.0, 3.0], :y3 => [1.0, 2.0, 3.0]), @__MODULE__)
        @test stanc_compiles(observed)
    end
end
