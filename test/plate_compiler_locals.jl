"""
A plate traces its cell twice: a discovery probe that finds the cell's bindings,
then the trace that emits the loop. An inlined helper's locals (`name__il_<id>`)
and a comprehension's result (`comprehension_result__lc_<id>`) carry a per-trace
id in their names, so the two traces named them differently. The probe's names
were promoted to per-cell storage the emitted loop never filled: a parameter
nothing reads (a `matrix` for a vector local), or an `int` array parameter that
stanc refuses (a helper's size binding, `n` of `x::vector[n]`). A likelihood
that read such a local directly referenced a name declared only in another
block's copy of the loop. Each is now loop scratch, recomputed in every block
copy whose statements read it (snag inline-comprehen-20e73a74).

Every model below is written once and traced twice: with `@inline` helpers and
with the same helpers as Stan functions, which never had the defect. The two
programs must have the same unconstrained coordinates, and the same density.
"""
@testmodule PlateCompilerLocalsFixtures begin
    using StanBlocks
    @deffun shift_plain(x::vector[n], b::real)::vector[n] = [v + b for v in x]
    @deffun @inline shift_inline(x::vector[n], b::real)::vector[n] = [v + b for v in x]
    @deffun outer_plain(x::vector[n], w::vector[m], b::real)::matrix[n, m] =
        [x[i] * w[j] + b for i in 1:n, j in 1:m]
    @deffun @inline outer_inline(x::vector[n], w::vector[m], b::real)::matrix[n, m] =
        [x[i] * w[j] + b for i in 1:n, j in 1:m]
    @deffun local_plain(x::vector[n], b::real)::real = begin
        t = x .+ b
        sum(t)
    end
    @deffun @inline local_inline(x::vector[n], b::real)::real = begin
        t = x .+ b
        sum(t)
    end
    @deffun size_plain(x::vector[n], b::real)::real = n * b
    @deffun @inline size_inline(x::vector[n], b::real)::real = n * b
    @deffun filled_plain(x::vector[n], b::real)::vector[n] = begin
        out::vector[n]
        for i in 1:n
            out[i] = x[i] + b
        end
        out
    end
    @deffun @inline filled_inline(x::vector[n], b::real)::vector[n] = begin
        out::vector[n]
        for i in 1:n
            out[i] = x[i] + b
        end
        out
    end
    const D = (; x = [1.0, 2.0], y = [0.1, 0.2, 0.3])
    const DJ = (; x = [1.0, 2.0], y = [0.1, 0.2], J = 2, S = 3)
    # `SHIFT`, `OUTER`, `LOCAL`, `SIZE` and `FILLED` name the helpers; each
    # module below binds them to one of the two sets.
    const MODELS = quote
        member = @slic begin
            s = sum(SHIFT(x, b))
            return s
        end
        models = (
            # A cell binding of the helper's value, collected (the reported shape).
            bound = @slic(D, begin
                b ~ normal(0, 1)
                shared ~ plate(y) do yi
                    selected = SHIFT(x, b)
                    mu = sum(selected)
                    yi ~ normal(mu, 1)
                    mu
                end
            end),
            # A comprehension written in the cell itself (no helper).
            raw_bound = @slic(D, begin
                b ~ normal(0, 1)
                shared ~ plate(y) do yi
                    selected = [v + b for v in x]
                    mu = sum(selected)
                    yi ~ normal(mu, 1)
                    mu
                end
            end),
            raw_direct = @slic(D, begin
                b ~ normal(0, 1)
                shared ~ plate(y) do yi
                    yi ~ normal(sum([v + b for v in x]), 1)
                end
            end),
            # The likelihood reads the comprehension directly.
            direct = @slic(D, begin
                b ~ normal(0, 1)
                shared ~ plate(y) do yi
                    yi ~ normal(sum(SHIFT(x, b)), 1)
                end
            end),
            data_only = @slic(D, begin
                b ~ normal(0, 1)
                shared ~ plate(y) do yi
                    mu = sum(SHIFT(x, 1.0)) * b
                    yi ~ normal(mu, 1)
                    mu
                end
            end),
            cell_parameter = @slic(D, begin
                b ~ normal(0, 1)
                zs ~ plate(y) do yi
                    z ~ normal(sum(SHIFT(x, b)), 1)
                    yi ~ normal(z, 1)
                    z
                end
            end),
            # One comprehension sized by another.
            chained = @slic(D, begin
                b ~ normal(0, 1)
                shared ~ plate(y) do yi
                    a = SHIFT(x, b)
                    c = sum(SHIFT(a, b))
                    yi ~ normal(c, 1)
                    c
                end
            end),
            # A two-axis comprehension over the helper's size bindings.
            two_axes = @slic(D, begin
                b ~ normal(0, 1)
                shared ~ plate(y) do yi
                    mu = sum(OUTER(x, x, b))
                    yi ~ normal(mu, 1)
                    mu
                end
            end),
            annotated = @slic(D, begin
                b ~ normal(0, 1)
                @plate for i in 1:3
                    y[i] ~ normal(sum(SHIFT(x, b)), 1)
                end
            end),
            scan = @slic(D, begin
                @scan begin
                    h[1] ~ normal(0, 1)
                    for t in 2:3
                        h[t] ~ normal(0.5 * sum(SHIFT(x, h[t-1])), 1)
                    end
                end
                y ~ normal(h, 1)
            end),
            nested = @slic(DJ, begin
                b ~ normal(0, 1)
                @plate for j in 1:J
                    tau_s ~ plate(; outer = S) do s
                        t ~ normal(sum(SHIFT(x, b)), 1.0)
                        t
                    end
                    y[j] ~ normal(sum(tau_s), 1.0)
                end
            end),
            submodel = @slic(D, begin
                b ~ normal(0, 1)
                shared ~ plate(y) do yi
                    m ~ member(; x, b)
                    yi ~ normal(m, 1)
                    m
                end
            end),
            # Outside any plate: the same sub-model called at model scope. Its
            # comprehension's fill left the sub-model's view of the result
            # stale, and the backward pass refused the fill as a write to a
            # parameter.
            submodel_model_scope = @slic(D, begin
                b ~ normal(0, 1)
                m ~ member(; x, b)
                y ~ normal(m, 1)
            end),
            vector_local = @slic(D, begin
                b ~ normal(0, 1)
                shared ~ plate(y) do yi
                    yi ~ normal(LOCAL(x, b), 1)
                end
            end),
            size_local = @slic(D, begin
                b ~ normal(0, 1)
                shared ~ plate(y) do yi
                    mu = SIZE(x, b)
                    yi ~ normal(mu, 1)
                    mu
                end
            end),
            filled_local = @slic(D, begin
                b ~ normal(0, 1)
                shared ~ plate(y) do yi
                    yi ~ normal(sum(FILLED(x, b)), 1)
                end
            end),
        )
    end
    module Plain
        using StanBlocks
        import ..shift_plain, ..outer_plain, ..local_plain, ..size_plain, ..filled_plain, ..D, ..DJ, ..MODELS
        const SHIFT = shift_plain; const OUTER = outer_plain; const LOCAL = local_plain
        const SIZE = size_plain; const FILLED = filled_plain
        Core.eval(@__MODULE__, MODELS)
    end
    module Inline
        using StanBlocks
        import ..shift_inline, ..outer_inline, ..local_inline, ..size_inline, ..filled_inline, ..D, ..DJ, ..MODELS
        const SHIFT = shift_inline; const OUTER = outer_inline; const LOCAL = local_inline
        const SIZE = size_inline; const FILLED = filled_inline
        Core.eval(@__MODULE__, MODELS)
    end
    # A mean of 3 + 2b: `sum(x .+ b)` for `x = [1, 2]`.
    const SAME_MEAN = (:bound, :raw_bound, :raw_direct, :direct, :annotated, :submodel,
        :submodel_model_scope, :vector_local, :filled_local)
end

@testitem "slic: an inlined helper's locals and a comprehension in a plate cell are loop scratch" tags=[:slic, :plate, :regression, :stanc] setup=[StanBlocksImports, StanBlocksTestSetup, PlateCompilerLocalsFixtures] begin
    using .StanBlocksTestSetup: stanc_compiles, stan_block
    using .PlateCompilerLocalsFixtures: Plain, Inline
    for name in keys(Inline.models)
        @testset "$name" begin
            inline, plain = Inline.models[name], Plain.models[name]
            @test stan_unc_names(inline) == stan_unc_names(plain)
            params = stan_block(stan_code(inline), "parameters")
            @test !occursin("__il_", params)
            @test !occursin("comprehension_result", params)
            @test stanc_compiles(inline)
        end
    end
    @test stan_unc_names(Inline.models.bound) == ["b"]
    @test stan_unc_names(Inline.models.raw_bound) == stan_unc_names(Inline.models.raw_direct) == ["b"]
    # Read only by the likelihood and its predictive draw, the comprehension is
    # recomputed inside the model block's loop and the generated-quantities
    # loop; transformed parameters holds nothing.
    code = stan_code(Inline.models.direct)
    fill = r"for\(value_index__vi_\d+ in 1:num_elements\(x\)\)"
    @test occursin(fill, stan_block(code, "model"))
    @test occursin(fill, stan_block(code, "generated quantities"))
    @test !occursin("comprehension_result", stan_block(code, "transformed parameters"))
end

@testitem "slic: plate cells with inlined helper locals keep the helper-call density" tags=[:slic, :plate, :regression, :bridgestan] setup=[StanBlocksImports, PlateCompilerLocalsFixtures] begin
    using .PlateCompilerLocalsFixtures: Inline, SAME_MEAN
    # b ~ normal(0, 1); each y[i] ~ normal(mu(b), 1).
    normal_lp(y, mu) = -0.5 * (y - mu)^2 - 0.5 * log(2pi)
    y = [0.1, 0.2, 0.3]
    expected(b, mu, dmu) = (normal_lp(b, 0.0) + sum(normal_lp.(y, mu)), -b + sum(y .- mu) * dmu)
    cases = Any[(name, b -> 3 + 2b, 2.0) for name in SAME_MEAN]
    push!(cases, (:chained, b -> 3 + 4b, 4.0), (:size_local, b -> 2b, 2.0))
    for (name, mean, slope) in cases
        @testset "$name" begin
            p = instantiate(Inline.models[name])
            @test BridgeStan.param_unc_names(p.model) == ["b"]
            for b in (-0.4, 0.3, 0.9)
                lp, grad = BridgeStan.log_density_gradient(p.model, [b]; propto=false, jacobian=false)
                lp_ref, grad_ref = expected(b, mean(b), slope)
                @test lp ≈ lp_ref
                @test grad ≈ [grad_ref]
            end
        end
    end
end
