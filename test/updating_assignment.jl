@testsnippet UpdatingDivisionShims begin
    using StanBlocks
    import StanBlocks.stan: @deffun
    @deffun @juliacompat begin
        # Back-substitution's diagonal step: copy, then divide each element in
        # place by its own factor.
        unit_solve(b::vector[n], factor::vector[n])::vector[n] = begin
            z = rep_vector(0.0, n)
            for i in 1:n
                z[i] = b[i]
            end
            for j in 1:n
                z[j] /= factor[j]
            end
            z
        end
        # Whole-matrix, scalar and matrix-element `/=`.
        halve_then_scale(A::matrix[m, k])::matrix[m, k] = begin
            B = A
            B /= 2.0
            s = 4.0
            s /= 2.0
            B[1, 1] /= s
            B
        end
    end
end

"""
`/=` in a `@deffun` body lowers to Stan's `/=`, like `+=`, `-=` and `*=`. It
was not registered, so its head fell through to a function lookup and tracing
failed with `Could not find /= in model, builtin, … or Main!` (snag
`compile-a-deffun-0cacca7d`, from a downstream module whose custom family draws
through a sparse back-substitution). The reported path is a prior-only model:
omitting the observation makes the family's `_rng` call the solve. Stan's
evaluation, read back from BridgeStan as a generated quantity, matches the
`@juliacompat` method.
"""
@testitem "slic: `/=` in a @deffun lowers to Stan's `/=`" tags=[:slic, :regression, :stanc, :bridgestan] setup=[StanBlocksImports, StanBlocksTestSetup, UpdatingDivisionShims] begin
    using .StanBlocksTestSetup: stan_block, stanc_compiles

    @deffun begin
        @lpxf solved_normal_lpdf(y::vector[n], b::vector[n], factor::vector[n], sigma::real)::real =
            normal_lpdf(y, unit_solve(b, factor), sigma)
        solved_normal_lpdfs(y::vector[n], b::vector[n], factor::vector[n], sigma::real)::vector[n] = begin
            mu = unit_solve(b, factor)
            out::vector[n]
            for i in 1:n
                out[i] = normal_lpdf(y[i], mu[i], sigma)
            end
            out
        end
        solved_normal_rng(vector[n], b::vector[n], factor::vector[n], sigma::real)::vector[n] = begin
            mu = unit_solve(b, factor)
            out::vector[n]
            for i in 1:n
                out[i] = normal_rng(mu[i], sigma)
            end
            out
        end
    end
    b, factor = [1.0, -2.0, 3.0], [2.0, 4.0, 8.0]
    observed = @slic (; b, factor, y = [0.4, -0.6, 0.3]) begin
        sigma ~ normal(0.0, 1.0; lower=0.0)
        y ~ solved_normal(b, factor, sigma)
    end
    prior_only = @slic (; b, factor) begin
        sigma ~ normal(0.0, 1.0; lower=0.0)
        y ~ solved_normal(b, factor, sigma)
    end
    for model in (observed, prior_only)
        functions = stan_block(stan_code(model), "functions")
        @test occursin("z[j] /= factor[j];", functions)
        @test stanc_compiles(model)
    end
    @test occursin(r"y = solved_normal\w*_rng\(", stan_block(stan_code(prior_only), "generated quantities"))

    # Whole-container, scalar and element forms.
    forms = @slic (; A = [1.0 2.0; 3.0 4.0], obs = 0.2) begin
        mu ~ normal(0.0, 1.0)
        obs ~ normal(mu, 1.0)
        out = halve_then_scale(A + mu)
    end
    functions = stan_block(stan_code(forms), "functions")
    for stmt in ("B /= 2.0;", "s /= 2.0;", "B[1, 1] /= s;")
        @test occursin(stmt, functions)
    end
    @test stanc_compiles(forms)

    # Stan's values match the Julia methods.
    @test unit_solve(b, factor) == b ./ factor
    @test halve_then_scale([1.0 2.0; 3.0 4.0]) == [0.25 1.0; 1.5 2.0]
    solved = @slic (; b, factor, obs = 0.2) begin
        mu ~ normal(0.0, 1.0)
        obs ~ normal(mu, 1.0)
        out = unit_solve(b + mu, factor)
    end
    for (model, julia) in ((solved, unit_solve(b, factor)),
                           (forms, vec(halve_then_scale([1.0 2.0; 3.0 4.0]))))
        p = instantiate(stan_model(model))
        names = BridgeStan.param_names(p.model; include_tp=true, include_gq=true)
        values = BridgeStan.param_constrain(p.model, [0.0]; include_tp=true, include_gq=true,
            rng=BridgeStan.StanRNG(p.model, 1))
        @test values[[startswith(s, "out.") for s in names]] == julia
    end
end

"""
An updating assignment of a bare local (`m += 1`, `m /= 2`, `v .= …`) is a
reassignment, so a result size routed through that local has no single defining
expression. Like `m = m + 1`, it now refuses the inferred size by name. Before
snag `compile-a-deffun-0cacca7d` the size was substituted from the local's first
definition: `m = n; m += 1; rep_vector(0.0, m)` declared the caller's result
`vector[n]` for an `n + 1`-element value, which stanc accepts and Stan rejects
at run time.
"""
@testitem "slic: an updating assignment refuses a result size routed through its local" tags=[:slic, :regression] setup=[StanBlocksImports] begin
    @deffun begin
        grown_add(x::vector[n]) = begin
            m = n
            m += 1
            rep_vector(0.0, m)
        end
        grown_div(x::vector[n]) = begin
            m = 2 * n
            m /= 2
            rep_vector(0.0, m)
        end
        grown_broadcast(idx::int[n]) = begin
            v = idx
            v .= v .+ 1
            k = max(v)
            rep_vector(0.0, k)
        end
        # The same body with the size spelled through the argument infers.
        grown_ok(x::vector[n]) = begin
            m = n
            m += 1
            rep_vector(0.0, n + 1)
        end
    end
    model(call) = StanBlocks.stan.SlicModel(quote
        s ~ normal(0.0, 1.0)
        z = $call
        y ~ normal(sum(z) + s, 1.0)
    end, (; x = [1.0, 2.0, 3.0], idx = [3, 1, 2], y = 0.3), @__MODULE__)
    errmsg(call) = try
        stan_code(model(call))
        nothing
    catch e
        sprint(showerror, e)
    end
    for (call, name) in ((:(grown_add(x)), "m"), (:(grown_broadcast(idx)), "v"))
        err = errmsg(call)
        @test err !== nothing &&
            occursin("inferred result size references function-local `$name`, which is reassigned", err)
    end
    # `m /= 2` on an integer is refused before its size is read: `/` gives a real,
    # as in Julia (decision 12ehvh9, todo 14vbs5d).
    err = errmsg(:(grown_div(x)))
    @test err !== nothing && occursin("`m /= …` divides the integer `m`", err)
    @test occursin("vector[(x_n + 1)] z = grown_ok(x);", stan_code(model(:(grown_ok(x)))))
end
