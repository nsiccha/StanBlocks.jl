"""
Stan Math has no `binomial_logit_rng` in any signature (stanc 2.39
`--dump-stan-math-signatures`), so every predictive draw of a `binomial_logit`
observation lowers to `binomial_rng(N, inv_logit(eta))`. The sized-token
overloads already did that for a vector observation. The draw of a SCALAR
observation calls `binomial_logit_rng(int, real)`, which was not defined (snag
`sbbrmi-on-a-brm-234397df`, from a downstream module's plate cell). In a plate
cell the unresolved call reached stanc, which rejected it with `Function
"binomial_logit_rng" is not implemented for distribution "binomial_logit"`. A
scalar `y` at model scope failed to trace (`tracetype not defined for y_gen …
binomial_logit_rng(::int, ::real)`). The two container forms were declared as
native Stan functions, so a direct call of either was rejected by stanc too.
"""
@testitem "slic: a scalar binomial-logit observation draws through binomial_rng" tags=[:slic, :plate, :regression, :descriptor, :stanc, :bridgestan] setup=[StanBlocksImports, StanBlocksTestSetup] begin
    using .StanBlocksTestSetup: stan_block, stanc_compiles
    using Statistics: mean
    inv_logit(x) = 1 / (1 + exp(-x))

    # (A) The reported shape: a scalar logit-binomial observation in a plate cell.
    cell = @slic (; y = [3, 5, 7], n = [10, 10, 10]) begin
        eta ~ normal(0.0, 1000.0)
        @plate for i in eachindex(y)
            y[i] ~ binomial_logit(n[i], eta)
        end
    end
    code = stan_code(cell)
    let f = stan_block(code, "functions")
        @test occursin("int binomial_logit_rng(int N, real eta)", f)
        @test occursin("return binomial_rng(N, inv_logit(eta));", f)
    end
    @test occursin("y_gen[i] = binomial_logit_rng(n[i], eta);", stan_block(code, "generated quantities"))
    @test stanc_compiles(cell)
    d = stan_descriptor(cell; name = :plate_binomial_logit)
    outs = Dict(o.name => o for o in d.outputs)
    @test outs[:y_gen].type == :int
    @test outs[:y_gen].size == (:y_n,)
    @test outs[:y_gen].generative == :draw
    @test stan_operation(d, :predict).outputs == (:y_gen,)
    # Live: at eta = 2 every cell draws Binomial(10, inv_logit(2)). Over 6000
    # draws the mean's standard error is about 0.013.
    pred = stan_execute(d, :predict; draws = fill(2.0, 1, 2000), seed = 2026)
    @test size(pred.y_gen) == (3, 2000)
    @test all(v -> isinteger(v) && 0 <= v <= 10, pred.y_gen)
    @test abs(mean(pred.y_gen) - 10 * inv_logit(2.0)) < 0.1

    # (B) A per-cell linear predictor in the same cell.
    linear = @slic (; y = [3, 5, 7], n = [10, 10, 10], x = [-1.0, 0.0, 1.0]) begin
        a ~ normal(0.0, 1.0)
        b ~ normal(0.0, 1.0)
        @plate for i in eachindex(y)
            y[i] ~ binomial_logit(n[i], a + b * x[i])
        end
    end
    @test occursin("binomial_logit_rng(n[i], (a + (b * x[i])))", stan_block(stan_code(linear), "generated quantities"))
    @test stanc_compiles(linear)

    # (C) A scalar observation at model scope used to fail while tracing.
    scalar = @slic (; y = 3, n = 10) begin
        eta ~ normal(0.0, 1.0)
        y ~ binomial_logit(n, eta)
    end
    @test occursin("int y_gen = binomial_logit_rng(n, eta);", stan_block(stan_code(scalar), "generated quantities"))
    @test stanc_compiles(scalar)
    scalar_outs = Dict(o.name => o for o in stan_descriptor(scalar; name = :scalar_binomial_logit).outputs)
    @test scalar_outs[:y_gen].type == :int
    @test scalar_outs[:y_gen].size == ()

    # (D) Direct calls of every form. The two container-only forms used to be
    # declared native, which emitted a call stanc rejects.
    direct = @slic (; y = [3, 5, 7], n = [10, 10, 10], N = 10, x = [-1.0, 0.0, 1.0]) begin
        a ~ normal(0.0, 1.0)
        b ~ normal(0.0, 1.0)
        log_odds = a + b * x
        y ~ binomial_logit(n, log_odds)
        r1 = binomial_logit_rng(N, a)
        r2 = binomial_logit_rng(n, log_odds)
        r3 = binomial_logit_rng(N, log_odds)
        r4 = binomial_logit_rng(n, a)
    end
    let g = stan_block(stan_code(direct), "generated quantities")
        @test occursin("int r1 = binomial_logit_rng(N, a);", g)
        @test occursin(r"array\[\w+\] int r2 = binomial_logit_rng\(n, log_odds\);", g)
        @test occursin(r"array\[\w+\] int r3 = binomial_logit_rng\(N, log_odds\);", g)
        @test occursin(r"array\[\w+\] int r4 = binomial_logit_rng\(n, a\);", g)
    end
    @test stanc_compiles(direct)

    # (E) An empty plate still compiles beside the scalar log odds.
    @test stanc_compiles(@slic (; y = Int[], n = Int[]) begin
        eta ~ normal(0.0, 1.0)
        @plate for i in eachindex(y)
            y[i] ~ binomial_logit(n[i], eta)
        end
    end)
end
