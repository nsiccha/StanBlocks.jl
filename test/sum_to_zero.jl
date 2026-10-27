"""
Stan's isometric sum-to-zero transforms, `sum_to_zero_constrain`,
`sum_to_zero_unconstrain` and `sum_to_zero_jacobian`, take a vector or a
matrix. None was registered, so a body calling one failed to trace and a
downstream module carried a hand-written copy of the transform (snag
`sum-to-zero-cons-90a72024`). The constrained value is one larger (a vector) or
one larger in each dimension (a matrix); the free value one smaller.
"""
@testitem "slic: Stan's sum_to_zero transforms trace over vectors and matrices" tags=[:slic, :regression, :shapes, :stanc] setup=[StanBlocksImports, StanBlocksTestSetup] begin
    using .StanBlocksTestSetup: stan_block, stanc_compiles, transpiles

    # The reported shape: values from a total and the free coordinates of the
    # centred values, and the inverse, in Stan-only `@deffun`s.
    @deffun begin
        total_values(c::vector[n])::vector[n] = c[1] / n + sum_to_zero_constrain(tail(c, n - 1))
        total_coordinates(x::vector[n])::vector[n] =
            append_row(sum(x), sum_to_zero_unconstrain(x - mean(x)))
    end
    reported = @slic (; n=4, y=[0.1, -0.2, 0.3, 0.5]) begin
        c::vector[n] ~ normal(0.0, 1.0)
        x = total_values(c)
        y ~ normal(x, 1.0)
        back = total_coordinates(x)
    end
    code = stan_code(reported)
    @test occursin("sum_to_zero_constrain(tail(c, (n - 1)))", code)
    @test occursin("sum_to_zero_unconstrain((x - mean(x)))", code)
    @test stanc_compiles(reported)

    # Every overload, as a transformed parameter or, of data, transformed data.
    data = (; m=2, n=3, y=0.3, d=[0.5, -0.5, 0.0], D=[1.0 -1.0; -1.0 1.0])
    model(rhs, use=:(sum(out))) = StanBlocks.stan.SlicModel(quote
        v::vector[n] ~ normal(0.0, 1.0)
        M::matrix[m, n] ~ normal(0.0, 1.0)
        V::vector[m, n] ~ multi_normal(rep_vector(0.0, n), identity_matrix(n))
        out = $rhs
        y ~ normal($use + sum(v) + sum(M) + sum(V[1]), 1.0)
    end, data, @__MODULE__)
    cases = [
        :(sum_to_zero_constrain(v)) =>
            ("transformed parameters", "vector[(n + 1)] out = sum_to_zero_constrain(v);"),
        :(sum_to_zero_jacobian(v)) =>
            ("transformed parameters", "vector[(n + 1)] out = sum_to_zero_jacobian(v);"),
        :(sum_to_zero_unconstrain(sum_to_zero_constrain(v))) =>
            ("transformed parameters", "out = sum_to_zero_unconstrain(sum_to_zero_constrain(v));"),
        :(sum_to_zero_constrain(M)) =>
            ("transformed parameters", "matrix[(m + 1), (n + 1)] out = sum_to_zero_constrain(M);"),
        :(sum_to_zero_jacobian(M)) =>
            ("transformed parameters", "matrix[(m + 1), (n + 1)] out = sum_to_zero_jacobian(M);"),
        :(sum_to_zero_unconstrain(sum_to_zero_constrain(M))) =>
            ("transformed parameters", "out = sum_to_zero_unconstrain(sum_to_zero_constrain(M));"),
        :(sum_to_zero_unconstrain(d)) =>
            ("transformed data", "vector[(d_n - 1)] out = sum_to_zero_unconstrain(d);"),
        :(sum_to_zero_unconstrain(D)) =>
            ("transformed data", "matrix[(D_m - 1), (D_n - 1)] out = sum_to_zero_unconstrain(D);"),
    ]
    for (rhs, (block, decl)) in cases
        m = model(rhs)
        @test occursin(decl, stan_block(stan_code(m), block))
        @test stanc_compiles(m)
    end

    # refused: Stan has no row-vector form, so the call stays untyped and its
    # use stops at trace time.
    @test !transpiles(model(:(sum_to_zero_constrain(v')), :(sum(out[1]))); re=false)
    # Stan's array forms (`array[] vector`, `array[] matrix`) are not registered:
    # a `@juliacompat` method could not tell such an argument from a matrix.
    @test_broken transpiles(model(:(sum_to_zero_constrain(V)), :(sum(out[1]))); re=false)
end

@testsnippet SumToZeroShims begin
    using StanBlocks
    import StanBlocks.stan: @deffun
    @deffun @juliacompat begin
        stz(y::vector[n])::vector[n + 1] = sum_to_zero_constrain(y)
        stz_jacobian(y::vector[n])::vector[n + 1] = sum_to_zero_jacobian(y)
        stz_free(z::vector[n])::vector[n - 1] = sum_to_zero_unconstrain(z)
        stzm(y::matrix[m, n])::matrix[m + 1, n + 1] = sum_to_zero_constrain(y)
        stzm_free(z::matrix[m, n])::matrix[m - 1, n - 1] = sum_to_zero_unconstrain(z)
    end
    # The vector transform's matrix: column `i` is the image of the `i`-th unit vector.
    stz_basis(k) = reduce(hcat, [stz([j == i ? 1.0 : 0.0 for j in 1:k]) for i in 1:k])
end

"""
The `@juliacompat` methods of the sum-to-zero transforms give Stan's values, with
Stan's input check on the free form. Stan's own builtins, evaluated by
BridgeStan as generated quantities, are the oracle.
"""
@testitem "slic: @juliacompat sum_to_zero transforms match Stan" tags=[:slic, :bridgestan] setup=[StanBlocksImports, SumToZeroShims] begin
    y = [0.3, -1.2, 0.8, 2.1]
    Y = [0.4 -0.3 1.1; 0.2 0.9 -0.7]

    # The vector form sums to zero, is an isometry, and round-trips.
    z = stz(y)
    @test length(z) == 5
    @test abs(sum(z)) < 1e-12
    A = stz_basis(4)
    @test A' * A ≈ [i == j ? 1.0 : 0.0 for i in 1:4, j in 1:4] atol=1e-12
    @test A * y ≈ z atol=1e-12
    @test stz_free(z) ≈ y atol=1e-12
    @test stz_jacobian(y) == z
    @test stz(Float64[]) == [0.0]
    @test stz_free([0.0]) == Float64[]
    # The matrix form applies an isometric basis along both dimensions: the
    # vector transform's, shifted cyclically by one coordinate.
    Z = stzm(Y)
    @test size(Z) == (3, 4)
    @test maximum(abs, sum(Z; dims=1)) < 1e-12
    @test maximum(abs, sum(Z; dims=2)) < 1e-12
    @test Z ≈ circshift(stz_basis(2) * Y * stz_basis(3)', (-1, -1)) atol=1e-12
    @test stzm_free(Z) ≈ Y atol=1e-12
    @test stzm(zeros(0, 3)) == zeros(1, 4)
    # Stan's check: a free value of a non-zero-sum or empty argument is an error.
    @test_throws DomainError stz_free([1.0, 1.0])
    @test_throws DomainError stzm_free([1.0 0.0; 0.0 0.0])
    @test_throws ArgumentError stz_free(Float64[])

    # Stan's values, as generated quantities of a model that reads `mu`. (Stan
    # allows `sum_to_zero_jacobian` only in transformed parameters.)
    model = @slic (; y, Y, obs=0.2) begin
        mu ~ normal(0.0, 1.0)
        obs ~ normal(mu, 1.0)
        z = sum_to_zero_constrain(mu * y)
        free = sum_to_zero_unconstrain(z)
        Z = sum_to_zero_constrain(mu * Y)
        free_m = sum_to_zero_unconstrain(Z)
    end
    p = instantiate(stan_model(model))
    names = BridgeStan.param_names(p.model; include_tp=true, include_gq=true)
    values = BridgeStan.param_constrain(p.model, [1.0]; include_tp=true, include_gq=true,
        rng=BridgeStan.StanRNG(p.model, 1))
    stan(name, dims) = reshape(values[[startswith(s, name * ".") for s in names]], dims)
    @test stan("z", (5,)) ≈ z atol=1e-14
    @test stan("free", (4,)) ≈ stz_free(z) atol=1e-14
    @test stan("Z", (3, 4)) ≈ Z atol=1e-14
    @test stan("free_m", (2, 3)) ≈ stzm_free(Z) atol=1e-14
end

"""
Mooncake differentiates the Julia sum-to-zero methods. The transform is linear
and isometric, so the gradient of `w ⋅ f(y)` is the free form of `w`'s
projection onto the sum-to-zero values.
"""
@testitem "slic: @juliacompat sum_to_zero transforms are Mooncake differentiable" tags=[:slic, :ad] setup=[StanBlocksImports, SumToZeroShims] begin
    import Mooncake
    gradient(f, x...) = begin
        cache = Mooncake.prepare_gradient_cache(f, x...)
        Mooncake.value_and_gradient!!(cache, f, x...)
    end

    y, w = [0.3, -1.2, 0.8, 2.1], [0.5, -0.1, 0.7, 1.3, -0.2]
    value, (_, dy) = gradient(y -> sum(w .* stz(y)), y)
    @test value ≈ sum(w .* stz(y)) atol=1e-12
    @test dy ≈ stz_free(w .- Statistics.mean(w)) atol=1e-12
    # The free form reads `z[2:end]` only (the sum fixes `z[1]`), so its
    # gradient agrees with `stz(y)` along the sum-to-zero values.
    _, (_, dz) = gradient(z -> sum(y .* stz_free(z)), stz(y))
    @test dz[1] == 0.0
    @test dz .- Statistics.mean(dz) ≈ stz(y) atol=1e-12

    Y, W = [0.4 -0.3 1.1; 0.2 0.9 -0.7], [0.1 0.4 -0.6 0.2; 1.0 -0.3 0.5 0.0; 0.3 0.2 -0.1 0.8]
    centred = W .- Statistics.mean(W; dims=1) .- Statistics.mean(W; dims=2) .+ Statistics.mean(W)
    _, (_, dY) = gradient(Y -> sum(W .* stzm(Y)), Y)
    @test dY ≈ stzm_free(centred) atol=1e-12
end
