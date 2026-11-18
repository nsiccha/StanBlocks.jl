@testsnippet JuliaCompatStaticFixtures begin
    using StanBlocks
    import StanBlocks.stan: @deffun
    using StanBlocks.StaticArrays: SVector, SMatrix, SOneTo
    @deffun @juliacompat begin
        # A fixed 3x3 generator built from vector literals.
        st_generator(a, b, c)::matrix[3, 3] = append_col([-a, a, 0.], [0., -b, b], [0., 0., -c])
        # A vector[12] of constants filled by index.
        st_constants(A::matrix[3, 3])::vector[12] = begin
            k = rep_vector(0., 12)
            for i in 1:3
                for j in 1:3
                    k[3 * (i - 1) + j] = A[i, j]
                end
            end
            k[10] = A[1, 1] + A[2, 2]
            k[11] = A[2, 2] + A[3, 3]
            k[12] = A[1, 1] * A[3, 3]
            k
        end
        # A 3x3 step matrix filled by index, with an updating write.
        st_step(A::matrix[3, 3], k::vector[n_k], dt)::matrix[3, 3] = begin
            P = rep_matrix(0., 3, 3)
            for i in 1:3
                for j in 1:3
                    P[i, j] = dt * A[i, j] + 0.5 * dt * dt * k[3 * (i - 1) + j]
                end
                P[i, i] += 1.
            end
            P
        end
        # Sizes that follow from a static argument.
        st_augment(G::matrix[n, n], r)::matrix[n + 1, n + 1] =
            append_row(append_col(G, rep_vector(0., n)), to_row_vector([r, r, r, -r]))
        # An `int` helper that returns a constant sizes its callers' containers.
        st_n_states(a)::int = 3
        st_walk(a, b, c, dts::vector[m])::real = begin
            G = st_generator(a, b, c)
            k = st_constants(G)
            state = rep_vector(0., st_n_states(a))
            state[1] = 1.
            for t in 1:m
                state = st_step(G, k, dts[t]) * state
            end
            H = st_augment(G, a)
            sum(state) + H[4, 4]
        end
    end
    # Julia-only shapes of the static lowering.
    st_system() = nothing
    @deffun @juliacompat begin
        # An `int` helper that dispatches on a system's type, as a tag.
        st_size_of(::typeof(st_system))::int = 3
        st_start(system, x) = begin
            state = rep_vector(0., st_size_of(system))
            state[1] = x
            state
        end
        # A static local reassigned from `matrix_exp` stays static.
        st_propagator(G::matrix[n, n], dt)::matrix[n, n] = begin
            P = rep_matrix(0., n, n)
            for i in 1:n
                P[i, i] = 1.
            end
            if dt > 0
                P = matrix_exp(G * dt)
            end
            P
        end
        st_dose(x::vector[n], amount)::vector[n + 1] = begin
            augmented = append_row(x, 1.)
            augmented[1] += amount
            augmented
        end
        st_power(P::matrix[n, n], count::int)::matrix[n, n] = matrix_power(P, count)
        # `rows` of a static value, a local assigned once, a typed local, `.=`
        # and `end`.
        st_locals(G::matrix[n, n])::vector[n] = begin
            r = rows(G)
            diagonal::vector[r]
            diagonal .= 0.
            for i in 1:r
                diagonal[i] = G[i, i]
            end
            out = rep_vector(1., r + 0)
            out .= diagonal
            out[end] += 10.
            out
        end
        # Slice writes, and a matrix refilled from a vector.
        st_columns(v::vector[4])::matrix[2, 3] = begin
            M = to_matrix(v, 2, 2)
            M[:, 1] = rep_vector(0., 2)
            M[1, :] .= 5.
            append_col(M, rep_matrix(v[1], 2, 1))
        end
        st_repeat(v::vector[n])::matrix[n, 2] = rep_matrix(v, 2)
        # A written local that a comprehension also reads.
        st_captured(x::vector[n])::real = begin
            acc = rep_vector(0., 3)
            acc[1] = x[1]
            total = sum([acc[1] * i for i in 1:3])
            acc[2] = total
            acc[1] + acc[2]
        end
        # A size known only at run time, and a literal size past the limit.
        st_runtime(x::vector[n], m::int) = rep_vector(x[1], m)
        st_large(x) = rep_vector(x, 101)
    end

    # The same computations in plain Julia on dense arrays.
    ref_generator(a, b, c) = [-a 0.0 0.0; a -b 0.0; 0.0 b -c]
    ref_constants(A) = [vec(permutedims(A)); A[1, 1] + A[2, 2]; A[2, 2] + A[3, 3]; A[1, 1] * A[3, 3]]
    function ref_step(A, k, dt)
        P = [dt * A[i, j] + 0.5 * dt * dt * k[3 * (i - 1) + j] for i in 1:3, j in 1:3]
        P + one(P)
    end
    ref_augment(G, r) = [G zeros(size(G, 1)); fill(r, size(G, 1))' -r]
    function ref_walk(a, b, c, dts)
        G = ref_generator(a, b, c)
        k = ref_constants(G)
        state = [1.0, 0.0, 0.0]
        for dt in dts
            state = ref_step(G, k, dt) * state
        end
        sum(state) + ref_augment(G, a)[4, 4]
    end
    # Bytes one call allocates. The signature specializes on the function and
    # the arguments; without that, the call boxes them and its result.
    allocated(f::F, args::Vararg{Any,N}) where {F,N} = (f(args...); @allocated f(args...))
end

"""
A `@juliacompat` body whose container sizes are known when its Julia method
compiles runs on StaticArrays and does not allocate (snag
`juliacompat-fixe-b2a30659`). Such a size is an integer literal, a dimension of
a static-size argument (or `rows`/`cols`/`num_elements` of a static value), a
local assigned once from such a size, or what an `int` helper returns from one.
Vector literals of numbers are static vectors, and a written local is rebuilt
rather than mutated. Before, every call allocated dense arrays.
"""
@testitem "slic: @juliacompat bodies with compile-time sizes run on StaticArrays" tags=[:slic, :regression] setup=[StanBlocksImports, JuliaCompatStaticFixtures] begin
    a, b, c = 0.7, 0.4, 0.2
    dts = collect(range(0.05, 0.3; length=10))
    G = st_generator(a, b, c)
    k = st_constants(G)
    @test G isa SMatrix{3,3,Float64}
    @test k isa SVector{12,Float64}
    @test st_step(G, k, 0.1) isa SMatrix{3,3,Float64}
    @test st_augment(G, a) isa SMatrix{4,4,Float64}
    @test G == ref_generator(a, b, c)
    @test k == ref_constants(ref_generator(a, b, c))
    @test st_step(G, k, 0.1) ≈ ref_step(Matrix(G), Vector(k), 0.1) rtol=1e-15
    @test st_augment(G, a) == ref_augment(Matrix(G), a)
    @test st_walk(a, b, c, dts) ≈ ref_walk(a, b, c, dts) rtol=1e-14

    # The whole chain infers and allocates nothing.
    @test Base.return_types(st_walk, (Float64, Float64, Float64, Vector{Float64})) == [Float64]
    @test allocated(st_walk, a, b, c, dts) == 0
    @test allocated(st_step, G, k, 0.1) == 0
    @test allocated(st_constants, G) == 0

    # An `int` helper's constant reaches a size position as a static size.
    @test StanBlocks._julia_size_twin(st_n_states, a) === SOneTo(3)
    @test st_n_states(a) === 3

    # Literal sizes are static whatever the arguments are; sizes that follow
    # from a dense argument stay dense, as before.
    @test st_constants(Matrix(G)) isa SVector{12,Float64}
    @test st_step(Matrix(G), Vector(k), 0.1) isa SMatrix{3,3,Float64}
    @test st_augment(Matrix(G), a) isa Matrix{Float64}
    @test st_augment(Matrix(G), a) == ref_augment(Matrix(G), a)

    # A tag-dispatched `int` helper, `matrix_exp`, `append_row` of a scalar
    # and `matrix_power` keep static sizes.
    @test st_start(st_system, 2.0) isa SVector{3,Float64}
    @test st_start(st_system, 2.0) == [2.0, 0.0, 0.0]
    # Through a closure: a function passed among the arguments is not specialized on.
    @test allocated(x -> st_start(st_system, x), 2.0) == 0
    @test st_propagator(G, 0.1) isa SMatrix{3,3,Float64}
    @test st_propagator(G, 0.1) ≈ StanBlocks.LinearAlgebra.exp(0.1 * Matrix(G)) rtol=1e-13
    @test st_propagator(G, -1.0) == [1.0 0.0 0.0; 0.0 1.0 0.0; 0.0 0.0 1.0]
    @test allocated(st_propagator, G, 0.1) == 0
    @test st_propagator(Matrix(G), 0.1) isa Matrix{Float64}
    @test st_dose(SVector(1.0, 2.0, 3.0), 0.5) isa SVector{4,Float64}
    @test st_dose(SVector(1.0, 2.0, 3.0), 0.5) == [1.5, 2.0, 3.0, 1.0]
    @test st_power(G, 3) isa SMatrix{3,3,Float64}
    @test st_power(G, 3) ≈ Matrix(G)^3 rtol=1e-14

    # `rows`, a local assigned once, a typed local, `.=`, `end` and slices.
    expected = [G[1, 1], G[2, 2], G[3, 3] + 10]
    @test st_locals(G) isa SVector{3,Float64}
    @test st_locals(G) == expected
    @test st_locals(Matrix(G)) isa Vector{Float64}
    @test st_locals(Matrix(G)) == expected
    @test allocated(st_locals, G) == 0
    v = SVector(1.0, 2.0, 3.0, 4.0)
    @test st_columns(v) isa SMatrix{2,3,Float64}
    @test st_columns(v) == [5.0 5.0 1.0; 0.0 4.0 1.0]
    @test st_columns(Vector(v)) isa Matrix{Float64}
    @test st_columns(Vector(v)) == [5.0 5.0 1.0; 0.0 4.0 1.0]
    @test st_repeat(SVector(1.0, 2.0)) isa SMatrix{2,2,Float64}
    @test st_repeat(SVector(1.0, 2.0)) == [1.0 1.0; 2.0 2.0]
    @test st_repeat([1.0, 2.0]) isa Matrix{Float64}
    @test st_captured([2.0]) == 2.0 + 12.0

    # A size known only at run time, or past the static limit, stays dense.
    @test st_runtime([1.5], 3) isa Vector{Float64}
    @test st_runtime([1.5], 3) == fill(1.5, 3)
    @test st_large(1.0) isa Vector{Float64}
    @test length(st_large(1.0)) == 101

    # Stan's checks still hold: a mismatched static argument and an
    # out-of-range write raise as before.
    @test_throws DimensionMismatch st_constants(SMatrix{2,2}(0.0, 0.0, 0.0, 0.0))
    @test_throws BoundsError StanBlocks._deffun_julia_setindex!!(SVector(1.0, 2.0), 0.0, 3)

    # The method's recorded source, which ReactiveKernels translates to JAX,
    # keeps the dense lowering; the method itself runs the static one.
    step_method = which(st_step, Tuple{Matrix{Float64},Vector{Float64},Float64})
    static_names = (:_deffun_julia_setindex!!, :_JuliaStaticSize, :_deffun_julia_vect, :_julia_size_call)
    mentions(x::Expr, name) = any(a -> mentions(a, name), x.args)
    mentions(x::QuoteNode, name) = x.value === name
    mentions(x::GlobalRef, name) = x.name === name
    mentions(x, name) = x === name
    source = StanBlocks._deffun_julia_source(step_method.sig)
    @test !isnothing(source)
    @test !any(name -> mentions(source[3], name), static_names)
    lowered = Base.uncompressed_ir(step_method).code
    @test any(st -> mentions(st, :_deffun_julia_setindex!!), lowered)
end

"""
Enzyme differentiates the static lowering, and the gradient matches finite
differences of the dense reference.
"""
@testitem "slic: @juliacompat static-size bodies are Enzyme differentiable" tags=[:slic, :ad, :regression] setup=[StanBlocksImports, EnzymeGradient, JuliaCompatStaticFixtures] begin
    dts = collect(range(0.05, 0.3; length=10))
    walk(x) = st_walk(x[1], x[2], x[3], dts)
    x = [0.7, 0.4, 0.2]
    value, (dx,) = gradient(walk, x)
    @test value ≈ ref_walk(x..., dts) rtol=1e-14
    h = 1e-6
    fd = map(1:3) do i
        e = zeros(3)
        e[i] = h
        (ref_walk((x + e)..., dts) - ref_walk((x - e)..., dts)) / 2h
    end
    @test dx ≈ fd rtol=1e-6
end

"""
The emitted Stan of the static-size helpers and their Julia methods agree:
BridgeStan evaluates the chain as a transformed parameter.
"""
@testitem "slic: static-size @juliacompat helpers match their emitted Stan" tags=[:slic, :regression, :stanc, :bridgestan] setup=[StanBlocksImports, StanBlocksTestSetup, JuliaCompatStaticFixtures] begin
    using .StanBlocksTestSetup: stanc_compiles
    dts = collect(range(0.05, 0.3; length=10))
    model = @slic (; dts, obs=0.2) begin
        mu ~ normal(0.0, 1.0)
        obs ~ normal(mu, 1.0)
        walked = st_walk(exp(mu), 0.4, 0.2, dts)
    end
    @test stanc_compiles(model)
    p = instantiate(stan_model(model))
    names = BridgeStan.param_names(p.model; include_tp=true, include_gq=true)
    mu = -0.3
    draw = BridgeStan.param_constrain(p.model, [mu]; include_tp=true, include_gq=true,
        rng=BridgeStan.StanRNG(p.model, 1))
    @test only(draw[names .== "walked"]) ≈ st_walk(exp(mu), 0.4, 0.2, dts) rtol=1e-13
end
