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
        # A system tag and a vararg tail passed through a walk: Julia only
        # specializes the method on them because the lowering types them.
        st_rate(::typeof(st_system), a, b) = a * b
        st_tag_walk(system, x::vector[n], theta...) = begin
            state = rep_vector(0., st_size_of(system))
            for i in 1:n
                state[1] += st_rate(system, theta...) * x[i]
            end
            state
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
    @test st_tag_walk(st_system, [1.0, 2.0], 2.0, 3.0) isa SVector{3,Float64}
    @test st_tag_walk(st_system, [1.0, 2.0], 2.0, 3.0) == [18.0, 0.0, 0.0]
    @test allocated(x -> st_tag_walk(st_system, x, 2.0, 3.0), [1.0, 2.0]) == 0
    @test all(Base.specializations(only(methods(st_tag_walk)))) do mi
        !(mi isa Core.MethodInstance) || (mi.specTypes.parameters[2] !== Function &&
            !Base.isvarargtype(mi.specTypes.parameters[end]))
    end
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

@testsnippet JuliaCompatStaticSlices begin
    using StanBlocks
    import StanBlocks.stan: @deffun
    using StanBlocks.StaticArrays: SVector, SMatrix
    @deffun @juliacompat begin
        # Slices whose bounds are known when the method compiles: literals, a
        # dimension of a static argument, `end`, and arithmetic of those.
        sl_block(G::matrix[n, n])::matrix[2, 2] = G[1:2, 1:2]
        sl_corner(G::matrix[n, n]) = G[2:end, (n - 1):n]
        sl_column(G::matrix[n, n]) = G[1:2, 2]
        sl_nested(x::vector[n])::real = sum(x[1:3]) + x[2:end][1]
        sl_sequence(x::vector[n]) = head(x, 2) + tail(x, 2) + segment(x, 2, 2)
        # A slice of a local rebuilt in a branch keeps the local's type.
        sl_dose(P::matrix[n, n], state::vector[n], amount, count::int)::vector[n] = begin
            rv::vector[n]
            rv .= state
            rv[1] += amount
            if count > 1
                affine = rep_matrix(0., n + 1, n + 1)
                for i in 1:n
                    for j in 1:n
                        affine[i, j] = P[i, j]
                    end
                end
                affine[1, n + 1] = amount
                affine[n + 1, n + 1] = 1.
                augmented = matrix_power(affine, count - 1) * append_row(rv, 1.)
                rv = augmented[1:n]
            end
            rv
        end
        # A literal block of a larger generator, every step and every read.
        sl_step(A::matrix[3, 3], dt)::matrix[3, 3] = begin
            P = rep_matrix(0., 3, 3)
            for i in 1:3
                P[i, i] = 1.
            end
            if dt > 0
                P = matrix_exp(A * dt)
            end
            P
        end
        sl_sites(G::matrix[n, n], dt)::matrix[n, n] = begin
            P = rep_matrix(0., n, n)
            for i in 1:n
                P[i, i] = exp(G[i, i] * dt)
            end
            A = G[1:3, 1:3]
            E = sl_step(A, dt)
            for i in 1:3
                for j in 1:3
                    P[i, j] = E[i, j]
                end
            end
            P
        end
        sl_walk(G::matrix[n, n], dts::vector[m])::real = begin
            g = [1., 2., 3.]
            state = rep_vector(0., n)
            state[1] = 1.
            acc = 0.
            for t in 1:m
                state = sl_sites(G, dts[t]) * state
                acc += g' * state[1:3]
            end
            acc
        end
        # Bounds known only at run time, and a slice past the static limit.
        sl_runtime(x::vector[n], m::int) = x[1:m]
        sl_large(x::vector[n]) = x[1:101]
        sl_square(G::matrix[n, n]) = G[1:11, 1:11]
    end

    # The same walk in plain Julia on dense arrays.
    function ref_sites(G, dt)
        P = Matrix(StanBlocks.LinearAlgebra.Diagonal(exp.(StanBlocks.LinearAlgebra.diag(G) .* dt)))
        P[1:3, 1:3] = dt > 0 ? exp(G[1:3, 1:3] * dt) : one(G[1:3, 1:3])
        P
    end
    function ref_slice_walk(G, dts)
        state = zeros(size(G, 1))
        state[1] = 1.0
        acc = 0.0
        for dt in dts
            state = ref_sites(G, dt) * state
            acc += [1.0, 2.0, 3.0]' * state[1:3]
        end
        acc
    end
    allocated(f::F, args::Vararg{Any,N}) where {F,N} = (f(args...); @allocated f(args...))
    # A four-state generator: a three-state block and a fourth, decaying site.
    slice_generator(a, b, c) = SMatrix{4,4}(-a, a, 0.0, 0.0, 0.0, -b, b, b / 2, 0.0, 0.0, -c, 0.0, 0.0, 0.0, 0.0, -0.3)
end

"""
A `@juliacompat` slice whose bounds are known when its Julia method compiles,
`x[1:3]`, `G[1:3, 1:3]`, `x[1:n]` with `n` a dimension of a static argument,
`x[2:end]` of a static `x`, or `head`/`tail`/`segment` with literal sizes, is
a static `SVector`/`SMatrix` (at most 100 elements), whether `x` is static or
dense (snag `juliacompat-stat-75c9ba31`). Before, every such slice built a
dense array on every call.
"""
@testitem "slic: @juliacompat slices with compile-time bounds are static" tags=[:slic, :regression] setup=[StanBlocksImports, JuliaCompatStaticSlices] begin
    G = slice_generator(0.7, 0.4, 0.2)
    D = Matrix(G)
    x = SVector(1.0, 2.0, 3.0, 4.0, 5.0)
    dts = collect(range(0.05, 0.3; length=10))

    # Literal bounds give a static slice of a static and of a dense argument.
    @test sl_block(G) isa SMatrix{2,2,Float64}
    @test sl_block(D) isa SMatrix{2,2,Float64}
    @test sl_block(D) == D[1:2, 1:2]
    @test allocated(sl_block, D) == 0
    @test sl_column(D) isa SVector{2,Float64}
    @test sl_column(D) == D[1:2, 2]
    # `end` and a signature dimension are static along a static axis only.
    @test sl_corner(G) isa SMatrix{3,2,Float64}
    @test sl_corner(G) == D[2:end, 3:4]
    @test sl_corner(D) isa Matrix{Float64}
    @test sl_corner(D) == D[2:end, 3:4]
    @test sl_nested(x) == 6.0 + 2.0
    @test allocated(sl_nested, x) == 0
    @test sl_sequence(x) isa SVector{2,Float64}
    @test sl_sequence(x) == [1.0 + 4.0 + 2.0, 2.0 + 5.0 + 3.0]
    @test sl_sequence(Vector(x)) == sl_sequence(x)
    @test allocated(sl_sequence, x) == 0

    # A slice reassigned to a static local keeps it static: the method infers
    # one concrete type and allocates nothing.
    P = SMatrix{2,2}(0.9, 0.1, 0.0, 0.8)
    s = SVector(1.0, 0.0)
    @test sl_dose(P, s, 2.0, 3) isa SVector{2,Float64}
    @test sl_dose(P, s, 2.0, 3) ≈ P * (P * [3.0, 0.0] + [2.0, 0.0]) + [2.0, 0.0] rtol=1e-14
    @test sl_dose(P, s, 2.0, 1) == [3.0, 0.0]
    @test Base.return_types(sl_dose, (typeof(P), typeof(s), Float64, Int)) == [SVector{2,Float64}]
    @test allocated(sl_dose, P, s, 2.0, 3) == 0

    # A literal block of a static or a dense generator, every step and every
    # read: the walk infers `Float64`, and allocates nothing when the
    # generator is static.
    @test sl_sites(G, 0.1) isa SMatrix{4,4,Float64}
    @test sl_sites(G, 0.1) ≈ ref_sites(D, 0.1) rtol=1e-13
    @test sl_walk(G, dts) ≈ ref_slice_walk(D, dts) rtol=1e-13
    @test sl_walk(D, dts) ≈ ref_slice_walk(D, dts) rtol=1e-13
    @test Base.return_types(sl_walk, (typeof(G), Vector{Float64})) == [Float64]
    @test Base.return_types(sl_walk, (Matrix{Float64}, Vector{Float64})) == [Float64]
    @test Base.return_types(sl_sites, (Matrix{Float64}, Float64)) == [Matrix{Float64}]
    @test allocated(sl_walk, G, dts) == 0
    @test allocated(sl_sites, G, 0.1) == 0

    # Run-time bounds, and slices past the static limit, stay dense.
    @test sl_runtime(x, 2) isa Vector{Float64}
    @test sl_runtime(x, 2) == [1.0, 2.0]
    @test sl_large(collect(1.0:120.0)) isa Vector{Float64}
    @test sl_large(collect(1.0:120.0)) == collect(1.0:101.0)
    @test sl_square(rand(12, 12)) isa Matrix{Float64}

    # Stan's range checks still hold.
    @test_throws BoundsError sl_block(SMatrix{1,1}(1.0))
    @test_throws BoundsError sl_block(ones(1, 1))
    @test_throws BoundsError sl_sequence(SVector(1.0, 2.0))
    @test_throws BoundsError sl_nested(SVector(1.0, 2.0))

    # The source recorded for ReactiveKernels' JAX translation keeps the dense
    # slice; the method itself reads the static one.
    method = which(sl_walk, Tuple{Matrix{Float64},Vector{Float64}})
    mentions(x::Expr, name) = any(a -> mentions(a, name), x.args)
    mentions(x::QuoteNode, name) = x.value === name
    mentions(x::GlobalRef, name) = x.name === name
    mentions(x, name) = x === name
    source = StanBlocks._deffun_julia_source(method.sig)
    @test !isnothing(source)
    @test !any(name -> mentions(source[3], name), (:_julia_getindex, :_julia_static_range))
    @test any(st -> mentions(st, :_julia_getindex), Base.uncompressed_ir(method).code)
end

"""
Enzyme differentiates a walk through static slices, and the gradient matches
finite differences of the dense reference.
"""
@testitem "slic: @juliacompat static slices are Enzyme differentiable" tags=[:slic, :ad, :regression] setup=[StanBlocksImports, EnzymeGradient, JuliaCompatStaticSlices] begin
    dts = collect(range(0.05, 0.3; length=10))
    walk(x) = sl_walk(slice_generator(x[1], x[2], x[3]), dts)
    x = [0.7, 0.4, 0.2]
    value, (dx,) = gradient(walk, x)
    @test value ≈ ref_slice_walk(Matrix(slice_generator(x...)), dts) rtol=1e-13
    h = 1e-6
    fd = map(1:3) do i
        e = zeros(3)
        e[i] = h
        (ref_slice_walk(Matrix(slice_generator((x + e)...)), dts) -
         ref_slice_walk(Matrix(slice_generator((x - e)...)), dts)) / 2h
    end
    @test dx ≈ fd rtol=1e-6
end

"""
The emitted Stan of the slicing helpers and their Julia methods agree:
BridgeStan evaluates the walk as a transformed parameter.
"""
@testitem "slic: static slices in @juliacompat helpers match their emitted Stan" tags=[:slic, :regression, :stanc, :bridgestan] setup=[StanBlocksImports, StanBlocksTestSetup, JuliaCompatStaticSlices] begin
    using .StanBlocksTestSetup: stanc_compiles
    dts = collect(range(0.05, 0.3; length=10))
    G = Matrix(slice_generator(0.7, 0.4, 0.2))
    model = @slic (; G, dts, obs=0.2) begin
        mu ~ normal(0.0, 1.0)
        obs ~ normal(mu, 1.0)
        walked = sl_walk(exp(mu) * G, dts)
        dosed = sum(sl_dose(sl_block(G), [1.0, 0.0], exp(mu), 3))
    end
    @test stanc_compiles(model)
    p = instantiate(stan_model(model))
    names = BridgeStan.param_names(p.model; include_tp=true, include_gq=true)
    mu = -0.3
    draw = BridgeStan.param_constrain(p.model, [mu]; include_tp=true, include_gq=true,
        rng=BridgeStan.StanRNG(p.model, 1))
    @test only(draw[names .== "walked"]) ≈ sl_walk(exp(mu) * G, dts) rtol=1e-13
    @test only(draw[names .== "dosed"]) ≈ sum(sl_dose(sl_block(G), [1.0, 0.0], exp(mu), 3)) rtol=1e-13
end
