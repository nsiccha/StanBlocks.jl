@testitem "ReactiveKernels receives dispatched juliacompat source" begin
    using ReactiveKernels

    @deffun @juliacompat source_loop(x::real, n::int)::real = begin
        total = 0.0
        for i in 1:n
            total = total + x * i
        end
        if x > 0
            total = total * x
        end
        total
    end
    @deffun @juliacompat source_loop(x::vector[n], k::int)::real = sum(x) + k

    ext = Base.get_extension(StanBlocks, :StanBlocksReactiveKernelsExt)
    @test !isnothing(ext)
    scalar = ext.deffun_source(source_loop, Tuple{Float64,Int})
    vector = ext.deffun_source(source_loop, Tuple{Vector{Float64},Int})
    @test scalar isa TraceableSource
    @test vector isa TraceableSource
    @test scalar !== vector
    @test isnothing(ext.deffun_source(sin, Tuple{Float64}))
    @test source_loop(2.0, 3) == 24.0

    @kernel source_kernel(x::Float64) = begin
        result = source_loop(x, 3)
        return result
    end
    src = jax_source(source_kernel; have=(:x,), want=:result)
    @test occursin("lax.fori_loop", string(src))
    @test occursin("lax.cond", string(src))

    python = get(ENV, "RK_JAX_PYTHON", "")
    if isempty(python)
        @test_skip "Set RK_JAX_PYTHON to check JAX values and gradients"
    else
        mktempdir() do dir
            write(joinpath(dir, "source_kernel.py"), src)
            check = joinpath(dir, "check.py")
            write(check, """
            import jax
            import source_kernel as model
            for x, expected, gradient in ((2.0, 24.0, 24.0), (-2.0, -12.0, 6.0)):
                assert abs(float(jax.jit(model.kernel)(x)) - expected) < 1e-10
                assert abs(float(jax.jit(jax.grad(model.kernel))(x)) - gradient) < 1e-10
            """)
            @test success(`$python $check`)
        end
    end

    @eval source_loop(x::Float64, n::Int) = x - n
    @test isnothing(ext.deffun_source(source_loop, Tuple{Float64,Int}))
    @test ext.deffun_source(source_loop, Tuple{Float32,Int}) isa TraceableSource
    @eval source_loop(x::Real, n::Integer) = x - 2n
    @test isnothing(ext.deffun_source(source_loop, Tuple{Float32,Int}))
end

@testitem "ReactiveKernels matrix exponential compatibility" begin
    using ReactiveKernels
    @deffun @juliacompat source_matrix_exp(x) = matrix_exp(x)
    @kernel matrix_kernel(x::Matrix{Float64}) = begin
        result = sum(source_matrix_exp(x))
        return result
    end
    src = jax_source(matrix_kernel; have=(:x,), want=:result)
    @test occursin("jax.scipy.linalg.expm", string(src))
    python = get(ENV, "RK_JAX_PYTHON", "")
    if isempty(python)
        @test_skip "Set RK_JAX_PYTHON to check JAX values and gradients"
    else
        mktempdir() do dir
            write(joinpath(dir, "matrix_kernel.py"), src)
            check = joinpath(dir, "check.py")
            write(check, """
            import jax
            import jax.numpy as jnp
            import numpy as np
            import matrix_kernel as model
            a, b = -1.0, -2.0
            x = jnp.diag(jnp.array([a, b]))
            ea, eb = np.exp(a), np.exp(b)
            divided = (ea - eb) / (a - b)
            assert abs(float(jax.jit(model.kernel)(x)) - ea - eb) < 1e-10
            np.testing.assert_allclose(jax.jit(jax.grad(model.kernel))(x),
                [[ea, divided], [divided, eb]], rtol=1e-9, atol=1e-10)
            """)
            @test success(`$python $check`)
        end
    end
end
