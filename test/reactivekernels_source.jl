@testitem "ReactiveKernels receives dispatched juliacompat source" tags=[:reactivekernels] begin
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
            @test success(pipeline(`$python $check`; stdout=stdout, stderr=stderr))
        end
    end

    @eval source_loop(x::Float64, n::Int) = x - n
    @test isnothing(ext.deffun_source(source_loop, Tuple{Float64,Int}))
    @test ext.deffun_source(source_loop, Tuple{Float32,Int}) isa TraceableSource
    @eval source_loop(x::Real, n::Integer) = x - 2n
    @test isnothing(ext.deffun_source(source_loop, Tuple{Float32,Int}))
end

@testitem "ReactiveKernels scalar extrema preserve NaN semantics" tags=[:reactivekernels] begin
    using ReactiveKernels
    @deffun @juliacompat source_extrema(x::real, y::real, maximum::int)::real = begin
        if maximum == 0
            fmin(x, y)
        else
            fmax(x, y)
        end
    end
    @kernel extrema_kernel(x::Float64, y::Float64, maximum::Int) = begin
        result = source_extrema(x, y, maximum)
        return result
    end
    src = jax_source(extrema_kernel; have=(:x, :y, :maximum), want=:result)
    host = jax_source(extrema_kernel; have=(:x, :y, :maximum), want=:result,
        data=(:x, :y, :maximum))
    for maximum in (0, 1), (x, y) in ((-2.0, 3.0), (3.0, -2.0),
            (NaN, 3.0), (3.0, NaN), (NaN, NaN), (-Inf, Inf))
        expected = isnan(x) ? y : isnan(y) ? x : maximum == 0 ? min(x, y) : max(x, y)
        @test isequal(source_extrema(x, y, maximum), expected)
    end
    python = get(ENV, "RK_JAX_PYTHON", "")
    if isempty(python)
        @test_skip "Set RK_JAX_PYTHON to check JAX values and gradients"
    else
        mktempdir() do dir
            write(joinpath(dir, "extrema_kernel.py"), src)
            write(joinpath(dir, "extrema_host.py"), host)
            check = joinpath(dir, "check.py")
            write(check, """
            import jax
            import numpy as np
            import extrema_kernel as model
            import extrema_host as host
            run = jax.jit(model.kernel)
            grad = jax.jit(jax.grad(model.kernel, argnums=(0, 1)))
            for maximum in (0, 1):
                for x, y in ((-2.0, 3.0), (3.0, -2.0),
                             (np.nan, 3.0), (3.0, np.nan), (np.nan, np.nan),
                             (-np.inf, np.inf)):
                    expected = np.fmin(x, y) if maximum == 0 else np.fmax(x, y)
                    np.testing.assert_allclose(run(x, y, maximum), expected)
                    prepared = host.prepare_data(x=x, y=y, maximum=maximum)
                    np.testing.assert_allclose(host.kernel(**prepared), expected)
                for x, y in ((-2.0, 3.0), (3.0, -2.0)):
                    first = (x < y) if maximum == 0 else (x > y)
                    np.testing.assert_allclose(grad(x, y, maximum),
                                               (float(first), float(not first)))
            """)
            @test success(pipeline(`$python $check`; stdout=stdout, stderr=stderr))
        end
    end
end

@testitem "ReactiveKernels matrix exponential compatibility" tags=[:reactivekernels] begin
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
            @test success(pipeline(`$python $check`; stdout=stdout, stderr=stderr))
        end
    end
end

@testitem "ReactiveKernels uses generated source in host data preparation" tags=[:reactivekernels] begin
    using ReactiveKernels
    @deffun @juliacompat source_host_loop(x::real, n::int)::real = begin
        total = 0.0
        for i in 1:n
            total = total + x * i
        end
        total
    end
    @kernel prepared_kernel(x::Float64, n::Int, scale::Float64) = begin
        offset = source_host_loop(x, n)
        result = scale * offset
        return result
    end
    src = jax_source(prepared_kernel; have=(:x, :n, :scale), want=:result,
        data=(; x=2.0, n=3))
    all_prepared = jax_source(prepared_kernel; have=(:x, :n, :scale), want=:result,
        data=(:x, :n, :scale))
    @test occursin("def prepare_data", string(src))
    python = get(ENV, "RK_JAX_PYTHON", "")
    if isempty(python)
        @test_skip "Set RK_JAX_PYTHON to check host preparation and JAX gradients"
    else
        mktempdir() do dir
            write(joinpath(dir, "prepared_kernel.py"), src)
            write(joinpath(dir, "all_prepared_kernel.py"), all_prepared)
            check = joinpath(dir, "check.py")
            write(check, """
            import jax
            import prepared_kernel as model
            import all_prepared_kernel as all_data
            prepared = model.prepare_data(x=2.0, n=3)
            assert abs(float(jax.jit(model.kernel)(3.0, **prepared)) - 36.0) < 1e-10
            assert abs(float(jax.jit(jax.grad(model.kernel))(3.0, **prepared)) - 12.0) < 1e-10
            for n in (0, 1, 19):
                prepared_all = all_data.prepare_data(x=2.0, n=n, scale=3.0)
                expected = 3.0 * n * (n + 1)
                assert abs(float(all_data.kernel(**prepared_all)) - expected) < 1e-10
                assert abs(float(jax.jit(all_data.kernel)(**prepared_all)) - expected) < 1e-10
            """)
            @test success(pipeline(`$python $check`; stdout=stdout, stderr=stderr))
        end
    end
end

@testitem "ReactiveKernels source preserves shared dimension diagnostics" tags=[:reactivekernels] begin
    using ReactiveKernels
    @deffun @juliacompat source_guarded_sum(x::vector[n], y::vector[n])::real = sum(x + y)
    @kernel guarded_kernel(x::Vector{Float64}, y::Vector{Float64}, scale::Float64) = begin
        offset = source_guarded_sum(x, y)
        result = scale * offset
        return result
    end
    @test source_guarded_sum([1.0, 2.0], [3.0, 4.0]) == 10.0
    @test_throws DimensionMismatch source_guarded_sum([1.0], [3.0, 4.0])
    device = jax_source(guarded_kernel; have=(:x, :y, :scale), want=:result)
    host = jax_source(guarded_kernel; have=(:x, :y, :scale), want=:result,
        data=(:x, :y))
    python = get(ENV, "RK_JAX_PYTHON", "")
    if isempty(python)
        @test_skip "Set RK_JAX_PYTHON to check dimension diagnostics"
    else
        mktempdir() do dir
            write(joinpath(dir, "guards.py"), device)
            write(joinpath(dir, "host_guards.py"), host)
            check = joinpath(dir, "check.py")
            write(check, """
            import jax
            import jax.numpy as jnp
            import numpy as np
            import guards, host_guards
            x, y = jnp.array([1., 2.]), jnp.array([3., 4.])
            assert float(jax.jit(guards.kernel)(x=x, y=y, scale=2.)) == 20.
            np.testing.assert_allclose(jax.grad(lambda xx: guards.kernel(x=xx, y=y, scale=2.))(x), [2., 2.])
            prepared = host_guards.prepare_data(x=np.asarray(x), y=np.asarray(y))
            assert float(host_guards.kernel(scale=2., **prepared)) == 20.
            for call in (lambda: jax.jit(guards.kernel)(x=x[:1], y=y, scale=2.),
                         lambda: host_guards.prepare_data(x=np.asarray(x[:1]), y=np.asarray(y))):
                try:
                    call()
                except Exception as error:
                    message = str(error)
                    assert 'dim mismatch' in message, message
                    assert 'inferred from' in message and 'sizes:' in message, message
                else:
                    raise AssertionError('mismatched dimensions were accepted')
            """)
            @test success(pipeline(`$python $check`; stdout=stdout, stderr=stderr))
        end
    end
end

@testitem "ReactiveKernels source supports typed indexed buffers" tags=[:reactivekernels] begin
    using ReactiveKernels
    @deffun @juliacompat source_recurrence(x::vector[n], rate::real)::vector[n] = begin
        y::vector[n]
        y[1] = x[1]
        for i in 2:n
            y[i] = rate * y[i - 1] + x[i]
        end
        y
    end
    @kernel recurrence_kernel(x::Vector{Float64}, rate::Float64, scale::Float64) = begin
        path = source_recurrence(x, rate)
        result = scale * sum(path)
        return result
    end
    @kernel recurrence_f32(x::Vector{Float32}, rate::Float32) = begin
        result = source_recurrence(x, rate)
        return result
    end
    @test source_recurrence([1.0, 2.0, 3.0], 0.5) == [1.0, 2.5, 4.25]
    @test eltype(source_recurrence(Float32[1, 2, 3], 0.5f0)) === Float32
    device = jax_source(recurrence_kernel; have=(:x, :rate, :scale), want=:result)
    host = jax_source(recurrence_kernel; have=(:x, :rate, :scale), want=:result,
        data=(:x, :rate))
    f32 = jax_source(recurrence_f32; have=(:x, :rate), want=:result)
    python = get(ENV, "RK_JAX_PYTHON", "")
    if isempty(python)
        @test_skip "Set RK_JAX_PYTHON to check indexed buffers"
    else
        mktempdir() do dir
            write(joinpath(dir, "recurrence.py"), device)
            write(joinpath(dir, "host_recurrence.py"), host)
            write(joinpath(dir, "recurrence_f32.py"), f32)
            check = joinpath(dir, "check.py")
            write(check, """
            import jax
            import jax.numpy as jnp
            import numpy as np
            import recurrence, host_recurrence, recurrence_f32
            for n in (3, 6, 40):
                x = np.arange(1., n + 1) / n
                for rate in (-0.3, 0.5, 0.8):
                    path, deriv = x.copy(), np.zeros_like(x)
                    for i in range(1, n):
                        path[i] = rate * path[i-1] + x[i]
                        deriv[i] = path[i-1] + rate * deriv[i-1]
                    before = x.copy()
                    np.testing.assert_allclose(jax.jit(recurrence.kernel)(x=jnp.asarray(x), rate=rate, scale=1.), np.sum(path), rtol=1e-10)
                    np.testing.assert_allclose(jax.grad(lambda r: recurrence.kernel(x=jnp.asarray(x), rate=r, scale=1.))(rate), np.sum(deriv), rtol=1e-10)
                    prepared = host_recurrence.prepare_data(x=x, rate=rate)
                    np.testing.assert_allclose(host_recurrence.kernel(scale=1., **prepared), np.sum(path), rtol=1e-10)
                    np.testing.assert_array_equal(x, before)
            f32 = jax.jit(recurrence_f32.kernel)(x=jnp.array([1., 2., 3.], dtype=jnp.float32), rate=jnp.float32(0.5))
            assert f32.dtype == jnp.float32
            np.testing.assert_allclose(f32, [1., 2.5, 4.25])
            """)
            @test success(pipeline(`$python $check`; stdout=stdout, stderr=stderr))
        end
    end
end

@testitem "ReactiveKernels matrix mapping preserves host inputs and rejects padding" tags=[:reactivekernels] begin
    using ReactiveKernels
    @deffun @juliacompat source_exponential(x) = matrix_exp(x)
    @kernel matrix_product(A::Matrix{Float64}, v::Vector{Float64}, scale::Float64) = begin
        E = source_exponential(A)
        result = scale * sum(E * v)
        return result
    end
    @kernel padded_matrix(rows::Vector{Vector{Float64}}) = begin
        cells = plate(rows) do x
            E = source_exponential(x * x')
            sum(E)
        end
        result = sum(cells)
        return result
    end
    @test_throws r"unpadded dense matrix" jax_source(padded_matrix;
        have=(:rows,), want=:result)
    device = jax_source(matrix_product; have=(:A, :v, :scale), want=:result)
    host = jax_source(matrix_product; have=(:A, :v, :scale), want=:result,
        data=(:A, :v))
    python = get(ENV, "RK_JAX_PYTHON", "")
    if isempty(python)
        @test_skip "Set RK_JAX_PYTHON to check matrix host preparation"
    else
        mktempdir() do dir
            write(joinpath(dir, "matrix.py"), device)
            write(joinpath(dir, "host_matrix.py"), host)
            check = joinpath(dir, "check.py")
            write(check, """
            import jax
            import jax.numpy as jnp
            import numpy as np
            from scipy.linalg import expm
            import matrix, host_matrix
            for n in (2, 3):
                A, v = -np.diag(np.arange(1., n + 1)), np.arange(2., n + 2)
                expected = 2. * np.sum(expm(A) @ v)
                np.testing.assert_allclose(jax.jit(matrix.kernel)(A=jnp.asarray(A), v=jnp.asarray(v), scale=2.), expected, rtol=1e-10)
                gradient = jax.grad(lambda a: matrix.kernel(A=a, v=jnp.asarray(v), scale=2.))(jnp.asarray(A))
                numerical = np.empty_like(A)
                for i in range(n):
                    for j in range(n):
                        delta = np.zeros_like(A)
                        delta[i, j] = 1e-6
                        numerical[i, j] = np.sum((expm(A + delta) - expm(A - delta)) @ v) / 1e-6
                np.testing.assert_allclose(gradient, numerical, rtol=1e-7, atol=1e-8)
                before_A, before_v = A.copy(), v.copy()
                prepared = host_matrix.prepare_data(A=A, v=v)
                assert all(isinstance(value, (np.ndarray, np.generic, float, int)) for value in prepared.values())
                np.testing.assert_array_equal(A, before_A)
                np.testing.assert_array_equal(v, before_v)
                np.testing.assert_allclose(host_matrix.kernel(scale=2., **prepared), expected, rtol=1e-10)
            """)
            @test success(pipeline(`$python $check`; stdout=stdout, stderr=stderr))
        end
    end
end
