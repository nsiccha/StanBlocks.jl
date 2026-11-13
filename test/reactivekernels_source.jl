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

@testitem "ReactiveKernels translates Stan's log_sum_exp" tags=[:reactivekernels] begin
    # Snag `jax-source-has-n-a3964196`: the Julia twin's special cases are
    # branches, so `jax_source` refused any helper calling `log_sum_exp`.
    using ReactiveKernels
    @deffun @juliacompat begin
        source_log_sum_exp(x::real, y::real)::real = log_sum_exp(x, y)
        source_log_sum_exp(v::vector[n])::real = log_sum_exp(v)
        source_log_sum_exp(u::vector[n], v::vector[n])::vector[n] = log_sum_exp(u, v)
        source_soft_maximum(x::real, a::real, s::real)::real = s * log_sum_exp(x / s, a / s)
    end
    @kernel lse_pair(x::Float64, y::Float64) = begin
        result = source_log_sum_exp(x, y)
        return result
    end
    @kernel lse_vector(v::Vector{Float64}) = begin
        result = source_log_sum_exp(v)
        return result
    end
    @kernel lse_elementwise(u::Vector{Float64}, v::Vector{Float64}) = begin
        result = source_log_sum_exp(u, v)
        return result
    end
    @kernel lse_cells(x::Vector{Float64}, a::Vector{Float64}, s::Float64) = begin
        result = sum(source_soft_maximum.(x, a, s))
        return result
    end
    @kernel lse_rows(rows::Vector{Vector{Float64}}) = begin
        cells = plate(rows) do x
            source_log_sum_exp(x)
        end
        result = sum(cells)
        return result
    end
    sources = Dict(name => string(src) for (name, src) in (
        "lse_pair" => jax_source(lse_pair; have=(:x, :y), want=:result),
        "lse_pair_host" => jax_source(lse_pair; have=(:x, :y), want=:result, data=(:x, :y)),
        "lse_vector" => jax_source(lse_vector; have=(:v,), want=:result),
        "lse_elementwise" => jax_source(lse_elementwise; have=(:u, :v), want=:result),
        "lse_cells" => jax_source(lse_cells; have=(:x, :a, :s), want=:result),
        "lse_rows" => jax_source(lse_rows; have=(:rows,), want=:result, data=(:rows,))))
    @test occursin("_rk_logaddexp", sources["lse_pair"])
    @test occursin("_rk_logsumexp", sources["lse_vector"])

    # Expected values are the Julia twin's, Stan Math's special cases included.
    py(x::Real) = isnan(x) ? "nan" : isinf(x) ? (x > 0 ? "inf" : "-inf") : repr(Float64(x))
    py(x::AbstractVector) = "[" * join(py.(x), ", ") * "]"
    py(x::AbstractString) = x
    pairs = [(1.0, 2.0), (2.0, 1.0), (0.5, 0.5), (-Inf, 3.0), (3.0, -Inf), (-Inf, -Inf),
             (Inf, Inf), (Inf, 1.0), (NaN, 1.0), (1.0, NaN), (-800.0, -801.0), (700.0, 710.0)]
    vectors = [[1.0, 2.0, 3.0], Float64[], [-Inf, -Inf], [Inf, 1.0], [-Inf, 0.0],
               [NaN, 1.0], [-1000.0, -1000.5]]
    u, v = [1.0, -Inf, 0.5, 40.0], [2.0, 3.0, 0.5, -Inf]
    x, a, s = [0.1, 0.4, 2.0], [0.0, 1.5, 30.0], 0.2
    rows = [[0.5], [1.0, 2.0, 3.0], [-1.0, 0.25]]
    for (p, q) in pairs
        @test isequal(source_log_sum_exp(p, q), StanBlocks._julia_log_sum_exp(p, q))
    end
    @test source_log_sum_exp(u, v) == StanBlocks._julia_log_sum_exp.(u, v)
    expected = (
        pairs = "[" * join(("($(py(p)), $(py(q)), $(py(source_log_sum_exp(p, q))))"
                            for (p, q) in pairs), ", ") * "]",
        vectors = "[" * join(("($(py(z)), $(py(source_log_sum_exp(z))))" for z in vectors), ", ") * "]",
        elementwise = py(source_log_sum_exp(u, v)),
        cells = py(sum(source_soft_maximum.(x, a, s))),
        rows = py(sum(source_log_sum_exp.(rows))))
    python = get(ENV, "RK_JAX_PYTHON", "")
    if isempty(python)
        @test_skip "Set RK_JAX_PYTHON to check log_sum_exp values and gradients"
    else
        mktempdir() do dir
            for (name, src) in sources
                write(joinpath(dir, name * ".py"), src)
            end
            check = joinpath(dir, "check.py")
            write(check, """
            import jax
            import jax.numpy as jnp
            import numpy as np
            from numpy import inf, nan
            import lse_pair, lse_pair_host, lse_vector, lse_elementwise, lse_cells, lse_rows
            pair = jax.jit(lse_pair.kernel)
            pair_grad = jax.jit(jax.grad(lse_pair.kernel, argnums=(0, 1)))
            for x, y, expected in $(expected.pairs):
                np.testing.assert_allclose(pair(x, y), expected, rtol=1e-14)
                prepared = lse_pair_host.prepare_data(x=x, y=y)
                np.testing.assert_allclose(lse_pair_host.kernel(**prepared), expected, rtol=1e-14)
                if np.isfinite(expected) and not np.isnan(x) and not np.isnan(y):
                    # exp(x - result) and exp(y - result), 1/2 each at a tie,
                    # and no NaN beside a -inf argument.
                    np.testing.assert_allclose(pair_grad(x, y),
                        (np.exp(x - expected), np.exp(y - expected)), rtol=1e-12, atol=1e-300)
            vector = jax.jit(lse_vector.kernel)
            vector_grad = jax.jit(jax.grad(lse_vector.kernel))
            for v, expected in $(expected.vectors):
                v = jnp.asarray(v, dtype=jnp.float64)
                np.testing.assert_allclose(vector(v), expected, rtol=1e-14)
                if np.isfinite(expected):
                    np.testing.assert_allclose(vector_grad(v), np.exp(np.asarray(v) - expected),
                                               rtol=1e-12, atol=1e-300)
            u, v = jnp.array($(py(u))), jnp.array($(py(v)))
            np.testing.assert_allclose(jax.jit(lse_elementwise.kernel)(u, v), $(expected.elementwise), rtol=1e-14)
            x, a = jnp.array($(py(x))), jnp.array($(py(a)))
            np.testing.assert_allclose(jax.jit(lse_cells.kernel)(x, a, $(py(s))), $(expected.cells), rtol=1e-14)
            for gradient in jax.grad(lse_cells.kernel, argnums=(0, 1, 2))(x, a, $(py(s))):
                assert np.all(np.isfinite(gradient))
            # Padding of the shorter rows takes no part in each row's reduction.
            prepared = lse_rows.prepare_data(rows=[np.array(r) for r in $(py(py.(rows)))])
            np.testing.assert_allclose(jax.jit(lse_rows.kernel)(**prepared), $(expected.rows), rtol=1e-14)
            """)
            @test success(pipeline(`$python $check`; stdout=stdout, stderr=stderr))
        end
    end
end

@testitem "ReactiveKernels translates Stan's gamma family, lchoose and Bessel I" tags=[:reactivekernels] begin
    # Todo `1q0bqgu` (after snag `jax-source-has-n-a3964196`): the Julia twins
    # of these builtins branch, so `jax_source` refused any helper calling
    # them. Values are the twins', poles and non-finite arguments included,
    # and a domain error of a twin is a runtime check of the module.
    using ReactiveKernels, SpecialFunctions
    @deffun @juliacompat begin
        source_lgamma(x::real)::real = lgamma(x)
        source_digamma(x::real)::real = digamma(x)
        source_trigamma(x::real)::real = trigamma(x)
        source_lchoose(n::real, k::real)::real = lchoose(n, k)
        source_bessel_i(n::int, x::real)::real = modified_bessel_first_kind(n, x)
        source_bessel_i0(x::real)::real = modified_bessel_first_kind(0, x)
        source_bessel_j(x::real)::real = bessel_first_kind(1, x)
        source_bessel_y(x::real)::real = bessel_second_kind(1, x)
        source_bessel_k(x::real)::real = modified_bessel_second_kind(1, x)
    end
    @kernel gamma_lgamma(x::Float64) = begin
        result = source_lgamma(x)
        return result
    end
    @kernel gamma_digamma(x::Float64) = begin
        result = source_digamma(x)
        return result
    end
    @kernel gamma_trigamma(x::Float64) = begin
        result = source_trigamma(x)
        return result
    end
    @kernel gamma_cells(x::Vector{Float64}) = begin
        result = sum(source_lgamma.(x)) + sum(source_digamma.(x)) + sum(source_trigamma.(x))
        return result
    end
    @kernel choose_pair(n::Float64, k::Float64) = begin
        result = source_lchoose(n, k)
        return result
    end
    @kernel choose_cells(n::Vector{Float64}, k::Vector{Float64}) = begin
        result = sum(source_lchoose.(n, k))
        return result
    end
    @kernel bessel_pair(n::Int, x::Float64) = begin
        result = source_bessel_i(n, x)
        return result
    end
    @kernel bessel_zero(x::Float64) = begin
        result = source_bessel_i0(x)
        return result
    end
    sources = Dict(name => string(src) for (name, src) in (
        "gamma_lgamma" => jax_source(gamma_lgamma; have=(:x,), want=:result),
        "gamma_lgamma_host" => jax_source(gamma_lgamma; have=(:x,), want=:result, data=(:x,)),
        "gamma_digamma" => jax_source(gamma_digamma; have=(:x,), want=:result),
        "gamma_trigamma" => jax_source(gamma_trigamma; have=(:x,), want=:result),
        "gamma_cells" => jax_source(gamma_cells; have=(:x,), want=:result),
        "choose_pair" => jax_source(choose_pair; have=(:n, :k), want=:result),
        "choose_pair_host" => jax_source(choose_pair; have=(:n, :k), want=:result, data=(:n, :k)),
        "choose_cells" => jax_source(choose_cells; have=(:n, :k), want=:result),
        "bessel_pair" => jax_source(bessel_pair; have=(:n, :x), want=:result),
        "bessel_zero" => jax_source(bessel_zero; have=(:x,), want=:result)))
    @test occursin("_stanblocks_pole", sources["gamma_digamma"])
    @test occursin("_rk_trigamma", sources["gamma_trigamma"])
    @test occursin("_rk_logbeta", sources["choose_pair"])
    @test occursin("jsp.i0", sources["bessel_zero"])
    # J, Y and K are an upstream boundary: JAX has none, nor does RK.
    @kernel boundary_j(x::Float64) = begin
        result = source_bessel_j(x)
        return result
    end
    @kernel boundary_y(x::Float64) = begin
        result = source_bessel_y(x)
        return result
    end
    @kernel boundary_k(x::Float64) = begin
        result = source_bessel_k(x)
        return result
    end
    for (name, kernel) in (("bessel_first_kind", boundary_j), ("bessel_second_kind", boundary_y),
                           ("modified_bessel_second_kind", boundary_k))
        message = try
            jax_source(kernel; have=(:x,), want=:result)
            ""
        catch err
            err isa ReactiveKernels.JAXEmissionError ? sprint(showerror, err) : rethrow()
        end
        @test occursin("`$name` has no JAX translation", message)
        @test occursin("upstream boundary", message)
    end

    # Expected values are the Julia twins'; `raise` where a twin throws.
    py(x::Real) = isnan(x) ? "nan" : isinf(x) ? (x > 0 ? "inf" : "-inf") : repr(Float64(x))
    py(x::Tuple) = "(" * join(py.(x), ", ") * ")"
    py(x::AbstractVector) = "[" * join(py.(x), ", ") * "]"
    py(::Nothing) = "None"
    twin(f, args...) = try
        f(args...)
    catch err
        err isa Union{DomainError,SpecialFunctions.AmosException} || rethrow()
        nothing
    end
    table(rows) = "[" * join((py(r) for r in rows), ", ") * "]"
    xs = [-Inf, -2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 2.5, 30.0, 1e100, Inf, NaN]
    poles = (-2.0, -1.0, 0.0)
    gamma_rows(f, derivative, xs) = [(x, f(x), x in poles ? 0.0 :
                                      isfinite(x) && isfinite(f(x)) ? derivative(x) : NaN) for x in xs]
    lgamma_rows = [(x, source_lgamma(x), x in poles || !isfinite(x) ? NaN : SpecialFunctions.digamma(x))
                   for x in xs]
    digamma_rows = gamma_rows(source_digamma, SpecialFunctions.trigamma, xs)
    # RK's `trigamma` is 0 at +Inf and 1/x past where `jax.scipy.special`'s
    # polygamma underflows (from RK 0.1.227, RK snag `rk-specialfuncti-72970c82`).
    trigamma_rows = gamma_rows(source_trigamma, x -> SpecialFunctions.polygamma(2, x),
                               vcat(xs, 1e200))
    cells = [0.3, 2.5, 7.0, -1.5]
    cells_value = sum(x -> source_lgamma(x) + source_digamma(x) + source_trigamma(x), cells)
    cells_gradient = [SpecialFunctions.digamma(x) + SpecialFunctions.trigamma(x) +
                      SpecialFunctions.polygamma(2, x) for x in cells]
    choose_args = [(5.0, 2.0), (5.0, 0.0), (5.0, 5.0), (5.0, 6.0), (4.5, 2.25), (20.0, 7.0),
                   (1e6, 3.0), (-1.0, -1.0), (-0.5, 0.3), (3.0, -1.0), (8.0, 9.0), (20.0, 21.0),
                   (20.0, -1.0), (NaN, 1.0), (1.0, NaN), (Inf, 1.0), (5.0, 3.0 + 1e-9),
                   (-2.0, 0.0), (1.0, -2.0), (5.0, 7.0), (30.0, 32.0)]
    # The gradient where it is finite: zero where the twin returns its
    # constant 0, else d/dn and d/dk of log gamma(n + 1) - log gamma(k + 1) -
    # log gamma(n + 1 - k), which every branch computes.
    choose_gradient(n, k) = k == 0 || k == n ? (0.0, 0.0) :
        (SpecialFunctions.digamma(n + 1) - SpecialFunctions.digamma(n + 1 - k),
         SpecialFunctions.digamma(n + 1 - k) - SpecialFunctions.digamma(k + 1))
    choose_rows = [(n, k, twin(source_lchoose, n, k),
                    all(isfinite, (n, k, n + 1 - k, k + 1)) && n > -1 && k > -1 && n + 1 - k > 0 ?
                    choose_gradient(n, k) : nothing) for (n, k) in choose_args]
    for (n, k, value, _) in choose_rows
        @test isequal(value, twin(StanBlocks._julia_lchoose, n, k))
    end
    choose_n, choose_k = [5.0, 4.5, 20.0, 1e6], [2.0, 2.25, 7.0, 3.0]
    bessel_rows = [(n, x, twin(source_bessel_i, n, x),
                    isfinite(x) && x != 800.0 ?
                    (SpecialFunctions.besseli(n - 1, x) + SpecialFunctions.besseli(n + 1, x)) / 2 : nothing)
                   for n in (0, 1, 2) for x in (-Inf, -1.5, 0.0, 0.5, 3.0, 700.0, 800.0, Inf, NaN)]
    @test isnothing(twin(source_bessel_i, 2, Inf)) && isnothing(twin(source_bessel_i, 0, 800.0))
    @test isnan(source_bessel_i(1, -Inf))

    python = get(ENV, "RK_JAX_PYTHON", "")
    if isempty(python)
        @test_skip "Set RK_JAX_PYTHON to check gamma-family, lchoose and Bessel I values and gradients"
    else
        mktempdir() do dir
            for (name, src) in sources
                write(joinpath(dir, name * ".py"), src)
            end
            check = joinpath(dir, "check.py")
            write(check, """
            import jax
            import jax.numpy as jnp
            import numpy as np
            from numpy import inf, nan
            import gamma_lgamma, gamma_lgamma_host, gamma_digamma, gamma_trigamma, gamma_cells
            import choose_pair, choose_pair_host, choose_cells, bessel_pair, bessel_zero

            def close(actual, expected, rtol):
                np.testing.assert_allclose(actual, expected, rtol=rtol, atol=1e-300, equal_nan=True)

            def raises(f, *args):
                try:
                    jax.block_until_ready(f(*args))
                except Exception:
                    return
                raise AssertionError("no error for %r" % (args,))

            for module, rows in ((gamma_lgamma, $(table(lgamma_rows))),
                                 (gamma_digamma, $(table(digamma_rows))),
                                 (gamma_trigamma, $(table(trigamma_rows)))):
                value, gradient = jax.jit(module.kernel), jax.jit(jax.grad(module.kernel))
                for x, expected, derivative in rows:
                    close(value(x), expected, 1e-13)
                    if not np.isnan(derivative):
                        close(gradient(x), derivative, 1e-12)
            for x, expected, _ in $(table(lgamma_rows)):
                close(gamma_lgamma_host.kernel(**gamma_lgamma_host.prepare_data(x=x)), expected, 1e-13)
            cells = jnp.array($(py(cells)))
            close(jax.jit(gamma_cells.kernel)(cells), $(py(cells_value)), 1e-13)
            close(jax.jit(jax.grad(gamma_cells.kernel))(cells), $(py(cells_gradient)), 1e-12)

            pair, pair_grad = jax.jit(choose_pair.kernel), jax.jit(jax.grad(choose_pair.kernel, argnums=(0, 1)))
            for n, k, expected, gradient in $(table(choose_rows)):
                if expected is None:
                    raises(pair, n, k)
                    raises(lambda n, k: choose_pair_host.prepare_data(n=n, k=k), n, k)
                    continue
                close(pair(n, k), expected, 1e-13)
                close(choose_pair_host.kernel(**choose_pair_host.prepare_data(n=n, k=k)), expected, 1e-13)
                if gradient is not None:
                    close(pair_grad(n, k), gradient, 1e-9)
            n, k = jnp.array($(py(choose_n))), jnp.array($(py(choose_k)))
            close(jax.jit(choose_cells.kernel)(n, k), $(py(sum(source_lchoose.(choose_n, choose_k)))), 1e-13)
            raises(jax.jit(choose_cells.kernel), jnp.array([5.0, 5.0]), jnp.array([2.0, 7.0]))

            pair, pair_grad = jax.jit(bessel_pair.kernel), jax.jit(jax.grad(bessel_pair.kernel, argnums=1))
            zero, zero_grad = jax.jit(bessel_zero.kernel), jax.jit(jax.grad(bessel_zero.kernel))
            for n, x, expected, gradient in $(table(bessel_rows)):
                runs = [(pair, pair_grad, (n, x))] + ([(zero, zero_grad, (x,))] if n == 0 else [])
                for value, derivative, args in runs:
                    if expected is None:
                        raises(value, *args)
                        continue
                    close(value(*args), expected, 1e-12)
                    if gradient is not None:
                        close(derivative(*args), gradient, 1e-10)
            """)
            @test success(pipeline(`$python $check`; stdout=stdout, stderr=stderr))
        end
    end
end
