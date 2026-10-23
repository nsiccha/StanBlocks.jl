"""
Verify `slic: descriptor outputs report their evaluated constraint values`.

Snag read-the-numeric-9a6adfef (from a downstream regression package):
`ModelOutput.constraints` reports each bound as spelled, an internal expression
such as `fmax(0.0, (2 * floor_))`, and nothing public evaluated it. A consumer
reproducing Stan's constraining transform had to re-implement Stan's expression
semantics. `constraint_values` now carries each bound evaluated against the
current data, by the scope that resolves `dims`: data arithmetic, transformed-data
values, data vectors, Stan's integer division, literal per-coordinate vectors
with unbounded (`negative_infinity()`) coordinates, and the per-coordinate arrays
the compiler fills in transformed-data loops. A bound that reads a parameter has
no data value and reports `nothing`; a data-only bound computed by a Stan-only
function fails closed, as a size does.
"""
@testitem "slic: descriptor outputs report their evaluated constraint values" tags=[:slic, :plate, :scan, :ragged, :descriptor, :regression] setup=[StanBlocksImports, StanBlocksTestSetup] begin
    bound_values(m, name) = only(o.constraint_values for o in stan_descriptor(m).outputs if o.name === name)
    # Compiler-owned carrier names end in a trace counter; drop it.
    bounds(m) = Dict(
        Symbol(replace(string(o.name), r"_\d+$" => "")) => o.constraint_values
        for o in stan_descriptor(m).outputs if !isempty(o.constraints)
    )

    # The reported shape: arithmetic over data, `fmax`, two-sided, integer
    # arithmetic, a data vector, a transformed-data value, an affine pair.
    reported = @slic (; floor_ = 0.3, k = 7, lo = [0.1, 0.2, 0.0], xs = [0.1, 0.4], s = 2.0, mu0 = 1.0, y = [1.0, 2.0]) begin
        rho ~ lognormal(0., 4.; lower = fmax(0.0, 2 * floor_))
        b ~ normal(0., 1.; lower = 2 * floor_, upper = 5.0)
        i ~ normal(0., 1.; lower = k ÷ 2, upper = k * log(k))
        j ~ normal(0., 1.; lower = k / 2)
        c::vector[3] ~ normal(0., 1.; lower = lo)
        m0 = maximum(xs)
        a ~ normal(0., 1.; lower = m0)
        w ~ normal(mu0, s; offset = mu0, multiplier = s)
        sigma ~ exponential(1.)
        y ~ normal(rho + b + i + j + sum(c) + a + w, sigma)
    end
    d = stan_descriptor(reported)
    rho = only(o for o in d.outputs if o.name === :rho)
    @test rho.constraints.lower isa StanBlocks.StanExpr
    @test rho.constraint_values === (lower = 0.6,)
    @test bound_values(reported, :b) === (lower = 0.6, upper = 5.0)
    @test bound_values(reported, :i) == (lower = 3, upper = 7 * log(7))
    # Stan divides two ints as ints: `lower=(k / 2)` is 3, not 3.5.
    @test bound_values(reported, :j) === (lower = 3,)
    @test bound_values(reported, :c) == (lower = [0.1, 0.2, 0.0],)
    @test bound_values(reported, :a) === (lower = 0.4,)
    @test bound_values(reported, :w) === (offset = 1.0, multiplier = 2.0)
    @test bound_values(reported, :sigma) === (lower = 0.0,)
    @test all(keys(o.constraint_values) == keys(o.constraints) for o in d.outputs)
    @test all(isempty(o.constraint_values) for o in d.outputs if isempty(o.constraints))
    @test all(keys(i.constraint_values) == keys(i.constraints) for i in d.inputs)
    # Rebound data gives rebound values.
    rebound = StanBlocks.stan_model(reported)(; floor_ = 1.0, k = 9, lo = [0.5, 0.6, 0.7], xs = [2.0])
    @test bound_values(rebound, :rho) === (lower = 2.0,)
    @test bound_values(rebound, :j) === (lower = 4,)
    @test bound_values(rebound, :c) == (lower = [0.5, 0.6, 0.7],)
    @test bound_values(rebound, :a) === (lower = 2.0,)

    # Per-coordinate bounds: a literal vector with unbounded coordinates, and
    # the arrays the compiler fills in transformed data, which hold the bound
    # each coordinate gets (`-Inf` where a coordinate is unbounded).
    @test bound_values(@slic((; y = 0.3), begin
        tau ~ vcat(normal(0., 1.), exponential(1.), normal(0., 1.))
        y ~ normal(sum(tau), 1.)
    end), :tau) == (lower = [-Inf, 0.0, -Inf],)
    indexed = @slic (; y = 0.3, pos = [1, 3], free = [2]) begin
        tau::vector[3]
        tau[pos] ~ exponential(1.)
        tau[free] ~ normal(0., 1.)
        y ~ normal(sum(tau), 1.)
    end
    @test bound_values(indexed, :tau) == (lower = [0.0, -Inf, 0.0],)
    @test bound_values(StanBlocks.stan_model(indexed)(; pos = [2], free = [1, 3]), :tau) ==
        (lower = [-Inf, 0.0, -Inf],)
    @test bound_values(@slic((; y = 0.3, rate = [1.5, 3.0]), begin
        tau ~ vcat(exponential(rate), normal(0., 1.))
        y ~ normal(sum(tau), 1.)
    end), :tau) == (lower = [0.0, 0.0, -Inf],)
    @test bound_values(@slic((; y = randn(3), L = [0., 1., 2.]), begin
        @scan begin
            for t in 1:3
                x[t] ~ normal(0., 1.; lower = L[t])
            end
        end
        y ~ normal(x, 1.)
    end), :x) == (lower = [0.0, 1.0, 2.0],)
    # A plate cell bound reading the cell's data is hoisted the same way, as is
    # a `@scan` nested in a `@plate for`, over every enclosing axis.
    @test bounds(@slic((; ys = [0.1, 0.2, 0.3, 0.4], L = [0.1, 0.2, 0.3, 0.4]), begin
        th ~ plate(ys, L; outer = 4) do yi, li
            t ~ normal(0., 1.; lower = li)
            yi ~ normal(t, 1.0)
            t
        end
    end)) == Dict(:th_t => (lower = [0.1, 0.2, 0.3, 0.4],), :th => (lower = [0.1, 0.2, 0.3, 0.4],))
    @test bound_values(@slic((; y = randn(6), S = 2), begin
        @plate for j in 1:S
            @scan begin
                h[1] ~ exponential(1.)
                for t in 2:3
                    h[t] ~ normal(h[t-1], 1.)
                end
            end
        end
        y ~ normal(to_vector(h), 1.)
    end), :h) == (lower = [0.0 0.0; -Inf -Inf; -Inf -Inf],)
    axes = bound_values(@slic((; y = randn(3), S = 2, R = 2, L = [0., 1.], U = [2., 3., 4.]), begin
        @plate for i in 1:R, j in 1:S
            @scan begin
                h[1] ~ normal(0., 1.; lower = L[j])
                for t in 2:3
                    h[t] ~ normal(h[t-1], 1.; upper = U[t])
                end
            end
        end
        y ~ normal(h[2][:, 1], 1.)
    end), :h)
    @test size(axes.lower) == size(axes.upper) == (2, 3, 2)
    @test axes.lower[:, 1, :] == [0. 0.; 1. 1.] && all(==(-Inf), axes.lower[:, 2:3, :])
    @test all(==(Inf), axes.upper[:, 1, :]) && axes.upper[:, 2:3, 1] == [3. 4.; 3. 4.]
    # A varying-length cell's flat memory, laid out by the cell lengths.
    ragged = StanBlocks.SlicModel(quote
        phi ~ uniform(-1, 1); s ~ exponential(1)
        @plate for j in 1:J
            @scan begin
                h[1] ~ exponential(1)
                for t in 2:T[j]
                    h[t] ~ normal(phi * h[t-1], s)
                end
            end
            obs[j] ~ normal(h, 1.0)
        end
    end, Dict{Symbol,Any}(:T => [2, 3], :J => 2, :obs => [[0.1, 0.2], [0.3, 0.4, 0.5]]), @__MODULE__)
    @test bounds(ragged)[:h__pl_mem] == (lower = [0.0, -Inf, 0.0, -Inf, -Inf],)

    # A bound that reads a parameter has no data value.
    param = @slic (; y = [1.0, 2.0]) begin
        a ~ normal(0., 1.)
        b ~ normal(0., 1.; lower = a)
        y ~ normal(a + b, 1.0)
    end
    @test bound_values(param, :b) === (lower = nothing,)
    @test only(o.constraints for o in stan_descriptor(param).outputs if o.name === :b).lower isa StanBlocks.StanExpr

    # A data-only bound computed by a Stan-only function fails closed, naming
    # the output, the key and the fix; `@juliacompat` gives it a value.
    @deffun bounds_stanonly_floor(x::real)::real = 2.0 * x
    @deffun @juliacompat bounds_jc_floor(x::real)::real = 2.0 * x
    stan_only = @slic (; f = 0.2, y = [1.0, 2.0]) begin
        a ~ normal(0., 1.; lower = bounds_stanonly_floor(f))
        y ~ normal(a, 1.0)
    end
    msg = try
        stan_descriptor(stan_only)
        nothing
    catch err
        sprint(showerror, err)
    end
    @test !isnothing(msg) && occursin("`lower` constraint of output `a`", msg) && occursin("@juliacompat", msg)
    # Printing the model never evaluates a bound, so it still works.
    @test occursin("outputs", sprint(show, MIME"text/plain"(), StanBlocks.stan_model(stan_only)))
    @test bound_values(@slic((; f = 0.2, y = [1.0, 2.0]), begin
        a ~ normal(0., 1.; lower = bounds_jc_floor(f))
        y ~ normal(a, 1.0)
    end), :a) === (lower = 0.4,)
end

"""
Verify `slic: the Julia layer divides two ints as Stan does`.

Snag read-the-numeric-9a6adfef: Stan divides two integers as integers,
truncating toward zero, but `/` in StanBlocks' Julia layer fell through to
Julia's float division. A `@juliacompat` helper returned `3.5` where the Stan
function returns `3`, and `stan_descriptor` threw for the valid size
`vector[n / 2]`. Stan's non-finite constants had no Julia value either.
"""
@testitem "slic: the Julia layer divides two ints as Stan does" tags=[:slic, :descriptor, :regression] setup=[StanBlocksImports, StanBlocksTestSetup] begin
    @deffun @juliacompat divint_half(n::int)::int = n / 2
    @test Base.invokelatest(divint_half, 7) === 3
    # Stan, like C++, truncates toward zero.
    @test Base.invokelatest(divint_half, -7) === -3
    @test StanBlocks.jcall(Val(:/), StanBlocks, 7, 2) === 3
    @test StanBlocks.jcall(Val(:/), StanBlocks, 7.0, 2) === 3.5
    @test StanBlocks.jcall(Val(:/), StanBlocks, 7, 2.0) === 3.5
    @test occursin("(n / 2)", stan_code(@slic((; n = 7, y = [1.0, 2.0]), begin
        z ~ std_normal(; n = n / 2)
        y ~ normal(sum(z), 1.0)
    end)))
    @test only(o.dims for o in stan_descriptor(@slic((; n = 7, y = [1.0, 2.0]), begin
        z ~ std_normal(; n = n / 2)
        y ~ normal(sum(z), 1.0)
    end)).outputs if o.name === :z) == (3,)
    # Stan's non-finite constants have Julia values too.
    @test StanBlocks.jcall(Val(:negative_infinity), StanBlocks) === -Inf
    @test StanBlocks.jcall(Val(:positive_infinity), StanBlocks) === Inf
    @test isnan(StanBlocks.jcall(Val(:not_a_number), StanBlocks))
    @deffun @juliacompat divint_floor_or_unbounded(x::real)::real = x > 0 ? x : negative_infinity()
    @test Base.invokelatest(divint_floor_or_unbounded, 0.5) === 0.5
    @test Base.invokelatest(divint_floor_or_unbounded, -1.0) === -Inf
end

"""
Verify `slic: Stan's vectorised builtins evaluate elementwise in the Julia layer`.

Snag stanblocks-stan-458f365d (from a downstream regression package): a vector
parameter bounded by `fmax(0.0, lo)` over a data vector, which Stan evaluates
elementwise, made `stan_descriptor` throw, because the Julia twin of `fmax` was
`max(x, y)`. `fmax` of two vectors did not throw but returned the
lexicographically larger vector, and `inv` of a matrix returned the matrix
inverse. Every math builtin Stan vectorises now applies its scalar twin to each
element, in a descriptor bound and a `@juliacompat` body alike.
"""
@testitem "slic: Stan's vectorised builtins evaluate elementwise in the Julia layer" tags=[:slic, :descriptor, :regression] setup=[StanBlocksImports, StanBlocksTestSetup] begin
    bound_values(m, name) = only(o.constraint_values for o in stan_descriptor(m).outputs if o.name === name)
    data = (; lo = [0.1, -0.2], hi = [3.0, 4.0], y = [1.0, 2.0])
    # The reported shape: a data vector floored by a scalar, one bound per coordinate.
    floored = @slic data begin
        rho::vector[2] ~ lognormal(0., 4.; lower = fmax(0.0, lo))
        y ~ normal(sum(rho), 1.)
    end
    @test bound_values(floored, :rho) == (lower = [0.1, 0.0],)
    @test bound_values(StanBlocks.stan_model(floored)(; lo = [-1.0, 2.0]), :rho) == (lower = [0.0, 2.0],)
    @test bound_values(@slic(data, begin
        rho::vector[2] ~ lognormal(0., 4.; lower = fmax(lo, 0.0))
        y ~ normal(sum(rho), 1.)
    end), :rho) == (lower = [0.1, 0.0],)
    @test bound_values(@slic(data, begin
        rho::vector[2] ~ normal(0., 4.; upper = fmin(3.5, hi))
        y ~ normal(sum(rho), 1.)
    end), :rho) == (upper = [3.0, 3.5],)
    # Two vectors pair up elementwise; Julia's `max` returns all of `lo`.
    @test bound_values(@slic(data, begin
        rho::vector[2] ~ lognormal(0., 4.; lower = fmax(lo, hi .- 3.5))
        y ~ normal(sum(rho), 1.)
    end), :rho) == (lower = [0.1, 0.5],)
    @test bound_values(@slic(data, begin
        rho::vector[2] ~ normal(0., 4.; lower = log1p(hi))
        y ~ normal(sum(rho), 1.)
    end), :rho) == (lower = log1p.([3.0, 4.0]),)

    jc(name, args...) = StanBlocks.jcall(Val(name), StanBlocks, args...)
    # Stan's `inv` is elementwise; Julia's `inv` of a matrix inverts it.
    @test jc(:inv, [1.0 2.0; 4.0 8.0]) == [1.0 0.5; 0.25 0.125]
    @test jc(:fmin, [1.0 5.0; -1.0 3.0], 2.0) == [1.0 2.0; -1.0 2.0]
    # A nested array recurses into its elements.
    @test jc(:fmax, 0.0, [[-1.0, 1.0], [2.0, -2.0, 3.0]]) == [[0.0, 1.0], [2.0, 0.0, 3.0]]
    # Stan's `fmax`/`fmin` are C's: a NaN argument yields the other one.
    @test jc(:fmax, NaN, 1.0) === 1.0 && jc(:fmin, 2.0, NaN) === 2.0
    @test jc(:fmax, [NaN, 1.0], 0.5) == [0.5, 1.0]
    # refused: Stan Math's `apply_scalar_binary` requires containers of one size,
    # where Julia's broadcast would stretch the one-element vector.
    @test_throws DimensionMismatch jc(:fmax, [1.0], [1.0, 2.0])
    # This package binds `log2` to its own builtin; the twin is still Julia's.
    @test jc(:log2, 8.0) === 3.0 && jc(:log2, [2.0, 4.0]) == [1.0, 2.0]

    @deffun @juliacompat elementwise_floor(x::vector[n])::vector[n] = fmax(0.0, x)
    @deffun @juliacompat elementwise_inv(x::matrix[m, n])::matrix[m, n] = inv(x)
    @test Base.invokelatest(elementwise_floor, [0.1, -0.2]) == [0.1, 0.0]
    @test Base.invokelatest(elementwise_inv, [1.0 2.0; 4.0 8.0]) == [1.0 0.5; 0.25 0.125]

    # Every vectorised builtin's container call agrees with its scalar twin.
    # Those still without a scalar twin are tracked by todo
    # `StanBlocks/todos/2026-10-23T09-47-31-458-1j6rdy6`.
    no_twin = (:inv_sqrt, :Phi, :cbrt, :erf, :erfc, :Phi_approx, :inv_Phi, :inv_cloglog,
        :lambert_w0, :fdim, :fmod, :atan2, :lbeta, :log_diff_exp, :owens_t, :gamma_p, :gamma_q,
        :binary_log_loss, :bessel_first_kind, :bessel_second_kind, :modified_bessel_first_kind,
        :modified_bessel_second_kind, :log_modified_bessel_first_kind, :fma)
    base = [0.1 0.2; 0.3 0.45]
    for f in StanBlocks._ELEMENTWISE_UNARY
        name, owner = nameof(f), parentmodule(f)
        twin(x) = StanBlocks.jcall(Val(name), owner, x)
        # Inside each builtin's domain.
        x = name === :acosh ? 1 .+ base : name === :log1m_exp ? -base : base
        if name in no_twin
            @test_broken twin(x[1]) isa Real
        else
            @test isequal(twin(x), map(twin, x))
            @test isequal(twin(vec(x)), map(twin, vec(x)))
        end
    end
    for (f, n) in StanBlocks._ELEMENTWISE_NARY
        name, owner = nameof(f), parentmodule(f)
        twin(args...) = StanBlocks.jcall(Val(name), owner, args...)
        if name in no_twin
            @test_broken twin(ntuple(_ -> 1, n)...) isa Real
        else
            @test isequal(twin(base, 0.25), map(x -> twin(x, 0.25), base))
            @test isequal(twin(0.25, base), map(x -> twin(0.25, x), base))
            @test isequal(twin(base, base'), map(twin, base, base'))
        end
    end
end

"""
Verify `slic: descriptor constraint values are the bounds Stan applies`.

Snag read-the-numeric-9a6adfef: each parameter's `constraint_values`, put
through Stan's constraining transform, reproduces BridgeStan's
`param_constrain`. This covers integer division (`lower=(k / 2)` is 3), literal
and compiler-filled per-coordinate bounds, a hoisted plate cell bound, a `@scan`
nested in a `@plate for`, and a varying-length cell's flat memory.
"""
@testitem "slic: descriptor constraint values are the bounds Stan applies" tags=[:slic, :plate, :scan, :ragged, :descriptor, :bridgestan] setup=[StanBlocksImports, StanBlocksTestSetup] begin
    # Stan's constraining transform of one coordinate, given its bound values.
    transform(c, x) = haskey(c, :offset) || haskey(c, :multiplier) ?
        get(c, :offset, 0.0) + get(c, :multiplier, 1.0) * x :
        transform(get(c, :lower, -Inf), get(c, :upper, Inf), x)
    transform(lo, hi, x) =
        isfinite(lo) && isfinite(hi) ? lo + (hi - lo) / (1 + exp(-x)) :
        isfinite(lo) ? lo + exp(x) : isfinite(hi) ? hi - exp(x) : x
    coordinate(v, k) = v isa AbstractArray ? vec(v)[k] : v
    models = (
        @slic((; floor_ = 0.3, k = 7, lo = [0.1, 0.2, 0.0], xs = [0.1, 0.4], s = 2.0, mu0 = 1.0, y = [1.0, 2.0]), begin
            rho ~ lognormal(0., 4.; lower = fmax(0.0, 2 * floor_))
            b ~ normal(0., 1.; lower = 2 * floor_, upper = 5.0)
            i ~ normal(0., 1.; lower = k ÷ 2, upper = k * log(k))
            j ~ normal(0., 1.; lower = k / 2)
            c::vector[3] ~ normal(0., 1.; lower = lo)
            m0 = maximum(xs)
            a ~ normal(0., 1.; lower = m0)
            w ~ normal(mu0, s; offset = mu0, multiplier = s)
            sigma ~ exponential(1.)
            y ~ normal(rho + b + i + j + sum(c) + a + w, sigma)
        end),
        @slic((; y = 0.3, pos = [1, 3], free = [2], rate = [1.5, 3.0]), begin
            t1 ~ vcat(normal(0., 1.), exponential(1.), normal(0., 1.))
            t2::vector[3]
            t2[pos] ~ exponential(1.)
            t2[free] ~ normal(0., 1.)
            t3 ~ vcat(exponential(rate), normal(0., 1.))
            y ~ normal(sum(t1) + sum(t2) + sum(t3), 1.)
        end),
        @slic((; ys = [0.1, 0.2, 0.3, 0.4], L = [0.1, 0.2, 0.3, 0.4]), begin
            th ~ plate(ys, L; outer = 4) do yi, li
                t ~ normal(0., 1.; lower = li)
                yi ~ normal(t, 1.0)
                t
            end
        end),
        @slic((; y = randn(6), S = 2), begin
            @plate for j in 1:S
                @scan begin
                    h[1] ~ exponential(1.)
                    for t in 2:3
                        h[t] ~ normal(h[t-1], 1.)
                    end
                end
            end
            y ~ normal(to_vector(h), 1.)
        end),
        StanBlocks.SlicModel(quote
            phi ~ uniform(-1, 1); s ~ exponential(1)
            @plate for j in 1:J
                @scan begin
                    h[1] ~ exponential(1)
                    for t in 2:T[j]
                        h[t] ~ normal(phi * h[t-1], s)
                    end
                end
                obs[j] ~ normal(h, 1.0)
            end
        end, Dict{Symbol,Any}(:T => [2, 3], :J => 2, :obs => [[0.1, 0.2], [0.3, 0.4, 0.5]]), @__MODULE__),
        # Stan's vectorised `fmax`/`fmin` bound each coordinate by its own entry
        # (snag stanblocks-stan-458f365d).
        @slic((; lo = [0.1, -0.2], hi = [3.0, 4.0], y = [1.0, 2.0]), begin
            r1::vector[2] ~ lognormal(0., 4.; lower = fmax(0.0, lo))
            r2::vector[2] ~ lognormal(0., 4.; lower = fmax(lo, hi .- 3.5))
            r3::vector[2] ~ normal(0., 4.; upper = fmin(3.5, hi))
            y ~ normal(sum(r1) + sum(r2) + sum(r3), 1.)
        end),
    )
    for m in models
        d = stan_descriptor(m)
        model = stan_execute(d, :instantiate).model
        names = BridgeStan.param_names(model)
        for x in (-1.3, 0.4, 2.1)
            theta = BridgeStan.param_constrain(model, fill(x, BridgeStan.param_unc_num(model)))
            for o in d.outputs
                o.kind === :parameter || continue
                got = [theta[i] for (i, s) in enumerate(names) if s == string(o.name) || startswith(s, string(o.name, "."))]
                @test length(got) == prod(o.dims)
                @test got ≈ [transform(map(v -> coordinate(v, k), o.constraint_values), x) for k in eachindex(got)]
            end
        end
    end
end
