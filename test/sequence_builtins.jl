"""
Stan's `head(v, n)`, `tail(v, n)` and `reverse(v)` take a vector, a row vector or
an array of any element type. None was registered: a `@deffun` body computing
the isometric sum-to-zero map with them failed with `Could not find tail in model,
builtin, … or Main!` (snag `stan-vector-buil-5792b9e5`), and `reverse` reached
Base's function, which has no Stan type. A downstream module spelled them as
`segment(append_row(c, 0.0), 2, n - 1)` and suffix sums
`w - cumulative_sum(w) + sum(w)`. Each now types as its argument with the leading
size replaced: `n` for `head`/`tail`, unchanged for `reverse`.
"""
@testitem "slic: Stan's head, tail and reverse trace over vectors, row vectors and arrays" tags=[:slic, :regression, :shapes, :stanc] setup=[StanBlocksImports, StanBlocksTestSetup] begin
    using .StanBlocksTestSetup: stan_block, stanc_compiles, transpiles

    # The reported shape: the isometric sum-to-zero map in Stan-only `@deffun`s.
    @deffun begin
        iso_sum_to_zero(c::vector[n])::vector[n] = begin
            j = linspaced_vector(n - 1, 1, n - 1)
            w = tail(c, n - 1) ./ sqrt(j .* (j + 1))
            append_row(reverse(cumulative_sum(reverse(w))), 0.0) - append_row(0.0, j .* w)
        end
        iso_partial_sums(z::vector[n])::vector[n - 1] = head(cumulative_sum(z), n - 1)
    end
    reported = @slic (; n=4, y=[0.1, -0.2, 0.3]) begin
        c::vector[n] ~ normal(0.0, 1.0)
        y ~ normal(iso_partial_sums(iso_sum_to_zero(c)), 1.0)
    end
    code = stan_code(reported)
    @test occursin("tail(c, (n - 1))", code)
    @test occursin("reverse(cumulative_sum(reverse(w)))", code)
    @test occursin("head(cumulative_sum(z), (n - 1))", code)
    @test stanc_compiles(reported)

    # Every overload, as a transformed parameter. `int` data stays data.
    data = (; m=2, n=3, k=2, y=0.3, ks=[3, 1, 2])
    model(rhs, use=:(sum(out))) = StanBlocks.stan.SlicModel(quote
        v::vector[n] ~ normal(0.0, 1.0)
        rv::row_vector[n] ~ normal(0.0, 1.0)
        a::real[n] ~ normal(0.0, 1.0)
        s::simplex[n] ~ dirichlet(rep_vector(1.0, n))
        V::vector[m, n] ~ multi_normal(rep_vector(0.0, n), identity_matrix(n))
        out = $rhs
        y ~ normal($use + sum(v) + sum(rv) + sum(a) + sum(s) + sum(V[1]), 1.0)
    end, data, @__MODULE__)
    cases = [
        :(head(v, k)) => ("transformed parameters", "vector[k] out = head(v, k);"),
        :(tail(v, 2)) => ("transformed parameters", "vector[2] out = tail(v, 2);"),
        :(head(v, 0)) => ("transformed parameters", "vector[0] out = head(v, 0);"),
        :(reverse(v)) => ("transformed parameters", "vector[n] out = reverse(v);"),
        :(head(rv, k)) => ("transformed parameters", "row_vector[k] out = head(rv, k);"),
        :(reverse(rv)) => ("transformed parameters", "row_vector[n] out = reverse(rv);"),
        :(tail(a, k)) => ("transformed parameters", "array[k] real out = tail(a, k);"),
        :(reverse(a)) => ("transformed parameters", "array[n] real out = reverse(a);"),
        # A slice of a simplex is a plain vector.
        :(tail(s, k)) => ("transformed parameters", "vector[k] out = tail(s, k);"),
        :(reverse(s)) => ("transformed parameters", "vector[n] out = reverse(s);"),
        :(tail(ks, k)) => ("transformed data", "array[k] int out = tail(ks, k);"),
        :(reverse(ks)) => ("transformed data", "array[ks_n] int out = reverse(ks);"),
    ]
    for (rhs, (block, decl)) in cases
        m = model(rhs)
        @test occursin(decl, stan_block(stan_code(m), block))
        @test stanc_compiles(m)
    end
    # An array keeps its element type, here a vector (Stan has no `sum` of an
    # `array[] vector`, so the use reads one element).
    for (rhs, decl) in (:(head(V, 1)) => "array[1] vector[n] out = head(V, 1);",
                        :(reverse(V)) => "array[m] vector[n] out = reverse(V);")
        m = model(rhs, :(sum(out[1])))
        @test occursin(decl, stan_block(stan_code(m), "transformed parameters"))
        @test stanc_compiles(m)
    end

    # A spliced `Base.reverse` value emits what the name does.
    @test stan_code(model(Expr(:call, Base.reverse, :v))) == stan_code(model(:(reverse(v))))

    # refused: Stan has no `head`/`tail`/`reverse` of a matrix or a scalar, so
    # the call stays untyped and its use stops at trace time.
    for rhs in (:(head(V[1] * rv, 1)), :(reverse(y)))
        @test !transpiles(model(rhs); re=false)
    end
end

"""
The `@juliacompat` Julia methods of `head`, `tail` and `reverse` follow Stan:
`head(v, 0)` is empty, a negative or too large `n` is out of range, and a row
vector held as an adjoint stays one.
"""
@testitem "slic: @juliacompat head, tail and reverse follow Stan" tags=[:slic] setup=[StanBlocksImports] begin
    shims = Module(:JuliaCompatSequenceShims)
    Core.eval(shims, :(using StanBlocks))
    Core.eval(shims, quote
        @deffun @juliacompat begin
            shim_head(v::vector[n], k::int)::vector[k] = head(v, k)
            shim_tail(v::vector[n], k::int)::vector[k] = tail(v, k)
            shim_reverse(v::vector[n])::vector[n] = reverse(v)
            shim_reverse_rv(v::vector[n])::row_vector[n] = reverse(v')
            # The reported map (`j = 1:n-1` passed in: `linspaced_vector` has
            # no Julia method).
            iso_sum_to_zero(c::vector[n], j::vector[m])::vector[n] = begin
                w = tail(c, n - 1) ./ sqrt(j .* (j + 1))
                append_row(reverse(cumulative_sum(reverse(w))), 0.0) - append_row(0.0, j .* w)
            end
        end
    end)
    call(f, args...) = Base.invokelatest(getproperty(shims, f), args...)

    v = [1.0, 2.0, 3.0, 4.0]
    @test call(:shim_head, v, 2) == [1.0, 2.0]
    @test call(:shim_tail, v, 3) == [2.0, 3.0, 4.0]
    @test call(:shim_head, v, 0) == Float64[]
    @test call(:shim_tail, v, 4) == v
    @test call(:shim_reverse, v) == [4.0, 3.0, 2.0, 1.0]
    @test call(:shim_reverse_rv, v) == [4.0 3.0 2.0 1.0]
    for k in (-1, 5), f in (:shim_head, :shim_tail)
        @test_throws BoundsError call(f, v, k)
    end
    jcall(name, args...) = StanBlocks.jcall(Val(name), shims, args...)
    @test jcall(:head, v', 2) == [1.0 2.0]
    @test jcall(:tail, transpose(v), 1) == transpose([4.0])
    @test jcall(:reverse, [3, 1, 2]) == [2, 1, 3]
    @test jcall(:head, [[1.0, 2.0], [3.0]], 1) == [[1.0, 2.0]]

    # Stan Math's `sum_to_zero_constrain` loop, of the last `n - 1` elements.
    function stan_sum_to_zero(y)
        N = length(y)
        z = zeros(N + 1)
        sum_w = 0.0
        for i in N:-1:1
            w = y[i] / sqrt(i * (i + 1))
            sum_w += w
            z[i] += sum_w
            z[i + 1] -= w * i
        end
        z
    end
    c = [0.3, -1.2, 0.8, 2.1, -0.4]
    z = call(:iso_sum_to_zero, c, [1.0, 2.0, 3.0, 4.0])
    @test z ≈ stan_sum_to_zero(c[2:end])
    @test abs(sum(z)) < 1e-12
end
