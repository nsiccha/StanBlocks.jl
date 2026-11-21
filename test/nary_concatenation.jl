using TestItemRunner

# Regression for snag `generate-stan-wi-ade7712e`: `hcat`, `append_col` and
# `append_row` of three or more operands lower to a balanced tree of Stan's
# binary `append_col` / `append_row`. Their vararg definitions peel one operand
# per call, so each arity used to be its own generated overload: a 48-column
# `hcat` emitted 48 `hcat` overloads (55 KB) and took about a minute.
@testsnippet NaryConcatModels begin
    columns(K; n = 5) = Dict{Symbol,Any}(Symbol("c", j) => collect(1.0:n) .+ 10j for j in 1:K)
    hcat_of(K) = Expr(:call, :hcat, (Symbol("c", j) for j in 1:K)...)
    concat_model(rhs, data) = StanBlocks.SlicModel(quote
        X = $rhs
        s ~ std_normal()
    end, data, @__MODULE__, ())
    concat_code(rhs, data) = stan_code(concat_model(rhs, data))
    concat_overloads(code) =
        count(r"^(matrix|vector|row_vector) (hcat|append_col|append_row)\("m, code)
    # The right-hand side that defines `X`, whitespace removed.
    x_rhs(code) = (m = match(r"\] X = (.*?);"s, code); m === nothing ? "" : replace(m.captures[1], r"\s+" => ""))
end

@testitem "slic: three or more operands concatenate through a balanced tree of native calls" tags=[:slic, :regression, :shapes, :stanc] setup=[StanBlocksImports, NaryConcatModels] begin
    for K in (3, 7, 48)
        code = concat_code(hcat_of(K), columns(K))
        @test concat_overloads(code) == 0
        @test count("append_col(", code) == K - 1
        # The operands keep their order (a declared size like `c48_n` is not one).
        @test [m.match for m in eachmatch(r"\bc\d+\b", x_rhs(code))] == ["c$j" for j in 1:K]
        @test stanc_check(code; warn_pedantic = false).ok
    end
    # Halves of ⌈K/2⌉ and ⌊K/2⌋ operands: ⌈log₂ K⌉ calls deep.
    @test x_rhs(concat_code(hcat_of(7), columns(7))) ==
          "append_col(append_col(append_col(c1,c2),append_col(c3,c4)),append_col(append_col(c5,c6),c7))"
    # Julia's bracket spelling is the same call.
    @test concat_code(:([c1 c2 c3]), columns(3)) == concat_code(hcat_of(3), columns(3))
    # One and two columns keep their definitions.
    @test x_rhs(concat_code(:(hcat(c1, c2)), columns(2))) == "hcat(c1,c2)"
    @test x_rhs(concat_code(:(hcat(c1)), columns(1))) == "hcat(c1)"

    data = merge(columns(3), Dict{Symbol,Any}(:M => reshape(collect(1.0:10.0), 5, 2), :r => [1.0, 2.0]))
    for (rhs, tree) in (
        :(hcat(M, c1, c2)) => "append_col(append_col(M,c1),c2)",
        :(append_col(c1, M, c2, M)) => "append_col(append_col(c1,M),append_col(c2,M))",
        :(append_row(c1, c2, c3)) => "append_row(append_row(c1,c2),c3)",
        :(append_row(r', r', M)) => "append_row(append_row((r'),(r')),M)",
        :(append_row(c1, 1.0, c2, 2.0)) => "append_row(append_row(c1,1.0),append_row(c2,2.0))",
        # A run of scalars becomes one literal, so no two scalars meet.
        :(append_row(c1, 1.0, 2.0, 3.0)) => "append_row(c1,[1.0,2.0,3.0]')",
        :(append_col(1.0, r', 2.0, 3.0)) => "append_col(append_col(1.0,(r')),([2.0,3.0]''))",
    )
        code = concat_code(rhs, data)
        @test x_rhs(code) == tree
        @test concat_overloads(code) == 0
        @test stanc_check(code; warn_pedantic = false).ok
    end

    # A shape the vararg definitions refuse keeps their diagnostic.
    @test_throws "`tracetype` not defined for hcat(::vector, ::matrix)" concat_code(:(hcat(c1, M, c2)), data)
    @test_throws "tracetype not defined for X = hcat(::vector, ::vector, ::matrix)" concat_code(:(hcat(c1, c2, M)), data)
    @test_throws "`tracetype` not defined for append_row(::real, ::real)" concat_code(:(append_row(1.0, 2.0, c1)), data)
end

@testitem "slic: three or more operands concatenate in argument order natively" tags=[:slic, :regression, :bridgestan] setup=[StanBlocksImports, NaryConcatModels] begin
    # A design matrix of data columns, the shape a regression formula emits.
    n, K = 4, 6
    data = columns(K; n)
    data[:y] = [0.3, -0.2, 0.5, 1.1]
    body = quote
        X = $(hcat_of(K))
        b ~ std_normal(; n = $K)
        y ~ normal(X * b, 1.0)
    end
    problem = instantiate(stan_model(StanBlocks.SlicModel(body, data, @__MODULE__, (:y,))))
    X = reduce(Base.hcat, [data[Symbol("c", j)] for j in 1:K])
    kernel(b) = -0.5 * sum(abs2, data[:y] - X * b) - 0.5 * sum(abs2, b)
    b1, b2 = collect(range(-0.4, 0.6; length = K)), collect(range(0.5, -0.3; length = K))
    @test LogDensityProblems.logdensity(problem, b1) - LogDensityProblems.logdensity(problem, b2) ≈
          kernel(b1) - kernel(b2)

    # Mixed operands, read back from the generated quantities.
    model = @slic (; n = 3) begin
        c1::vector[n] ~ std_normal()
        c2::vector[n] ~ std_normal()
        c3::vector[n] ~ std_normal()
        c4::vector[n] ~ std_normal()
        c5::vector[n] ~ std_normal()
        r1::row_vector[n] ~ std_normal()
        r2::row_vector[n] ~ std_normal()
        s1 ~ std_normal()
        s2 ~ std_normal()
        s3 ~ std_normal()
        H = hcat(c1, c2, c3, c4, c5)
        C = append_col(c1, hcat(c2, c3), c4, c5)
        R = append_col(s1, r1, s2, s3, r2)
        V = append_row(c1, s1, s2, c2, s3)
        W = append_row(r1, r2, append_row(r1, r2), r1)
        B = [c5 c4 c3 c2 c1]
    end
    sm = instantiate(stan_model(model)).model
    names = BridgeStan.param_names(sm; include_tp = true, include_gq = true)
    x = collect(range(-1.0, 1.0; length = BridgeStan.param_unc_num(sm)))
    d = Dict(zip(names, BridgeStan.param_constrain(sm, x; include_tp = true, include_gq = true,
        rng = BridgeStan.StanRNG(sm, 1))))
    vec_of(stem, len) = [d["$stem.$i"] for i in 1:len]
    mat_of(stem, r, c) = [d["$stem.$i.$j"] for i in 1:r, j in 1:c]
    c = [vec_of("c$k", 3) for k in 1:5]
    r1, r2 = permutedims(vec_of("r1", 3)), permutedims(vec_of("r2", 3))
    s1, s2, s3 = d["s1"], d["s2"], d["s3"]
    @test mat_of("H", 3, 5) == reduce(Base.hcat, c)
    @test mat_of("C", 3, 5) == reduce(Base.hcat, c)
    @test permutedims(vec_of("R", 9)) == Base.hcat(s1, r1, s2, s3, r2)
    @test vec_of("V", 9) == vcat(c[1], s1, s2, c[2], s3)
    @test mat_of("W", 5, 3) == vcat(r1, r2, r1, r2, r1)
    @test mat_of("B", 3, 5) == reduce(Base.hcat, reverse(c))
end
