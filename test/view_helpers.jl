@testsnippet ViewHelpers begin
    @deffun view_total(v)::real = begin
        total = 0.0
        for j in 1:length(v)
            total += sum(v[j])
        end
        total + sum(v[lastindex(v)])
    end
    @deffun view_forward(v)::real = view_total(v)
    @deffun make_columns(a::real, X::matrix[r,c]) = EachCol(a * X)
    @deffun make_rows(a::real, X::matrix[r,c]) = EachRow(a * X)

    function view_model(view)
        StanBlocks.SlicModel(quote
            a ~ normal(0, 1)
            v = $view(a * X)
            location = view_forward(v)
            y ~ normal(location, 0.8)
        end, Dict{Symbol,Any}(:X => [0.2 -0.4 0.7; 0.9 0.3 -0.1], :y => 0.6), @__MODULE__)
    end
end

@testitem "slic: helper-returned views keep their backing matrix" tags=[:slic, :regression, :stanc] setup=[StanBlocksImports, StanBlocksTestSetup, ViewHelpers] begin
    using .StanBlocksTestSetup: stanc_compiles
    model = @slic (; X=[0.2 -0.4 0.7; 0.9 0.3 -0.1], y=0.6) begin
        a ~ normal(0, 1)
        columns = make_columns(a, X)
        rows = make_rows(a, X)
        location = sum(columns[2]) + sum(rows[1]) + sum(columns.X) + length(rows)
        y ~ normal(location, 0.8)
    end
    code = stan_code(model)
    @test occursin("matrix make_columns(real a, matrix X)", code)
    @test occursin("matrix make_rows(real a, matrix X)", code)
    @test occursin("getindex_EachCol(make_columns(a, X), 2)", code)
    @test occursin("getindex_EachRow(make_rows(a, X), 1)", code)
    @test stanc_compiles(model)
end

@testitem "slic: matrix views cross ordinary helper boundaries" tags=[:slic, :regression, :stanc] setup=[StanBlocksImports, StanBlocksTestSetup, ViewHelpers] begin
    using .StanBlocksTestSetup: stanc_compiles
    for view in (StanBlocks.EachCol, StanBlocks.EachRow)
        model = view_model(view)
        code = stan_code(model)
        @test occursin("real view_forward_", code)
        @test occursin("matrix v", code)
        @test !occursin("tuple(matrix)", code)
        @test !occursin(r"\b(?:v|ec|er)\.1\b", code)
        @test stanc_compiles(model)
    end
end

@testitem "slic: one-field usertypes retain nominal helper dispatch" tags=[:slic, :regression, :stanc] setup=[StanBlocksImports, StanBlocksTestSetup] begin
    using .StanBlocksTestSetup: stanc_compiles
    using StanBlocks: @usertype
    @usertype struct ValueBox
        value::real
    end
    @usertype struct OuterBox
        inner
    end
    @usertype struct ValuePair
        boxed
        offset::real
    end
    @deffun box_value(b::ValueBox)::real = b.value
    @deffun box_value(x::real)::real = x + 1
    @deffun outer_value(o::OuterBox)::real = box_value(o.inner)
    @deffun make_box(x::real) = ValueBox(x + 0.25)
    @deffun pair_value(p::ValuePair)::real = box_value(p.boxed) + p.offset
    model = @slic (; y=0.6) begin
        a ~ normal(0, 1)
        box = make_box(a)
        nested = OuterBox(box)
        pair = ValuePair(nested.inner, a)
        y ~ normal(pair_value(pair) + outer_value(nested) + box_value(a), 1)
    end
    code = stan_code(model)
    @test occursin("real make_box(real x)", code)
    @test occursin("real box_value_ValueBox(real b)", code)
    @test occursin("real outer_value_OuterBox(real o)", code)
    @test occursin("real box_value(real x)", code)
    @test occursin("tuple(real, real)", code)
    @test !occursin("tuple(real)", code)
    @test stanc_compiles(model)
end

@testitem "slic: matrix view helpers preserve values and gradients" tags=[:slic, :regression, :bridgestan] setup=[StanBlocksImports, StanBlocksTestSetup, ViewHelpers] begin
    X = [0.2 -0.4 0.7; 0.9 0.3 -0.1]
    for (view, last_slice) in ((StanBlocks.EachCol, X[:, end]), (StanBlocks.EachRow, X[end, :]))
        coefficient = sum(X) + sum(last_slice)
        reference = @slic (; coefficient, y=0.6) begin
            a ~ normal(0, 1)
            y ~ normal(a * coefficient, 0.8)
        end
        problem = instantiate(view_model(view))
        control = instantiate(reference)
        @test LogDensityProblems.dimension(problem) == 1
        for a in (-0.7, 0.0, 0.4)
            lp, grad = LogDensityProblems.logdensity_and_gradient(problem, [a])
            expected_lp, expected_grad = LogDensityProblems.logdensity_and_gradient(control, [a])
            @test lp ≈ expected_lp atol=1e-12
            @test grad ≈ expected_grad atol=1e-12
            @test only(grad) ≈ -a + coefficient * (0.6 - a * coefficient) / 0.8^2 atol=1e-12
        end
    end

    returned = @slic (; X, y=0.6) begin
        a ~ normal(0, 1)
        columns = make_columns(a, X)
        rows = make_rows(a, X)
        y ~ normal(sum(columns[2]) + sum(rows[1]), 0.8)
    end
    coefficient = sum(X[:, 2]) + sum(X[1, :])
    problem = instantiate(returned)
    for a in (-0.7, 0.0, 0.4)
        lp, grad = LogDensityProblems.logdensity_and_gradient(problem, [a])
        # Stan drops parameter-independent normalizers, including log(0.8).
        @test lp ≈ -a^2 / 2 - ((0.6 - a * coefficient) / 0.8)^2 / 2 atol=1e-12
        @test only(grad) ≈ -a + coefficient * (0.6 - a * coefficient) / 0.8^2 atol=1e-12
    end
end
