@testitem "slic: scalar-token RNGs support prior and observed predictions" tags=[:slic, :regression, :stanc] begin
    using StanBlocks
    const SB = StanBlocks
    @deffun begin
        @lpxf token_normal_lpdf(y::real, mu::real)::real = normal_lpdf(y, mu, 1.)
        token_normal_lpdfs(y::real, mu::real)::real = token_normal_lpdf(y, mu)
        token_normal_rng(real, mu::real)::real = normal_rng(mu, 1.)
        @lhs @lpxf token_count_lpmf(y::int, p::real)::real = bernoulli_lpmf(y, p)
        token_count_lpmfs(y::int, p::real)::real = token_count_lpmf(y, p)
        token_count_rng(int, p::real)::int = bernoulli_rng(p)
        @lpxf dual_lpdf(y::real, mu::real)::real = normal_lpdf(y, mu, 1.)
        dual_lpdfs(y::real, mu::real)::real = dual_lpdf(y, mu)
        dual_rng(mu::real)::real = normal_rng(mu, 1.)
        dual_rng(real, mu::real)::real = normal_rng(mu, 9.)
        @lpxf token_zero_lpdf(y::real)::real = normal_lpdf(y, 0., 1.)
        token_zero_lpdfs(y::real)::real = token_zero_lpdf(y)
        token_zero_rng(real)::real = normal_rng(0., 1.)
    end
    function checked(m)
        code = stan_code(m)
        result = stanc_check(code; warn_pedantic=false)
        @test result.ok
        result.ok || println(result.output)
        code
    end
    real_model = @slic begin
        y ~ token_normal(0.25)
        return y
    end
    @test family_support(token_normal, :(mu::real)) ==
        (; kind=:continuous, type=:real, size=(), support=(;))
    for m in (real_model, real_model(; y=0.5))
        code = checked(m)
        @test occursin("token_normal_real_rng(0.25)", code)
        @test !occursin("tokenof", code)
    end
    count_model = @slic begin
        y ~ token_count(0.3)
        return y
    end
    for m in (count_model, count_model(; y=1))
        @test occursin("token_count_int_rng(0.3)", checked(m))
    end
    dual_model = @slic begin
        y ~ dual(0.25)
        return y
    end
    for m in (dual_model, dual_model(; y=0.5))
        code = checked(m)
        @test occursin("dual_rng(0.25)", code)
        @test !occursin("dual_real_rng", code)
    end
    @test occursin("token_zero_real_rng()", checked(@slic y ~ token_zero()))
    # Native scalar calls and container draws retain their original signatures.
    @test occursin("normal_rng(0.0, 1.0)", checked(@slic y ~ normal(0., 1.)))
    @test occursin("std_normal_rng()", checked(@slic y ~ std_normal()))
    @test occursin("normal_vector_rng", checked(@slic y::vector[3] ~ normal(0., 1.)))
end

@testitem "slic: scalar-token predictive selectors and combinators" tags=[:slic, :regression, :stanc] begin
    using StanBlocks
    const SB = StanBlocks
    @deffun begin
        @lhs @lpxf token_law_lpdf(y::real, mu::real)::real = normal_lpdf(y, mu, 1.)
        token_law_lpdfs(y::real, mu::real)::real = token_law_lpdf(y, mu)
        token_law_lcdf(y::real, mu::real)::real = normal_lcdf(y, mu, 1.)
        token_law_lccdf(y::real, mu::real)::real = normal_lccdf(y, mu, 1.)
        token_law_rng(real, mu::real)::real = normal_rng(mu, 1.)
        token_law_rng(vector[n], mu::real)::vector[n] = begin
            out::vector[n]
            for i in 1:n
                out[i] = token_law_rng(real, mu)
            end
            out
        end
        selected_rng(mu::real)::real = predictive(token_law, mu)
        explicit_rng(mu::real)::real = predictive(token_law, real, mu)
    end
    for rhs in (
        :(token_law(0.25)),
        :(weighted(token_law, 2., 0.25)),
        :(truncated(token_law, 0.25; lower=-1.)),
        :(censored(token_law, 0.25; lower=-1.)),
        :(censored(token_law, 0.25; upper=1.)),
        :(interval_censored(token_law, -1., 1., 0.25)),
    )
        for (data, observations) in ((Dict{Symbol,Any}(), ()),
                                     (Dict{Symbol,Any}(:y=>0.5), (:y,)),
                                     (Dict{Symbol,Any}(), (:y,)))
            # Weighted and censored laws apply to observations, not priors.
            rhs.args[1] in (:weighted, :censored, :interval_censored) && isempty(observations) && continue
            m = SB.SlicModel(Expr(:block, Expr(:call, :~, :y, rhs)), data, @__MODULE__, observations)
            code = stan_code(m)
            result = stanc_check(code; warn_pedantic=false)
            result.ok || println(result.output)
            @test result.ok
            @test occursin("token_law_real_rng", code)
            @test !occursin("predictive(", code)
        end
    end
    for call in (:(selected_rng(0.25)), :(explicit_rng(0.25)),
                 :(predictive(token_law, vector[3], 0.25)))
        m = SB.SlicModel(Expr(:(=), :y, call), Dict{Symbol,Any}(), @__MODULE__)
        code = stan_code(m)
        result = stanc_check(code; warn_pedantic=false)
        result.ok || println(result.output)
        @test result.ok
    end
    # No-argument predictive remains a function selector, not an implicit draw.
    @test SB.rng_expr(token_law) === token_law_rng
    @test occursin("token_law_vector_rng", stan_code(@slic y::vector[3] ~ token_law(0.25)))
end

@testitem "slic: value-family scalar-token RNGs and plain precedence" tags=[:slic, :regression, :stanc] begin
    using StanBlocks
    const SB = StanBlocks
    for discrete in (false, true), both in (false, true)
        name = discrete ? :value_count : :value_normal
        density = discrete ?
            :(value_count_lpmf(y::int, p::real)::real = bernoulli_lpmf(y, p)) :
            :(value_normal_lpdf(y::real, p::real)::real = normal_lpdf(y, p, 1.))
        pointwise = discrete ?
            :(value_count_lpmfs(y::int, p::real)::real = bernoulli_lpmf(y, p)) :
            :(value_normal_lpdfs(y::real, p::real)::real = normal_lpdf(y, p, 1.))
        token_rng = discrete ?
            :(value_count_rng(int, p::real)::int = bernoulli_rng(p)) :
            :(value_normal_rng(real, p::real)::real = normal_rng(p, 1.))
        plain_rng = discrete ?
            :(value_count_rng(p::real)::int = bernoulli_rng(p)) :
            :(value_normal_rng(p::real)::real = normal_rng(p, 1.))
        family = ValueFamily(name, discrete ? :lpmf : :lpdf, density, pointwise,
            both ? [token_rng, plain_rng] : token_rng, @__MODULE__)
        for weighted in (false, true), observed in (false, true)
            weighted && !observed && continue
            rhs = weighted ? Expr(:call, :weighted, family, 2., 0.3) : Expr(:call, family, 0.3)
            data = observed ? Dict{Symbol,Any}(:y => discrete ? 1 : 0.5) : Dict{Symbol,Any}()
            m = SB.SlicModel(Expr(:call, :~, :y, rhs), data, @__MODULE__)
            code = stan_code(m)
            result = stanc_check(code; warn_pedantic=false)
            result.ok || println(result.output)
            @test result.ok
            token_name = string(name, discrete ? "_int_rng" : "_real_rng")
            @test occursin(token_name, code) == !both
        end
    end
end

@testitem "slic: scalar RNG fallback does not hide matching-body errors" tags=[:slic, :regression] begin
    using StanBlocks
    const SB = StanBlocks
    @deffun begin
        @lhs @lpxf broken_draw_lpdf(y::real, mu::real)::real = normal_lpdf(y, mu, 1.)
        broken_draw_rng(mu::real) = undefined_rng_body(mu)
        broken_draw_rng(real, mu::real)::real = normal_rng(mu, 1.)
    end
    msg = try
        stan_code(@slic y ~ broken_draw(0.25))
        ""
    catch err
        sprint(showerror, err)
    end
    @test occursin("undefined_rng_body", msg)
    family = ValueFamily(:bad_value, :lpdf,
        :(bad_value_lpdf(y::real, mu::real)::real = normal_lpdf(y, mu, 1.)),
        :(bad_value_lpdfs(y::real, mu::real)::real = normal_lpdf(y, mu, 1.)),
        [:(bad_value_rng(mu::real) = undefined_value_rng_body(mu)),
         :(bad_value_rng(real, mu::real)::real = normal_rng(mu, 1.))], @__MODULE__)
    msg = try
        stan_code(SB.SlicModel(Expr(:call, :~, :y, Expr(:call, family, 0.25)),
            Dict{Symbol,Any}(), @__MODULE__))
        ""
    catch err
        sprint(showerror, err)
    end
    @test occursin("undefined_value_rng_body", msg)
end
