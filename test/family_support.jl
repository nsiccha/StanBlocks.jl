using TestItemRunner

@testitem "slic: typed symbolic family support" tags=[:slic, :regression] begin
    using StanBlocks
    const SB = StanBlocks
    scalar_real(s) = s.kind === :continuous && s.type === :real && isempty(s.size) && isempty(s.support)
    normal = family_support(SB.stan.normal, :(mu::real), :(sigma::real))
    @test normal == (; kind=:continuous, type=:real, size=(), support=(;))
    @test !scalar_real(family_support(SB.stan.exponential, :(rate::real)))
    @test family_support(SB.stan.exponential, :(rate::real)).support == (; lower=0.)
    @test family_support(SB.stan.uniform, :(lo::real), :(hi::real)).support == (; lower=:lo, upper=:hi)
    @test family_support(SB.stan.uniform, -2., 3.).support == (; lower=-2., upper=3.)
    @test family_support(SB.stan.bernoulli, :(p::real)).kind === :discrete
    vector = family_support(SB.stan.multi_normal, :(mu::vector[n]), :(sigma::matrix[n,n]))
    @test vector.type === :vector && vector.size == (:n,)
    @test !scalar_real(vector)
    @test !scalar_real(family_support(SB.stan.normal, :(mu::vector[n]), 1.))
    @test family_support(SB.stan.dirichlet, :(alpha::vector[n])).type === :simplex

    @deffun begin
        @lhs @lpxf shifted_lpdf(y::real, lo::real)::real = uniform_lpdf(y, lo, lo + 1.)
        shifted_rng(lo::real)::real = uniform_rng(lo, lo + 1.)
        @lhs @lpxf whole_lpdf(y::real, mu::real)::real = normal_lpdf(y, mu, 1.)
        whole_rng(mu::real)::real = normal_rng(mu, 1.)
        @lpxf sized_lpdf(y::vector[n], mu::vector[n])::real = normal_lpdf(y, mu, 1.)
        sized_rng(vector[n], mu::vector[n])::vector[n] = mu
    end
    SB.autokwargs(x::SB.CanonicalExpr{typeof(shifted),<:Tuple{<:SB.StanExpr2{SB.types.real,0}}}) =
        (; lower=x.args[1], upper=SB.stan_call(+, x.args[1], 1.))
    @test scalar_real(family_support(whole, :(mu::real)))
    @test family_support(sized, :(mu::vector[n])).size == (:n,)
    @test SB.autokwargs(SB.CanonicalExpr(shifted, :lo)) == (;)
    @test family_support(shifted, :(lo::real)).support == (; lower=:lo, upper="(lo + 1.0)")
    @test !scalar_real(family_support(shifted, :(lo::real)))
    arg = :(mu::vector[n + 1])
    before = deepcopy(arg)
    @test family_support(SB.stan.multi_normal, arg, :(sigma::matrix[n + 1,n + 1])).size == ("(n + 1)",)
    @test arg == before

    # No arbitrary sample values, no catch-and-empty result, no fallback from
    # an unresolved signature to autotype's permissive scalar default.
    @test_throws ArgumentError family_support(whole, :mu)
    @test_throws ArgumentError family_support(whole, :(mu::anything))
    @test_throws ArgumentError family_support(whole, :(mu::real), :(mu::real))
    @test_throws ArgumentError family_support(whole, :(mu::vector[n]))
    @test_throws Exception family_support(identity, :(mu::real))
    @test_throws Exception family_support(SB.stan.normal, :(mu::real))

    value = ValueFamily(:value_whole, :lpdf,
        :(value_whole_lpdf(y::real, mu::real)::real = normal_lpdf(y, mu, 1.)),
        :(value_whole_lpdfs(y::real, mu::real)::real = normal_lpdf(y, mu, 1.)),
        :(value_whole_rng(mu::real)::real = normal_rng(mu, 1.)), @__MODULE__)
    @test scalar_real(family_support(value, :(mu::real)))
    unbounded = ValueFamily(:unbounded, :lpdf,
        :(unbounded_lpdf(y::real, mu::real)::real = normal_lpdf(y, mu, 1.)),
        :(unbounded_lpdfs(y::real, mu::real)::real = normal_lpdf(y, mu, 1.)),
        :(unbounded_rng(mu::real)::real = normal_rng(mu, 1.)), @__MODULE__;
        support=(; lower=-Inf, upper=Inf))
    @test scalar_real(family_support(unbounded, :(mu::real)))
    @test_throws ArgumentError family_support(SB.stan.multi_normal,
        :(mu::vector[n]), :(sigma::matrix[n,n]), :(n::real))

    # The public entry point sees registrations made after a task was started.
    ready, input = Channel{Nothing}(1), Channel{Any}(1)
    old = Threads.@spawn begin
        put!(ready, nothing)
        family_support(take!(input), :(mu::real))
    end
    take!(ready)
    mod = Module(:LaterFamily)
    Core.eval(mod, :(using StanBlocks))
    Core.eval(mod, :(@deffun @lhs @lpxf later_lpdf(y::real, mu::real)::real = normal_lpdf(y, mu, 1.)))
    put!(input, mod.later)
    @test scalar_real(fetch(old))
    @test all(s -> s.support == (; lower=:lo, upper="(lo + 1.0)"),
        fetch.([Threads.@spawn family_support(shifted, :(lo::real)) for _ in 1:4]))
end

@testitem "slic: same-name ordinary families retain module identity" tags=[:slic, :regression, :stanc] begin
    using StanBlocks
    const SB = StanBlocks
    compiles(code) = begin
        checked = stanc_check(code; warn_pedantic=false)
        checked.ok || println(checked.output)
        checked.ok
    end
    function workspace(scale)
        # Even the modules have the same name. Identity, not a qualified string,
        # distinguishes their definitions.
        mod = Module(:RepeatedFamily)
        Core.eval(mod, :(using StanBlocks))
        Core.eval(mod, quote
            @deffun begin
                @lhs @lpxf shared_lpdf(y::real, mu::real)::real = normal_lpdf(y, mu, $scale)
                shared_lpdfs(y::real, mu::real)::real = shared_lpdf(y, mu)
                shared_rng(mu::real)::real = normal_rng(mu, $scale)
                apply_density(f, y::real, mu::real)::real = density(f, y, mu)
            end
        end)
        mod
    end
    a, b = workspace(1.25), workspace(2.75)
    @test a.shared !== b.shared
    @test SB.lpxf_expr(a.shared) === a.shared_lpdf
    @test SB.rng_expr(b.shared) === b.shared_rng
    body = quote
        mu ~ normal(0., 1.)
        y ~ $(a.shared)(mu)
        z ~ $(b.shared)(mu)
    end
    model = SB.SlicModel(body, Dict{Symbol,Any}(:y=>0.1, :z=>0.2), @__MODULE__)
    code = stan_code(model)
    @test compiles(code)
    @test occursin("shared__2_lpdf", code)
    @test occursin("z ~ shared__2(mu)", code)
    @test occursin("shared__2_rng(mu)", code)
    @test occursin("normal_lpdf(y | mu, 1.25)", code)
    @test occursin("normal_lpdf(y | mu, 2.75)", code)
    descriptor = stan_descriptor(model)
    defs = filter(d -> d.binding === :shared_lpdf, descriptor.definitions)
    @test length(defs) == 2
    @test length(unique(d.name for d in defs)) == 2
    @test all(d -> occursin(d.source, code), descriptor.definitions)
    @test all(==(code), fetch.([Threads.@spawn stan_code(model) for _ in 1:4]))

    # The producer path: actual ordinary family values inside a runtime family,
    # both directly and as function tokens in a specialized helper.
    f, g, apply = a.shared, b.shared, a.apply_density
    value = ValueFamily(:joined, :lpdf,
        :(joined_lpdf(y::real, mu::real)::real = log_sum_exp(
            log(0.5) + $apply($f, y, mu), log(0.5) + $apply($g, y, mu))),
        :(joined_lpdfs(y::real, mu::real)::real = joined_lpdf(y, mu)),
        :(joined_rng(mu::real)::real = begin
            if bernoulli_rng(0.5) == 1
                return predictive($f, mu)
            end
            predictive($g, mu)
        end), @__MODULE__)
    mixture = SB.SlicModel(quote
        mu ~ normal(0., 1.)
        y ~ $value(mu)
    end, Dict{Symbol,Any}(:y=>0.1), @__MODULE__)
    joined = stan_code(mixture)
    @test compiles(joined)
    @test occursin("apply_density_shared__2", joined)
    @test occursin("shared__2_rng(mu)", joined)
    @test !occursin("predictive(", joined)
    @test length(stan_descriptor(mixture).definitions) >= 7

    # A custom function sharing a builtin name keeps the builtin's Stan name
    # available to its own body; an explicit density call uses the same alias.
    custom = Module(:NativeName)
    Core.eval(custom, :(using StanBlocks))
    Core.eval(custom, quote
        @deffun begin
            @lhs @lpxf normal_lpdf(y::real, mu::real)::real = normal_lpdf(y, mu, 3.)
            normal_rng(mu::real)::real = normal_rng(mu, 3.)
        end
    end)
    # Bare symbols retain builtin-first lookup. Embed the actual custom
    # callable when selecting its density explicitly.
    Core.eval(custom, :(@deffun normal_lpdfs(y::real, mu::real)::real = $(custom.normal_lpdf)(y, mu)))
    native_collision = SB.SlicModel(quote
        mu ~ normal(0., 1.)
        y ~ $(custom.normal)(mu)
    end, Dict{Symbol,Any}(:y=>0.1), @__MODULE__)
    native_code = stan_code(native_collision)
    @test compiles(native_code)
    @test occursin("normal__2_lpdf", native_code)
    @test occursin("normal_lpdf(y | mu, 3.0)", native_code)
end
