cd(dirname(Base.active_project()))
using LinearAlgebra
using DataFrames, StanBlocks, Markdown, PosteriorDB, StanLogDensityProblems, LogDensityProblems, Statistics, OrderedCollections, PrettyTables, Serialization, Chairmarks, Enzyme, BridgeStan, Markdown, Pkg, Mooncake
using Logging, Test, LinearAlgebra
using Statistics, StanBlocks, PosteriorDB, Distributions, Random, OrdinaryDiffEq
using StatsPlots
getsampletimes(x::Chairmarks.Benchmark) = getfield.(x.samples, :time); 
const ENZYME_VERSION = filter(x->x.second.name=="Enzyme", Pkg.dependencies()) |> x->first(x)[2].version
const MOONCAKE_VERSION = filter(x->x.second.name=="Mooncake", Pkg.dependencies()) |> x->first(x)[2].version
const BRIDGESTAN_VERSION = filter(x->x.second.name=="BridgeStan", Pkg.dependencies()) |> x->first(x)[2].version
const VERSION_STRING = "Julia $VERSION + Enzyme $ENZYME_VERSION" 
plotlyjs()
BLAS.set_num_threads(1)
ENV["DATAFRAMES_ROWS"] = 500
mad(x) = median(abs.(x .- median(x)))
struct UncertainStatistic{F,V}
    f::F
    vals::V
end
se(s::UncertainStatistic{typeof(mean)}) = std(s.vals)/sqrt(length(s.vals))
se(s::UncertainStatistic{typeof(median)}) = mad(s.vals)/sqrt(length(s.vals))
Base.isless(s::UncertainStatistic, x::Float64) = isless(s.f(s.vals) + 2 * se(s), x)
Base.show(io::IO, s::UncertainStatistic) = print(io, round(s.f(s.vals); sigdigits=2), " ± ", round(se(s); sigdigits=2))
val(s::UncertainStatistic) = s.f(s.vals)
val(::Missing) = missing
ratio_of_means((v1,v2)) = mean(v1)/mean(v2)
ci(s::UncertainStatistic{typeof(ratio_of_means)}; q=.05, n=round(Int, 10 / q)) = begin 
    v1, v2 = s.vals
    m1, m2 = length(v1), length(v2)
    d1, d2 = Dirichlet(m1, 1.), Dirichlet(m2, 1.)
    w1, w2 = zeros(m1), zeros(m2)
    hist = zeros(n)
    for i in eachindex(hist)
        rand!(d2, w2)
        hist[i] = dot(w1, v1) / dot(w2, v2)
    end
    quantile(hist, (q, 1-q))
end

jimplementations_lines = readlines("PosteriorDBExt.jl")
posterior_dimension(posterior_name) = StanBlocks.dimension(StanBlocks.julia_implementation(PosteriorDB.posterior(PosteriorDB.database(), posterior_name)))
implementations_string(posterior_name) = begin
    data, model = split(posterior_name, "-")
    line = findfirst(line->contains(line, model), jimplementations_lines)
    Markdown.parse("[Stan](https://github.com/stan-dev/posteriordb/tree/master/posterior_database/models/stan/$model.stan), [Julia](https://github.com/nsiccha/StanBlocks.jl/blob/main/ext/PosteriorDBExt.jl#L$line)")
end
nan_on_error(f, x) = try
    f(x)
catch e
    NaN 
end
nan_on_error(f) = Base.Fix1(nan_on_error, f)
mkpath("cache")

const pdb = PosteriorDB.database()
posterior_names = PosteriorDB.posterior_names(pdb)

# julia_pdb_implementation(posterior_name; pdb=PosteriorDB.database()) = StanBlocks.julia_implementation(
#     PosteriorDB.posterior(pdb, posterior_name)
# )

# import PlotlyJS
# using Plots


cached(f, args...; path, kwargs...) = begin 
    mkpath(dirname(path))
    isfile(path) && return Serialization.deserialize(path)
    rv = try
        print("Generating...", path, ": ")
        f(args...; kwargs...)
    catch e
        print("\n")
        @error e
        nothing
    end
    println(rv)
    Serialization.serialize(path, rv)
    rv
end

begin
mutable struct PosteriorEvaluation
    posterior_name::String
    cache::NamedTuple
end
PosteriorEvaluation(posterior_name) = PosteriorEvaluation(posterior_name, (;))
cache_on_disc(x) = x in (:lpdf_difference, :lpdf_accuracy, :n_evals, :lpdf_comparison, :enzyme_accuracy, :mooncake_accuracy, :gradient_comparison, :allocations)#, :df_row)
Base.getproperty(e::PosteriorEvaluation, x::Symbol) = if hasfield(PosteriorEvaluation, x) 
    getfield(e, x)
else
    if hasproperty(e.cache, x)
        getproperty(e.cache, x)
    else
        rv = if cache_on_disc(x)
            path = joinpath("cache", "$(e.posterior_name)_$x.sjl")
            if isfile(path)
                old_rv = Serialization.deserialize(path)
                rv = compute_property(e, Val(x), old_rv)
                if isnothing(rv)
                    old_rv
                else
                    Serialization.serialize(path, rv)
                    rv
                end
            else
                Serialization.serialize(path, nothing)
                rv = try
                    println("Generating ", path, "...")
                    compute_property(e, Val(x))
                catch err
                    @error err
                    rethrow()
                    nothing
                end
                println(rv)
                Serialization.serialize(path, rv)
                rv
            end 
        else 
            compute_property(e, Val(x))
        end
        e.cache = merge(e.cache, (;(x=>rv)))
        rv
    end
end 
compute_property(e, ::Val, old_rv) = nothing 
compute_property(e, ::Val{:posterior}) = PosteriorDB.posterior(pdb, e.posterior_name)
compute_property(e, ::Val{:stan_path}) = PosteriorDB.path(PosteriorDB.implementation(PosteriorDB.model(e.posterior), "stan"))
compute_property(e, ::Val{:stan_problem}) = with_logger(ConsoleLogger(stderr, Logging.Error)) do 
    StanProblem(
        e.stan_path, 
        PosteriorDB.load(PosteriorDB.dataset(e.posterior), String);
        nan_on_error=true
    )
end
compute_property(e, ::Val{:stan_lpdf}) = begin
    (;stan_problem) = e
    Base.Fix1(LogDensityProblems.logdensity, stan_problem)
end
compute_property(e, ::Val{:julia_lpdf}) = StanBlocks.julia_implementation(e.posterior)
compute_property(e, ::Val{:julia_blpdf}) = begin 
    (;julia_lpdf) = e
    x = zeros(e.dimension)
    (rng) -> julia_lpdf(randn!(rng, x))
end
compute_property(e, ::Val{:stan_blpdf}) = begin 
    (;stan_lpdf) = e
    x = zeros(e.dimension)
    (rng) -> stan_lpdf(randn!(rng, x))
end
compute_property(e, ::Val{:dimension}) = LogDensityProblems.dimension(e.julia_lpdf)
compute_property(e, ::Val{:x}) = zeros(e.dimension)
compute_property(e, ::Val{:g1}) = zeros(e.dimension)
compute_property(e, ::Val{:g2}) = zeros(e.dimension)
compute_property(e, ::Val{:lpdf_difference}) = adaptive_median() do 
    x = randn!(e.x)
    (e.julia_lpdf(x)-e.stan_lpdf(x))
end
compute_property(e, ::Val{:lpdf_accuracy}) = adaptive_median() do 
    x = randn!(e.x)
    norm(e.julia_lpdf(x)-e.stan_lpdf(x)-e.lpdf_difference)/nonzero(norm(e.stan_lpdf(x)))
end
compute_property(e, ::Val{:enzyme_accuracy}) = adaptive_median() do 
    x = randn!(e.x)
    e.stan_gradient!(x, e.g1)
    e.enzyme!(x, e.g2)
    norm(e.g1-e.g2)/nonzero(max(norm(e.g1),norm(e.g2)))
end
compute_property(e, ::Val{:mooncake_accuracy}) = adaptive_median() do 
    x = randn!(e.x)
    e.stan_gradient!(x, e.g1)
    e.mooncake!(x, e.g2)
    norm(e.g1-e.g2)/nonzero(max(norm(e.g1),norm(e.g2)))
end
compute_property(e, ::Val{:usable}) = !isnothing(e.lpdf_accuracy) && e.lpdf_accuracy <= (e.posterior_name in ("sir-sir","one_comp_mm_elim_abs-one_comp_mm_elim_abs", "soil_carbon-soil_incubation", "hudson_lynx_hare-lotka_volterra") ? 1e-4 : 1e-8)
compute_property(e, ::Val{:n_evals}) = begin 
    b = (@be Xoshiro(0) _ e.stan_blpdf _)
    display(b)
    trunc(Int, b.samples[1].evals)
end
compute_property(e, ::Val{:lpdf_comparison}) = begin 
    (;means=adaptive_mean3(
        IterableDistribution(Xoshiro(0), RuntimeDistribution2(e.julia_blpdf, e.n_evals)),
        IterableDistribution(Xoshiro(0), RuntimeDistribution2(e.stan_blpdf, e.n_evals)),
    ), BRIDGESTAN_VERSION)
end
compute_property(e, ::Val{:lpdf_comparison}, rv::NamedTuple) = begin 
    all(m->Main.rtol(m) < .01, rv.means) && return
    (;means=adaptive_mean3((
        IterableDistribution(Xoshiro(0), RuntimeDistribution2(e.julia_blpdf, e.n_evals)),
        IterableDistribution(Xoshiro(0), RuntimeDistribution2(e.stan_blpdf, e.n_evals)),
    ), rv.means), BRIDGESTAN_VERSION)
end
compute_property(e, ::Val{:gradient_comparison}) = begin 
    (;means=adaptive_mean3(
        IterableDistribution(Xoshiro(0), RuntimeDistribution2(e.stan_bgradient!, e.n_evals)),
        IterableDistribution(Xoshiro(0), RuntimeDistribution2(e.benzyme!, e.n_evals)),
        IterableDistribution(Xoshiro(0), RuntimeDistribution2(e.bmooncake!, e.n_evals)),
    ), BRIDGESTAN_VERSION, ENZYME_VERSION, MOONCAKE_VERSION)
end
compute_property(e, ::Val{:gradient_comparison}, rv::NamedTuple) = begin 
    all(m->Main.rtol(m) < .01, rv.means) && return
    (;means=adaptive_mean3((
        IterableDistribution(Xoshiro(0), RuntimeDistribution2(e.stan_bgradient!, e.n_evals)),
        IterableDistribution(Xoshiro(0), RuntimeDistribution2(e.benzyme!, e.n_evals)),
        IterableDistribution(Xoshiro(0), RuntimeDistribution2(e.bmooncake!, e.n_evals)),
    ), rv.means), BRIDGESTAN_VERSION, ENZYME_VERSION, MOONCAKE_VERSION)
end
compute_property(e, ::Val{:allocations}) = (;e.julia_allocations, e.stan_allocations, e.stan_gradient_allocations, e.enzyme_allocations, e.mooncake_allocations)
allocations(f, x) = begin 
    f(x)
    @allocations f(x)
end
compute_property(e, ::Val{:julia_allocations}) = allocations(e.julia_blpdf, Xoshiro(0))
compute_property(e, ::Val{:stan_allocations}) = allocations(e.stan_blpdf, Xoshiro(0))
compute_property(e, ::Val{:stan_gradient_allocations}) = allocations(e.stan_bgradient!, Xoshiro(0))
compute_property(e, ::Val{:enzyme_allocations}) = allocations(e.benzyme!, Xoshiro(0))
compute_property(e, ::Val{:mooncake_allocations}) = allocations(e.bmooncake!, Xoshiro(0))

compute_property(e, ::Val{:stan_gradient!}) = begin
    (;stan_problem) = e
    (x, g)->(BridgeStan.log_density_gradient!(stan_problem.model, x, g))
end
# compute_property(e, ::Val{:stan_gradient}) = x->e.stan_gradient!((zero(x), collect(x)))[2]
compute_property(e, ::Val{:enzyme!}) = begin
    (;julia_lpdf) = e
    (x,g)->(Enzyme.autodiff(
        Enzyme.set_runtime_activity(Enzyme.ReverseWithPrimal), Enzyme.Const(julia_lpdf), 
        Enzyme.Active, 
        Enzyme.Duplicated(x, (g .= 0.))
    )[2], g)
end
# compute_property(e, ::Val{:enzyme}) = x->e.enzyme!((zero(x), collect(x)))[2]
compute_property(e, ::Val{:mooncake!}) = begin
    (;julia_lpdf) = e
    rule = Mooncake.build_rrule(julia_lpdf, randn(e.dimension))
    mooncake_lpdf = Mooncake.CoDual(julia_lpdf, zero_tangent(julia_lpdf))
    (x,g)->(Mooncake.__value_and_gradient!!(
        rule, mooncake_lpdf, Mooncake.CoDual(x, (g .= 0.))
    )[1], g)
end
# compute_property(e, ::Val{:mooncake}) = x->e.mooncake!((zero(x), collect(x)))[2]
compute_property(e, ::Val{:stan_bgradient!}) = begin 
    (;stan_gradient!) = e
    x, g = zeros(e.dimension), zeros(e.dimension)
    (rng) -> stan_gradient!(randn!(rng, x), g)
end
compute_property(e, ::Val{:benzyme!}) = begin 
    (;enzyme!) = e
    x, g = zeros(e.dimension), zeros(e.dimension)
    (rng) -> enzyme!(randn!(rng, x), g)
end
compute_property(e, ::Val{:bmooncake!}) = begin 
    (;mooncake!) = e
    x, g = zeros(e.dimension), zeros(e.dimension)
    (rng) -> mooncake!(randn!(rng, x), g)
end
compute_property(e, ::Val{:df_row}) = begin 
    (;posterior_name, dimension, lpdf_difference, lpdf_accuracy, usable) = e
    row = map(x->something(x, missing), (;posterior_name, dimension, lpdf_difference, lpdf_accuracy, usable))
    usable || return row
    enzyme_crashed = isnothing(e.enzyme_accuracy)
    merge(
        row,
        (;enzyme_accuracy=something(e.enzyme_accuracy, "FAILED"), mooncake_accuracy=something(e.mooncake_accuracy, "FAILED")),  
        isnothing(e.lpdf_comparison) ? (;) : (;julia_lpdf_times=e.lpdf_comparison.means[1], stan_lpdf_times=e.lpdf_comparison.means[2], e.lpdf_comparison.BRIDGESTAN_VERSION),
        (enzyme_crashed || isnothing(e.gradient_comparison)) ? (;) : (;stan_gradient_times=e.gradient_comparison.means[1], enzyme_times=e.gradient_comparison.means[2], mooncake_times=e.gradient_comparison.means[3], e.gradient_comparison.MOONCAKE_VERSION, e.gradient_comparison.ENZYME_VERSION),
        e.allocations
    )
end
end
pad_missing(rows) = begin 
    all_keys = Set()
    for row in rows
        union!(all_keys, keys(row))
    end
    map(row->merge((;Pair.(all_keys, missing)...), row), rows)
end