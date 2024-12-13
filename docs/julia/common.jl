cd(dirname(Base.active_project()))
using LinearAlgebra
using DataFrames, StanBlocks, Markdown, PosteriorDB, StanLogDensityProblems, LogDensityProblems, Statistics, OrderedCollections, PrettyTables, Serialization, Chairmarks, Enzyme, BridgeStan, Markdown, Pkg, Mooncake
using Logging, Test, LinearAlgebra
using Statistics, StanBlocks, PosteriorDB, Distributions, Random, OrdinaryDiffEq
using StatsPlots
getsampletimes(x::Chairmarks.Benchmark) = getfield.(x.samples, :time); 
const ENZYME_VERSION = filter(x->x.second.name=="Enzyme", Pkg.dependencies()) |> x->first(x)[2].version
const MOONCAKE_VERSION = filter(x->x.second.name=="Mooncake", Pkg.dependencies()) |> x->first(x)[2].version
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
        rand!(d1, w1)
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
cache_on_disc(x) = x in (:lpdf_difference, :lpdf_accuracy, :lpdf_comparison, :enzyme_accuracy, :mooncake_accuracy, :gradient_comparison)#, :df_row)
Base.getproperty(e::PosteriorEvaluation, x::Symbol) = if hasfield(PosteriorEvaluation, x) 
    getfield(e, x)
else
    if hasproperty(e.cache, x)
        getproperty(e.cache, x)
    else
        rv = if cache_on_disc(x)
            path = joinpath("cache", "$(e.posterior_name)_$x.sjl")
            if isfile(path)
                Serialization.deserialize(path)
            else
                Serialization.serialize(path, nothing)
                rv = try
                    println("Generating ", path, "...")
                    compute_property(e, Val(x))
                catch err
                    @error err
                    # rethrow()
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
compute_property(e, ::Val{:dimension}) = LogDensityProblems.dimension(e.julia_lpdf)
compute_property(e, ::Val{:positions}; m=200) = randn((e.dimension, m))
compute_property(e, ::Val{:julia_lpdfs}) = map(e.julia_lpdf, eachcol(e.positions))
compute_property(e, ::Val{:stan_lpdfs}) = map(e.stan_lpdf, eachcol(e.positions))
compute_property(e, ::Val{:finite_idxs1}) = filter(i->isfinite(e.julia_lpdfs[i]) && isfinite(e.stan_lpdfs[i]), 1:size(e.positions, 2))
finite_median_difference(x, y) = UncertainStatistic(median, (filter(isfinite, norm.(x.-y))))
finite_relative_difference(x, y) = UncertainStatistic(median, (filter(isfinite, norm.(x.-y)./(max.(norm.(x), norm.(y))))))
compute_property(e, ::Val{:lpdf_difference}) = finite_median_difference(e.julia_lpdfs, e.stan_lpdfs)
compute_property(e, ::Val{:lpdf_accuracy}) = finite_relative_difference(
    e.julia_lpdfs .+ median(filter(isfinite, e.stan_lpdfs - e.julia_lpdfs)), 
    e.stan_lpdfs
)
compute_property(e, ::Val{:usable}) = !isnothing(e.lpdf_accuracy) && e.lpdf_accuracy <= (e.posterior_name in ("sir-sir","one_comp_mm_elim_abs-one_comp_mm_elim_abs", "soil_carbon-soil_incubation", "hudson_lynx_hare-lotka_volterra") ? 1e-4 : 1e-8)
compute_property(e, ::Val{:julia_lpdf_benchmark}) = (@be randn(e.dimension) e.julia_lpdf)
compute_property(e, ::Val{:stan_lpdf_benchmark}) = (@be randn(e.dimension) e.stan_lpdf)
compute_property(e, ::Val{:lpdf_comparison}) = begin 
    (;julia_lpdf_benchmark, stan_lpdf_benchmark) = e
    min_time = min(mean(julia_lpdf_benchmark).time, mean(stan_lpdf_benchmark).time)
    (;
        julia_lpdf_times=UncertainStatistic(mean, getsampletimes(julia_lpdf_benchmark) ./ min_time), 
        julia_lpdf_allocs=mean(julia_lpdf_benchmark).allocs, 
        stan_lpdf_times=UncertainStatistic(mean, getsampletimes(stan_lpdf_benchmark) ./ min_time)
    )
end

compute_property(e, ::Val{:stan_gradient!}) = begin
    (;stan_problem) = e
    ((g,x),)->(BridgeStan.log_density_gradient!(stan_problem.model, x, g))
end
compute_property(e, ::Val{:stan_gradient}) = x->e.stan_gradient!((zero(x), collect(x)))[2]
compute_property(e, ::Val{:enzyme!}) = begin
    (;julia_lpdf) = e
    ((g,x),)->(Enzyme.autodiff(
        Enzyme.set_runtime_activity(Enzyme.ReverseWithPrimal), Enzyme.Const(julia_lpdf), 
        Enzyme.Active, 
        Enzyme.Duplicated(x, g)
    )[2], g)
end
compute_property(e, ::Val{:enzyme}) = x->e.enzyme!((zero(x), collect(x)))[2]
compute_property(e, ::Val{:mooncake!}) = begin
    (;julia_lpdf) = e
    rule = Mooncake.build_rrule(julia_lpdf, randn(e.dimension))
    mooncake_lpdf = Mooncake.CoDual(julia_lpdf, zero_tangent(julia_lpdf))
    ((g,x),)->(Mooncake.__value_and_gradient!!(
        rule, mooncake_lpdf, Mooncake.CoDual(x, g)
    )[1], g)
end
compute_property(e, ::Val{:mooncake}) = x->e.mooncake!((zero(x), collect(x)))[2]
compute_property(e, ::Val{:enzymes}) = mapreduce(e.enzyme, hcat, eachcol(e.positions))
compute_property(e, ::Val{:stan_gradients}) = mapreduce(e.stan_gradient, hcat, eachcol(e.positions))
compute_property(e, ::Val{:mooncakes}) = mapreduce(e.mooncake, hcat, eachcol(e.positions))
compute_property(e, ::Val{:enzyme_accuracy}) = finite_relative_difference(eachcol(e.enzymes), eachcol(e.stan_gradients))
compute_property(e, ::Val{:mooncake_accuracy}) = finite_relative_difference(eachcol(e.mooncakes), eachcol(e.stan_gradients))

compute_property(e, ::Val{:enzyme_benchmark}) = (@be (randn(e.dimension),zeros(e.dimension)) e.enzyme!)
compute_property(e, ::Val{:stan_gradient_benchmark}) = (@be (randn(e.dimension),zeros(e.dimension)) e.stan_gradient!)
compute_property(e, ::Val{:mooncake_benchmark}) = (@be (randn(e.dimension),zeros(e.dimension)) e.mooncake!)
compute_property(e, ::Val{:gradient_comparison}) = begin 
    (;enzyme_benchmark, stan_gradient_benchmark, mooncake_benchmark) = e
    min_time = minimum(b->mean(b).time, (enzyme_benchmark, stan_gradient_benchmark, mooncake_benchmark))
    # min_time = min(mean(enzyme_benchmark).time, mean(stan_gradient_benchmark).time)
    (;
        enzyme_times=UncertainStatistic(mean, getsampletimes(enzyme_benchmark) ./ min_time), 
        enzyme_allocs=mean(enzyme_benchmark).allocs, 
        mooncake_times=UncertainStatistic(mean, getsampletimes(mooncake_benchmark) ./ min_time), 
        mooncake_allocs=mean(mooncake_benchmark).allocs, 
        stan_gradient_times=UncertainStatistic(mean, getsampletimes(stan_gradient_benchmark) ./ min_time),
        ENZYME_VERSION, MOONCAKE_VERSION
    )
end
compute_property(e, ::Val{:df_row}) = begin 
    (;posterior_name, dimension, lpdf_difference, lpdf_accuracy, usable) = e
    row = map(x->something(x, missing), (;posterior_name, dimension, lpdf_difference, lpdf_accuracy, usable))
    usable || return row
    enzyme_crashed = isnothing(e.enzyme_accuracy)
    merge(
        row,
        (;enzyme_accuracy=something(e.enzyme_accuracy, "FAILED"), mooncake_accuracy=something(e.mooncake_accuracy, "FAILED")),  
        something(e.lpdf_comparison, (;)),
        !enzyme_crashed ? something(e.gradient_comparison, (;)) : (;)
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