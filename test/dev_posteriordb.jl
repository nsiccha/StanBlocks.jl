# Dev script for iterating on posteriordb model implementations.
# Caches transpilation/compilation results to disk via DynamicObjects.jl.
#
# Usage:
#   julia --project=test test/dev_posteriordb.jl              # transpile check (fast)
#   julia --project=test test/dev_posteriordb.jl compile      # compile check (slow, cached)
#   julia --project=test test/dev_posteriordb.jl iscorrect    # gradient correctness check (slow, cached)

using Test, Statistics
using StanBlocks, BridgeStan, StanLogDensityProblems, JSON
using DynamicObjects, LogDensityProblems
import StanBlocks.stan: transpiles, compiles
import PosteriorDB

const pdb = PosteriorDB.database()

function _gradient_check(slic, posterior_name; n_points=10, atol=1e-4)
    slic_problem = StanBlocks.stan.instantiate(StanBlocks.stan.stan_model(slic))

    post = PosteriorDB.posterior(pdb, posterior_name)
    ref_dir = joinpath(tempdir(), "posteriordb_ref")
    mkpath(ref_dir)
    ref_problem = StanLogDensityProblems.StanProblem(
        post, ref_dir;
        nan_on_error=true, make_args=["STAN_THREADS=true"], warn=false
    )

    D = LogDensityProblems.dimension(slic_problem)
    D_ref = LogDensityProblems.dimension(ref_problem)
    D != D_ref && return (ok=false, error="dimension mismatch: slic=$D, ref=$D_ref")

    max_diff = 0.0
    for _ in 1:n_points
        θ = 0.1 * randn(D)  # small values to avoid numerical blowup from exp(θ) transforms
        _, g_slic = LogDensityProblems.logdensity_and_gradient(slic_problem, θ)
        _, g_ref = LogDensityProblems.logdensity_and_gradient(ref_problem, θ)
        (any(isnan, g_slic) || any(isnan, g_ref)) && continue
        max_diff = max(max_diff, maximum(abs.(g_slic .- g_ref)))
    end
    max_diff <= atol ? (ok=true, error=nothing) : (ok=false, error="max gradient diff = $max_diff (atol=$atol)")
end

@dynamicstruct struct DevCache
    cache_path = joinpath(@__DIR__, "cache", "posteriordb")
    @cached transpile[model_expr, data] = try
        transpiles(StanBlocks.stan.SlicModel(model_expr, data))
        (ok=true, error=nothing)
    catch e
        (ok=false, error=first(split(sprint(showerror, e), "\n")))
    end
    @cached compile[model_expr, data] = try
        compiles(StanBlocks.stan.SlicModel(model_expr, data))
        (ok=true, error=nothing)
    catch e
        (ok=false, error=first(split(sprint(showerror, e), "\n")))
    end
    @cached iscorrect[model_expr, data, posterior_name] = try
        _gradient_check(StanBlocks.stan.SlicModel(model_expr, data), posterior_name)
    catch e
        (ok=false, error=first(split(sprint(showerror, e), "\n")))
    end
end

# --- Load model definitions (without the @testset) ---

slic_implementation(key; kwargs...) = nothing
slic_implementation(posterior::PosteriorDB.Posterior) = slic_implementation(
    Val(Symbol(PosteriorDB.name(PosteriorDB.model(posterior))));
    Dict([Symbol(k)=>v for (k, v) in pairs(PosteriorDB.load(PosteriorDB.dataset(posterior)))])...
)

let code = read(joinpath(@__DIR__, "posteriordb.jl"), String)
    code = replace(code, r"@testset\s+\"posteriordb\".*"s => "end")
    code = replace(code, r"^import PosteriorDB.*?slic_implementation\(\n.*?\n\)"sm => "")
    include_string(Main, code)
end

# --- Main ---

mode = length(ARGS) >= 1 ? Symbol(ARGS[1]) : :transpile
cache = DevCache()

println("Running $mode checks...")
println()

pass = String[]
fail = Tuple{String,String}[]

for pn in sort(PosteriorDB.posterior_names(pdb))
    post = slic_implementation(PosteriorDB.posterior(pdb, pn))
    isnothing(post) && continue

    result = if mode === :iscorrect
        cache.iscorrect[post.model, post.data, pn]
    else
        getproperty(cache, mode)[post.model, post.data]
    end

    if result.ok
        push!(pass, pn)
    else
        push!(fail, (pn, something(result.error, "unknown")))
        println("  FAIL: $pn")
        println("        $(result.error)")
    end
end

println()
println("$(length(pass)) pass, $(length(fail)) fail out of $(length(pass)+length(fail))")

if !isempty(fail)
    println()
    println("Failed models:")
    for (name, msg) in fail
        println("  $name: $msg")
    end
end
