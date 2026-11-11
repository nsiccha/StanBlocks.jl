using StanBlocks, LogDensityProblems, Distributions, SHA, TOML, Dates
include("model.jl")

# Deterministic quadrature in the unconstrained coordinate log(sigma).
# The Stan target includes the exp-coordinate Jacobian.
problem = stan_instantiate(model; nan_on_error=false)
@assert LogDensityProblems.dimension(problem) == 1
q = collect(range(log(0.01), log(0.15); length=4001))
lp = [LogDensityProblems.logdensity(problem, [v]) for v in q]
@assert all(isfinite, lp)
weights = exp.(lp .- maximum(lp))
weights ./= sum(weights)
@assert sum(weights[1:10]) + sum(weights[end-9:end]) < 1e-10
sigmas = exp.(q)
weighted_quantile(x, w, probability) = x[searchsortedfirst(cumsum(w), probability)]
quantiles = [weighted_quantile(sigmas, weights, p) for p in (0.025, 0.5, 0.975)]

# Independent normalized density, compared up to a parameter-independent constant.
function reference_density(qvalue)
    sigma = exp(qvalue)
    p = 2 .* cdf.(Normal(), asin.(golf_data.gap ./ golf_data.distance) ./ sigma) .- 1
    logpdf(truncated(Normal(0, 0.1); lower=0), sigma) + qvalue +
        sum(logpdf.(Binomial.(golf_data.attempts, p), golf_data.sunk))
end
checks = q[1:100:end]
deltas = [LogDensityProblems.logdensity(problem, [v]) - reference_density(v) for v in checks]
# At the smallest sigma, nearly unit probabilities amplify differences between
# the Stan and Julia normal-CDF implementations. Compare in log-density units.
@assert maximum(abs.(deltas .- first(deltas))) < 1e-6

# Refine the quadrature once to quantify discretization error in the median.
qfine = collect(range(first(q), last(q); length=8001))
wfine = exp.([LogDensityProblems.logdensity(problem, [v]) for v in qfine] .- maximum(lp))
wfine ./= sum(wfine)
fine_median = weighted_quantile(exp.(qfine), wfine, 0.5)
median_difference = abs(fine_median - quantiles[2])
@assert median_difference < 1e-5

rows = map(eachindex(golf_data.distance)) do i
    p = 2 .* cdf.(Normal(), asin(golf_data.gap / golf_data.distance[i]) ./ sigmas) .- 1
    order = sortperm(p)
    ps, ws = p[order], weights[order]
    n = golf_data.attempts[i]
    # The predictive count distribution integrates both parameter and binomial uncertainty.
    cumulative = [sum(weights .* cdf.(Binomial.(n, p), k)) for k in 0:n]
    lo = (searchsortedfirst(cumulative, 0.025) - 1) / n
    hi = (searchsortedfirst(cumulative, 0.975) - 1) / n
    Dict("distance_ft"=>golf_data.distance[i], "attempts"=>n,
         "observed"=>golf_data.sunk[i]/n,
         "probability_median"=>weighted_quantile(ps, ws, 0.5),
         "replicate_lower"=>lo, "replicate_upper"=>hi)
end
summary = Dict("sigma_rad"=>quantiles, "sigma_deg"=>quantiles .* (180/pi), "predictions"=>rows)
open(joinpath(@__DIR__, "summary.toml"), "w") do io
    TOML.print(io, summary)
end
key = bytes2hex(sha256(vcat(read(joinpath(@__DIR__, "model.jl")), read(@__FILE__))))
receipt = Dict("key"=>key, "method"=>"normalized log-scale quadrature",
    "created_utc"=>string(now(UTC)), "julia"=>string(VERSION),
    "stanblocks_version"=>string(pkgversion(StanBlocks)),
    "stan_source_sha256"=>bytes2hex(sha256(stan_code(model))),
    "summary_sha256"=>bytes2hex(sha256(read(joinpath(@__DIR__, "summary.toml")))),
    "grid_points"=>length(q), "refined_grid_points"=>length(qfine),
    "median_refinement_rad"=>median_difference,
    "density_difference_range"=>maximum(deltas)-minimum(deltas),
    "boundary_mass"=>sum(weights[1:10])+sum(weights[end-9:end]))
open(joinpath(@__DIR__, "receipt.toml"), "w") do io
    TOML.print(io, receipt)
end
println("Saved golf summary and receipt; key=", key)
