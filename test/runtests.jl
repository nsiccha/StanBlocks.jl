# using Pkg
# Pkg.activate("test")
using StanBlocks, Test, BridgeStan, StanLogDensityProblems, JSON

include("slic.jl")
# import StanBlocks, PosteriorDB

# import Test, StanLogDensityProblems, LogDensityProblems

# pdb = PosteriorDB.database()
# for posterior_name in PosteriorDB.posterior_names(pdb)
#     post = PosteriorDB.posterior(pdb, posterior_name)
    
#     jlpdf = StanBlocks.julia_implementation(post)
#     mkpath("stan_compilation")
#     stan_problem = StanLogDensityProblems.StanProblem(post, "stan_compilation", force=true)
#     slpdf(x) = LogDensityProblems.logdensity(stan_problem, x)
    
#     m = 10
#     n = LogDensityProblems.dimension(stan_problem)
#     @info "Testing $posterior_name ($n) with $m random draws"
#     Test.@test n == StanBlocks.dimension(jlpdf) == LogDensityProblems.dimension(jlpdf)
#     continue
#     try
#         Test.@inferred jlpdf(randn(n))
#     catch e
#         @error "Return type can't be inferred"
#         continue
#     end
#     X = randn((n, m))
#     jlpdfs = try
#         mapreduce(jlpdf, vcat, eachcol(X)) 
#     catch e
#         @error "Throws $e"
#         continue
#     end
#     slpdfs =  mapreduce(slpdf, vcat, eachcol(X)) 
#     if isapprox(jlpdfs, slpdfs; rtol=1e-4)
#         @info "Logdensity may be more or less correct."
#     else
#         @warn "Logdensity may be wrong: $(hcat(jlpdfs, slpdfs))"
#     end
# end