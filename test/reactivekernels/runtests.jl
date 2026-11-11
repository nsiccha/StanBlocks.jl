ENV["STANBLOCKS_TEST_REACTIVEKERNELS"] = "true"
push!(ARGS, "--tag=reactivekernels")
include(joinpath(@__DIR__, "..", "runtests.jl"))
