ENV["STANBLOCKS_TEST_SYMBOLICS"] = "true"
push!(ARGS, "--tag=symbolics")
include(joinpath(@__DIR__, "..", "runtests.jl"))
