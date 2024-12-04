module StanBlocks

using LinearAlgebra, Statistics, Distributions, LogExpFunctions

include("wrapper.jl")
include("macros.jl")
include("functions.jl")

julia_implementation(key; kwargs...) = missing

end # module StanBlocks
