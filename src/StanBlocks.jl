module StanBlocks

export @stan, @model, @parameters, @transformed_parameters, @generated_quantities, @bsum, with_gradient

using LinearAlgebra, Statistics, Distributions, LogExpFunctions

include("wrapper.jl")
include("macros.jl")
include("functions.jl")

julia_implementation(key; kwargs...) = missing
stan_implementation(key; kwargs...) = missing
include("check.jl")

end # module StanBlocks
