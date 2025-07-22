module StanBlocks

export @stan, @model, @parameters, @transformed_parameters, @generated_quantities, @bsum, with_gradient
export @slic, @defsig, @deffun
export stan_code, stan_model, stan_instantiate

using LinearAlgebra, Statistics, Distributions, LogExpFunctions

include("wrapper.jl")
include("macros.jl")
include("functions.jl")
include("slic_stan/slic.jl")

julia_implementation(key; kwargs...) = missing
stan_implementation(key; kwargs...) = missing
include("check.jl")

end # module StanBlocks
