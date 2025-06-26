module JSONExt
import JSON, StanBlocks
prepare_for_stan(x::Dict) = Dict([
    key=>prepare_for_stan(value)
    for (key, value) in x
])
prepare_for_stan(x::Number) = x
prepare_for_stan(x::AbstractVector{<:Number}) = x
prepare_for_stan(x::AbstractMatrix{<:Number}) = x'
StanBlocks.stan.bridgestan_data(x::Dict) = JSON.json(prepare_for_stan(x))
end