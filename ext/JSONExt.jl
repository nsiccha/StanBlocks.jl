module JSONExt
import JSON, StanBlocks
StanBlocks.stan.bridgestan_data(x::Dict) = JSON.json(x)
end