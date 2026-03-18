module JSONExt
import JSON, StanBlocks
StanBlocks.stan.bridgestan_data(x::Dict) = JSON.json(StanBlocks.stan.prepare_for_stan(x))
end
