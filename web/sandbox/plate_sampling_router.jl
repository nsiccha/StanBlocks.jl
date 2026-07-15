# Test-only producer for the shared plate-routing contract. The public plate
# surface is the Julia do-block (`rv ~ plate(...) do ... end`); this probe only
# injects the lowered outer declarations + symbolic loop so the compiler pass
# can be verified independently of the still-separate public plate emitter.
function _plate_sampling_router_probe end

function StanBlocks.stan.expand_inline_or_trace(
        call::StanBlocks.stan.CanonicalExpr{typeof(_plate_sampling_router_probe)};
        info)
    n, obs = call.args
    injected = quote
        plate_x::vector[$n]
        plate_y::vector[$n]
        plate_prior::vector[$n]
        plate_flat::real
        plate_rv::vector[$n]
        plate_data_copy::vector[$n]
        for plate_i in 1:$n
            plate_x[plate_i] ~ normal(0.0, 1.0)
            plate_y[plate_i] ~ normal(plate_x[plate_i] + plate_flat, 1.0)
            plate_prior[plate_i] ~ normal(0.0, 1.0)
            plate_rv[plate_i] = plate_x[plate_i] + plate_y[plate_i]
            plate_data_copy[plate_i] = $obs[plate_i]
            $obs[plate_i] ~ normal(plate_rv[plate_i], 1.0)
        end
    end
    StanBlocks.stan.forward!(StanBlocks.stan.canonical(injected); info)
end

plate_router_submodel = @slic (;obs=randn(4), n=4) begin
    _plate_sampling_router_probe(n, obs)
    return plate_rv
end

plate_router_model = @slic (;) begin
    routed ~ plate_router_submodel
end

StanBlocks.stan.compiles(plate_router_model) || error("plate router model failed BridgeStan compilation")
plate_router_model
