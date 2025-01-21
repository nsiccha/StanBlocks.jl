module MooncakeExt
using StanBlocks, Mooncake

StanBlocks.with_gradient(p::StanBlocks.VectorPosterior) = begin 
    f = p.f
    rule = Mooncake.build_rrule(Tuple{typeof(f), Vector{Float64}})
    g(x::Vector{Float64}) = begin 
        rv, (_, g_) = Mooncake.value_and_gradient!!(rule, f, x)
        rv, g_
    end
    StanBlocks.VectorPosterior(f, g, p.gq, p.n)
end
end