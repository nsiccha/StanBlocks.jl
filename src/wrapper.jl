struct VectorPosterior{F,G,GQ,N}
    f::F
    g::G
    gq::GQ
    n::N
end
@inline (p::VectorPosterior)(x) = p.f(x)
@inline dimension(p::VectorPosterior) = p.n
@inline generate_quantities(p::VectorPosterior, x) = p.gq(x)

function with_gradient end