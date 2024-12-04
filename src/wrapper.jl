struct VectorPosterior{F,N}
    f::F
    n::N
end
@inline (p::VectorPosterior)(x) = p.f(x)
@inline dimension(p::VectorPosterior) = p.n