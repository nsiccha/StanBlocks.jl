module DifferentiationInterfaceExt
using StanBlocks, DifferentiationInterface

StanBlocks.with_gradient(p::StanBlocks.VectorPosterior, backend::DifferentiationInterface.AbstractADType; T=Float64) = begin 
    f = p.f
    prep = DifferentiationInterface.prepare_gradient(f, backend, zeros(T, p.n))
    g(x::Vector) = DifferentiationInterface.value_and_gradient(f, prep, backend, x)
    StanBlocks.VectorPosterior(f, g, p.gq, p.n)
end
end