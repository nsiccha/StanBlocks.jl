module StanBlocksReactiveKernelsExt

using StanBlocks
using ReactiveKernels: TraceableSource, register_source_provider!
import ReactiveKernels

# The Julia implementation uses a differentiable Taylor series; JAX has its
# own matrix exponential with its own derivative. This preserves the operation
# without translating the Julia implementation's data-dependent while loop.
ReactiveKernels._jax_elementwise(::typeof(StanBlocks._deffun_julia_matrix_exp)) =
    "_stanblocks_matrix_exp" => """
    def _stanblocks_matrix_exp(x):
        import jax.scipy.linalg
        return jax.scipy.linalg.expm(x)
    """

function deffun_source(f, argument_types::Type{<:Tuple})
    StanBlocks._with_slic_read() do
        hasmethod(f, argument_types) || return nothing
        method = which(f, argument_types)
        source = StanBlocks._deffun_julia_source(method.sig)
        isnothing(source) && return nothing
        original, names, body, mod = source
        original === method || return nothing
        TraceableSource(names, body, mod)
    end
end

function __init__()
    register_source_provider!(deffun_source)
end

end
