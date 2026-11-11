module StanBlocksReactiveKernelsExt

using StanBlocks
using ReactiveKernels: TraceableSource, register_source_provider!
import ReactiveKernels

# The Julia implementation uses a differentiable Taylor series; JAX has its
# own matrix exponential with its own derivative. This preserves the operation
# without translating the Julia implementation's data-dependent while loop.
function ReactiveKernels._jax_function_call(ctx::ReactiveKernels._JaxContext,
        ::typeof(StanBlocks._deffun_julia_matrix_exp), args::Vector{ReactiveKernels._PyVal})
    argument = only(args)
    isempty(argument.extent) && iszero(argument.padded) || throw(ArgumentError(
        "StanBlocks matrix_exp JAX translation requires an unpadded dense matrix; " *
        "padded ragged matrices are not supported."))
    ReactiveKernels._jax_result(StanBlocks._deffun_julia_matrix_exp, args,
        "jax.scipy.linalg.expm(" * argument.code * ")")
end

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
