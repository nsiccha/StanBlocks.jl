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

# Stan's `log_sum_exp` is ReactiveKernels' tie-exact translation of the
# LogExpFunctions function of the same shape, value for value (snag
# `jax-source-has-n-a3964196`). Of two scalars, also elementwise when
# broadcast, it is `logaddexp`: a `-Inf` argument gives the other, two `+Inf`
# give `+Inf`, and a tie's derivative is 1/2 per argument. Of an array it is
# `logsumexp`: `-Inf` when empty, a non-finite maximum as is, and the padding
# of a ragged row excluded. The Julia twin spells those cases as early
# returns, which RK does not translate.
ReactiveKernels._jax_elementwise(::typeof(StanBlocks._julia_log_sum_exp)) =
    ReactiveKernels._jax_elementwise(ReactiveKernels.LogExpFunctions.logaddexp)
ReactiveKernels._jax_function_call(ctx::ReactiveKernels._JaxContext,
        ::typeof(StanBlocks._julia_log_sum_exp), args::Vector{ReactiveKernels._PyVal}) =
    ReactiveKernels._jax_function_call(ctx, ReactiveKernels.LogExpFunctions.logsumexp, args)

# The rest of Stan's math builtins whose Julia twins branch (todo `1q0bqgu`,
# after snag `jax-source-has-n-a3964196`). Each is Stan's rule over
# ReactiveKernels' translation of the SpecialFunctions function the twin calls,
# so the numerics stay RK's (`lgamma`, log|gamma(x)|, is JAX's `gammaln`
# itself); it applies to scalars and elementwise in broadcasts. A domain error
# of the twin is an RK runtime check.
const _PyVal = ReactiveKernels._PyVal
const _JaxContext = ReactiveKernels._JaxContext
const SF = StanBlocks.SpecialFunctions

const _HELPERS = Dict{String,String}(
    "_stanblocks_lgamma" => """
def _stanblocks_lgamma(x):
    # Stan's lgamma, log|gamma(x)|: +inf at the poles and at both infinities,
    # where the host twin's SciPy gammaln gives -inf at -inf.
    return jnp.where(x == -jnp.inf, jnp.inf, jsp.gammaln(x))
""",
    "_stanblocks_pole" => """
def _stanblocks_pole(x):
    # Stan Math's is_nonpositive_integer: zero, a negative integer or -inf.
    return (x <= 0) & (jnp.floor(x) == x)
""",
    "_stanblocks_lchoose_k" => """
def _stanblocks_lchoose_k(n, k):
    # Stan Math's binomial_coefficient_log takes k > n / 2 as n - k.
    return jnp.where((n > -1) & (k > n / 2 + 1e-8), n - k, k)
""",
    "_stanblocks_lchoose" => """
def _stanblocks_lchoose(n, k, log_beta):
    # Stan Math's binomial_coefficient_log of n and the reflected k: NaN for a
    # NaN argument, 0 at k = 0, three lgammas below n + 1 = 10, and from there
    # -log_beta - log1p(n), log_beta being lbeta(n + 1 - k, k + 1). Each branch
    # is evaluated on arguments valid for it.
    small = n + 1 < 10
    ns, ks = jnp.where(small, n, 0.0), jnp.where(small, k, 0.0)
    gammas = jsp.gammaln(ns + 1) - jsp.gammaln(ks + 1) - jsp.gammaln(ns + 1 - ks)
    value = jnp.where(small, gammas, -log_beta - jnp.log1p(jnp.where(small, 9.0, n)))
    value = jnp.where(k == 0, 0.0, value)
    return jnp.where(jnp.isnan(n) | jnp.isnan(k), jnp.nan, value)
""",
)
_helper!(ctx::_JaxContext, name) = ReactiveKernels._jax_helper!(ctx, name, _HELPERS[name])
_bind!(ctx::_JaxContext, hint, code, T) = ReactiveKernels._jax_bind!(ctx, hint, _PyVal(code, T))
_result(f, args, code) =
    _PyVal(code, ReactiveKernels._jax_return_type(f, Any[a.T isa Type ? a.T : Any for a in args]))
_check!(ctx::_JaxContext, valid, message, E = DomainError) =
    ReactiveKernels._jax_call(ctx, ReactiveKernels._runtime_check,
                              [_PyVal(valid, Bool), _PyVal(repr(message), E)])

ReactiveKernels._jax_elementwise(::typeof(StanBlocks._julia_lgamma)) =
    "_stanblocks_lgamma" => _HELPERS["_stanblocks_lgamma"]

# At a pole `digamma` is NaN and `trigamma` +Inf; RK's function is evaluated
# at 1/2 there, so no pole's infinite derivative reaches the gradient: it is
# zero there, the derivative of the twin's constant.
for (twin, f, value) in ((StanBlocks._julia_digamma, SF.digamma, "jnp.nan"),
                         (StanBlocks._julia_trigamma, SF.trigamma, "jnp.inf"))
    @eval function ReactiveKernels._jax_elementwise_call(ctx::_JaxContext, ::typeof($twin),
                                                        args::Vector{_PyVal})
        x = ReactiveKernels._jax_bind!(ctx, :x, only(args))
        pole = _bind!(ctx, :pole, _helper!(ctx, "_stanblocks_pole") * "(" * x.code * ")", Bool)
        inner = ReactiveKernels._jax_call(ctx, $f,
            [_PyVal("jnp.where(" * pole.code * ", 0.5, " * x.code * ")", x.T)])
        _result($twin, args, "jnp.where(" * pole.code * ", " * $value * ", " * inner.code * ")")
    end
end

function ReactiveKernels._jax_elementwise_call(ctx::_JaxContext, ::typeof(StanBlocks._julia_lchoose),
                                              args::Vector{_PyVal})
    n, k = (_bind!(ctx, name, "1.0 * (" * a.code * ")", Float64) for (name, a) in zip((:n, :k), args))
    reflected = _bind!(ctx, :k, _helper!(ctx, "_stanblocks_lchoose_k") * "(" * n.code * ", " *
                                k.code * ")", Float64)
    # Checked after the reflection, as the twin's recursive call checks; a NaN
    # argument returns before any check.
    nan = "jnp.isnan(" * n.code * ") | jnp.isnan(" * k.code * ") | "
    _check!(ctx, nan * "(" * n.code * " >= -1)", "lchoose(n, k) requires n >= -1")
    _check!(ctx, nan * "(" * reflected.code * " >= -1)", "lchoose(n, k) requires k >= -1")
    _check!(ctx, nan * "(" * n.code * " + 1 - " * reflected.code * " >= 0)",
            "lchoose(n, k) requires n + 1 - k >= 0")
    large = "(" * n.code * " + 1 >= 10)"
    log_beta = ReactiveKernels._jax_call(ctx, SF.logbeta, [
        _PyVal("jnp.where(" * large * ", " * n.code * " + 1 - " * reflected.code * ", 1.0)", Float64),
        _PyVal("jnp.where(" * large * ", " * reflected.code * " + 1, 1.0)", Float64)])
    _result(StanBlocks._julia_lchoose, args, _helper!(ctx, "_stanblocks_lchoose") * "(" *
            n.code * ", " * reflected.code * ", " * log_beta.code * ")")
end

# Stan's `modified_bessel_first_kind` is RK's `besseli` (a literal order 0 or 1
# is `jax.scipy.special`'s own function), NaN at either infinity for orders 0
# and 1. Where the twin raises, the module does: a NaN argument, and AMOS's
# errors, at either infinity for any other order and where the value overflows.
function ReactiveKernels._jax_elementwise_call(ctx::_JaxContext,
        ::typeof(StanBlocks._julia_modified_bessel_first_kind), args::Vector{_PyVal})
    order, x = args
    x = ReactiveKernels._jax_bind!(ctx, :x, x)
    _check!(ctx, "jnp.logical_not(jnp.isnan(" * x.code * "))",
            "modified_bessel_first_kind: x must not be NaN")
    low = order.code in ("0", "1") ? "True" :
        "((" * order.code * ") == 0) | ((" * order.code * ") == 1)"
    value = ReactiveKernels._jax_bind!(ctx, :value, ReactiveKernels._jax_call(ctx, SF.besseli, [order, x]))
    _check!(ctx, "jnp.isfinite(" * x.code * ") | " * low,
            "modified_bessel_first_kind: input argument magnitude too large", ErrorException)
    _check!(ctx, "jnp.isinf(" * x.code * ") | jnp.isfinite(" * value.code * ")",
            "modified_bessel_first_kind: overflow", ErrorException)
    _result(StanBlocks._julia_modified_bessel_first_kind, args,
            "jnp.where(jnp.isinf(" * x.code * "), jnp.nan, " * value.code * ")")
end

# JAX has no Bessel function J, Y or K of integer order (its `bessel_jn` loses
# accuracy from moderate arguments), and RK translates none: an upstream
# boundary (stanblocks-use §5), named here rather than as a branch RK cannot
# translate.
for (twin, name) in ((StanBlocks._julia_bessel_first_kind, "bessel_first_kind"),
                     (StanBlocks._julia_bessel_second_kind, "bessel_second_kind"),
                     (StanBlocks._julia_modified_bessel_second_kind, "modified_bessel_second_kind"))
    @eval ReactiveKernels._jax_elementwise_call(ctx::_JaxContext, ::typeof($twin), args::Vector{_PyVal}) =
        ReactiveKernels._jax_error("Stan's `", $name, "` has no JAX translation: JAX has no ",
            "accurate Bessel function of this kind and ReactiveKernels translates none ",
            "(an upstream boundary)", ReactiveKernels._jax_where(ctx))
end

# Buffer element promotion is native Julia type work, not numerical work for
# JAX. Expose the equivalent concrete constructor; RK owns allocation shapes,
# indexed updates and host-array ownership in the translated body.
function allocation_source(f, argument_types)
    result_type = Base.promote_op(f, argument_types.parameters...)
    isconcretetype(result_type) && result_type <: Array || return nothing
    element = eltype(result_type)
    isconcretetype(element) || return nothing
    names = (:kind, :dims,
        (Symbol(:argument_, i) for i in 3:length(argument_types.parameters))...)
    TraceableSource(names, :(Array{$element}(undef, dims)), StanBlocks)
end

function deffun_source(f, argument_types::Type{<:Tuple})
    StanBlocks._with_slic_read() do
        hasmethod(f, argument_types) || return nothing
        if f === StanBlocks._deffun_julia_alloc || f === StanBlocks._deffun_julia_alloc_type
            return allocation_source(f, argument_types)
        end
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
