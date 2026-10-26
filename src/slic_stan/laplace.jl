# ── Stan's embedded Laplace approximation (`laplace_marginal`) ──
#
#     z::vector[n] ~ laplace_marginal(normal(mu, sigma); <options>) do z
#         y ~ poisson_log(z)            # the latent's likelihood
#     end
#
# integrates the Gaussian latent `z` out of the model block with Stan's
# embedded Laplace approximation (Stan >= 2.37; snag `laplace-marginal-b2fb4189`,
# design decision `1wf36ma`). The statement lowers to
#   - a likelihood function `real <ll>(vector θ, <captures…>)`: the do block with
#     the latent read as `mu + θ`, and each observation of bound data replaced by
#     the log density its own `~` selects (`_laplace_density`);
#   - a covariance function `matrix <K>(<captures…>)` built from the prior;
#   - `target += laplace_marginal(<ll>, (<captures>), <block size>, <K>, (<captures>));`
#     in the model block. The compiler writes this `target +=`; a model never does;
#   - `z = mu + laplace_latent_rng(…)` in generated quantities, and the do block
#     replayed after it, reading that draw. Each replayed observation keeps its
#     `<obs>_gen` / `<obs>_likelihood` twins and leaves the model block
#     (`_laplace_observation`), because its density is inside the approximation.
# Stan requires the prior to have mean zero: θ = z - mu has the prior N(0, K),
# so the mean is added in the likelihood function and to the draw.
#
# A do block with no bound observation (prior simulation, or every observation
# cv-held-out) informs nothing: the latent is then drawn from its prior, and the
# block is replayed as ordinary statements (`_forward_laplace_unconditioned!`).

function _laplace_target end
function _laplace_observation end
function _laplace_density end

const LaplaceMarginal = Union{typeof(builtin.laplace_marginal), typeof(builtin.laplace_marginal_tol)}
const LaplaceLatent = Union{typeof(builtin.laplace_latent_rng), typeof(builtin.laplace_latent_tol_rng)}
const LaplaceFunction = Union{LaplaceMarginal, LaplaceLatent}
_laplace_variant(::Val{false}) = (builtin.laplace_marginal, builtin.laplace_latent_rng)
_laplace_variant(::Val{true}) = (builtin.laplace_marginal_tol, builtin.laplace_latent_tol_rng)

const _LAPLACE_PRIORS = (:std_normal, :normal, :multi_normal, :multi_normal_cholesky, :multi_normal_prec)
# Stan's `_tol` controls, in Stan's tuple order. Any one of them selects the
# `_tol` variants; the others take Stan Math 5.3.0's defaults
# (`stan/math/prim/fun/generate_laplace_options.hpp`). `theta_init` is the
# Newton start for the latent itself, default the prior mean.
const _LAPLACE_TOL_OPTIONS = (:theta_init, :tol, :max_num_steps, :solver, :max_steps_line_search, :allow_fallthrough)
const _LAPLACE_TOL_DEFAULTS = (; tol=1.49012e-8, max_num_steps=500, solver=1, max_steps_line_search=1000, allow_fallthrough=1)
const _LAPLACE_OPTIONS = (:hessian_block_size, _LAPLACE_TOL_OPTIONS...)
# Names the generated likelihood and covariance functions bind for themselves.
const _LAPLACE_RESERVED = (:laplace_theta, :laplace_mean, :laplace_sigma, :laplace_n,
    :laplace_Sigma, :laplace_L, :laplace_Omega)

forward!(x::SamplingExpr{Symbol,<:CanonicalExprV{:laplace_marginal}}; info) =
    _forward_laplace!(x.args[1], nothing, x.args[2]; info)
forward!(x::SamplingExpr{<:DeclExpr,<:CanonicalExprV{:laplace_marginal}}; info) = begin
    decl = x.args[1]
    decl.args[1] isa Symbol || error(
        "laplace_marginal: the latent must be a name, got `", decl.args[1], "`.")
    _forward_laplace!(decl.args[1], decl.args[2], x.args[2]; info)
end

_forward_laplace!(name::Symbol, decl_type, rhs; info) = begin
    # In a plate cell the latent is promoted cell storage, bound in advance;
    # its draw fills that storage like any cell `=` (`_forward_plate_assignment!`).
    name in keys(info) && _plate_promoted_target(name; info) === nothing && error(
        "laplace_marginal: `", name, "` is already bound; the integrated latent needs a fresh name.")
    spec = _laplace_parse(name, rhs; info)
    pargs = Any[forward!(a; info) for a in spec.prior.args]
    n = _laplace_size(name, decl_type, spec.prior, pargs; info)
    prior = _laplace_prior(name, spec.prior, pargs, n)
    stmts = _laplace_classify(name, spec.arg, spec.body; info)
    any(s -> s.kind === :observed, stmts) ||
        return _forward_laplace_unconditioned!(name, spec, n, pargs, stmts; info)
    ll = _laplace_likelihood(name, spec.arg, stmts, prior.mean; info)
    K = _laplace_closure((), prior.cov, prior.captures; info)
    opts = _laplace_options(name, spec.options, n, prior.mean; info)
    marginal, latent = _laplace_variant(Val(opts.tol))
    args = (ll, opts.hessian_block_size, K, n, opts.controls...)
    target = CanonicalExpr(_laplace_target, _trace_stan_call(marginal, args...; info))
    draw = _trace_stan_call(latent, args...; info)
    value = prior.mean === nothing ? draw : _trace_stan_call(+, prior.mean, draw; info)
    out = Any[target, forward!(CanonicalExpr(:(=), name, value); info)]
    _note_laplace_latent!(name; info)
    observed = Set(s.base for s in stmts if s.kind === :observed)
    replay = forward!(canonical(_laplace_replay_block(name, spec.arg, stmts)); info)
    append!(out, _laplace_mark_observations(replay, observed).args)
    CanonicalExpr(:block, out...)
end

# A raw prior call in a message: its family, not the traced call's mangled name.
_laplace_label(prior::CanonicalExpr) = string(head(prior), "(…)")
_laplace_label(prior) = string(prior)

_laplace_parse(name, rhs; info) = begin
    args = rhs.args
    (!isempty(args) && args[1] isa CanonicalExprV{:->}) || error(
        "laplace_marginal: `", name, " ~ laplace_marginal(prior) do ", name, " … end` needs a ",
        "`do` block holding the latent's likelihood.")
    length(args) == 2 || error(
        "laplace_marginal: `", name, " ~ laplace_marginal(…)` takes one positional argument, the ",
        "latent's Gaussian prior (", join(map(p -> "`$p`", _LAPLACE_PRIORS), ", "), "); got ",
        length(args) - 1, ".")
    lambda, prior = args
    params, vararg = _parse_lambda_lhs(lambda.args[1])
    (vararg === nothing && length(params) == 1) || error(
        "laplace_marginal: the do block takes one argument, the latent (`do ", name, "`).")
    arg = only(params)
    arg in _LAPLACE_RESERVED && error(
        "laplace_marginal: the do block's argument `", arg, "` is a name the generated ",
        "likelihood function binds itself; call it `", name, "`.")
    arg !== name && arg in keys(info) && error(
        "laplace_marginal: the do block's argument `", arg, "` shadows a model name; call it `", name, "`.")
    body = lambda.args[2]
    Meta.isexpr(body, :block) || (body = Expr(:block, body))
    (prior isa CanonicalExpr && head(prior) isa Symbol && head(prior) in _LAPLACE_PRIORS) || error(
        "laplace_marginal: the prior of `", name, "` must be one of ",
        join(map(p -> "`$p(…)`", _LAPLACE_PRIORS), ", "), "; got `", _laplace_label(prior), "`.")
    unknown = setdiff(keys(rhs.kwargs), _LAPLACE_OPTIONS)
    isempty(unknown) || error(
        "laplace_marginal: unknown option", length(unknown) == 1 ? " `" : "s `", join(unknown, "`, `"),
        "`; the options are ", join(map(o -> "`$o`", _LAPLACE_OPTIONS), ", "), ".")
    used = Set{Symbol}()
    _collect_all_syms!(body, used)
    clash = intersect(used, _LAPLACE_RESERVED)
    isempty(clash) || error(
        "laplace_marginal: the do block of `", name, "` uses `", first(clash), "`, a name the ",
        "generated likelihood function binds itself. Rename it.")
    (; arg, body, prior, options=rhs.kwargs)
end

# The latent's length: its declared `vector[n]`, the prior's `n=`, or the length
# of the prior's first vector (or matrix) argument.
_laplace_size(name, decl_type, prior, pargs; info) = begin
    if decl_type !== nothing
        ct, sizes... = _is_getindex_expr(decl_type) ? decl_type.args : (decl_type,)
        (ct === :vector && length(sizes) == 1) || error(
            "laplace_marginal: the integrated latent is a `vector[n]`; got `", name, "::",
            pretty_type_expr(decl_type), "`.")
        return forward!(only(sizes); info)
    end
    haskey(prior.kwargs, :n) && return forward!(prior.kwargs[:n]; info)
    i = findfirst(a -> a isa StanExpr && stan_ndim(a) >= 1, pargs)
    i === nothing && error(
        "laplace_marginal: the length of `", name, "` does not follow from its prior `",
        _laplace_label(prior), "`; declare it, `", name, "::vector[n] ~ laplace_marginal(…)`, or ",
        "give the prior an `n=`.")
    stan_size(pargs[i])[1]
end

# The prior's mean (or `nothing` for zero) and its covariance, as the body and
# captures of the covariance function.
_laplace_prior(name, prior, pargs, n) = _laplace_prior(Val(head(prior)), name, prior, pargs, n)
_laplace_arity(name, prior, pargs, k) = length(pargs) == k || error(
    "laplace_marginal: `", head(prior), "` takes ", k, " argument", k == 1 ? "" : "s",
    " as the prior of `", name, "`; got ", length(pargs), ".")
_laplace_mean(mu) = expr(mu) isa Number && iszero(expr(mu)) ? nothing : mu
_laplace_prior(::Val{:std_normal}, name, prior, pargs, n) = begin
    _laplace_arity(name, prior, pargs, 0)
    (; mean=nothing, cov=:(identity_matrix(laplace_n)), captures=(:laplace_n => n,))
end
_laplace_prior(::Val{:normal}, name, prior, pargs, n) = begin
    _laplace_arity(name, prior, pargs, 2)
    mu, sigma = pargs
    all(a -> stan_ndim(a) <= 1, pargs) || error(
        "laplace_marginal: `normal(mu, sigma)` as the prior of `", name, "` takes scalars or vectors.")
    (; mean=_laplace_mean(mu), cov=:(diag_matrix(rep_vector(1.0, laplace_n) .* square(laplace_sigma))),
        captures=(:laplace_sigma => sigma, :laplace_n => n))
end
_laplace_prior(::Val{:multi_normal}, name, prior, pargs, n) = begin
    _laplace_arity(name, prior, pargs, 2)
    (; mean=_laplace_mean(pargs[1]), cov=:laplace_Sigma, captures=(:laplace_Sigma => pargs[2],))
end
_laplace_prior(::Val{:multi_normal_cholesky}, name, prior, pargs, n) = begin
    _laplace_arity(name, prior, pargs, 2)
    (; mean=_laplace_mean(pargs[1]), cov=:(multiply_lower_tri_self_transpose(laplace_L)),
        captures=(:laplace_L => pargs[2],))
end
_laplace_prior(::Val{:multi_normal_prec}, name, prior, pargs, n) = begin
    _laplace_arity(name, prior, pargs, 2)
    (; mean=_laplace_mean(pargs[1]), cov=:(inverse_spd(laplace_Omega)),
        captures=(:laplace_Omega => pargs[2],))
end

# The do block's statements. `observed`: a bound, not held-out data variable,
# whose density goes into the likelihood function; `held_out` (cv-marked) and
# `simulated` (unbound) observations only replay in generated quantities.
_laplace_lhs_base(x::Symbol) = x
_laplace_lhs_base(x::Expr) = x.head in (:ref, :., :(::)) ? _laplace_lhs_base(x.args[1]) : nothing
_laplace_lhs_base(_) = nothing
_laplace_classify(name, arg, body; info) = begin
    stmts = NamedTuple[]
    for s in body.args
        if s isa LineNumberNode
            push!(stmts, (; kind=:lnn, stmt=s, base=nothing))
        elseif Meta.isexpr(s, :call) && length(s.args) == 3 && s.args[1] === :~
            base = _laplace_lhs_base(s.args[2])
            base === nothing && error(
                "laplace_marginal: the do block of `", name, "` observes `", s.args[2], "`, which ",
                "names no data variable.")
            base === arg && error(
                "laplace_marginal: the do block samples the latent `", arg, "` itself; its prior ",
                "is the argument of `laplace_marginal`.")
            kind = if base in keys(info)
                qual(info[base]) == :data || error(
                    "laplace_marginal: the do block of `", name, "` samples `", s.args[2], "`, which is ",
                    "not data. The block is the latent's likelihood: it observes data and binds ",
                    "values with `=`; declare parameters before the `laplace_marginal` statement.")
                cv(info[base]) ? :held_out : :observed
            else
                :simulated
            end
            push!(stmts, (; kind, stmt=s, base))
        elseif Meta.isexpr(s, :(=)) && _laplace_lhs_base(s.args[1]) isa Symbol &&
                !Meta.isexpr(s.args[1], :ref)
            push!(stmts, (; kind=:assign, stmt=s, base=_laplace_lhs_base(s.args[1])))
        else
            error(
                "laplace_marginal: the do block of `", name, "` holds the latent's likelihood, ",
                "written as `=` bindings and observations `y ~ family(…)`; got `", s, "`.")
        end
    end
    stmts
end

# The likelihood function: the statements the bound observations need, the
# latent read as `mean + θ`, and the sum of the observations' densities.
_laplace_syms(x) = (s = Set{Symbol}(); _collect_all_syms!(x, s); s)
_laplace_likelihood(name, arg, stmts, mean; info) = begin
    needed = Set{Symbol}()
    kept = Any[]
    for s in Iterators.reverse(stmts)
        if s.kind === :observed
            union!(needed, _laplace_syms(s.stmt.args[2]), _laplace_syms(s.stmt.args[3]))
            pushfirst!(kept, s)
        elseif s.kind === :assign && s.base in needed
            union!(needed, _laplace_syms(s.stmt.args[2]))
            pushfirst!(kept, s)
        elseif s.kind === :simulated && s.base in needed
            error(
                "laplace_marginal: `", s.base, "` is sampled inside the do block of `", name, "` ",
                "without data, but the latent's likelihood reads it. A quantity the likelihood ",
                "depends on is either part of the integrated latent or a parameter declared before ",
                "the `laplace_marginal` statement.")
        elseif s.kind === :lnn
            pushfirst!(kept, s)
        end
    end
    body = Any[mean === nothing ? :($arg = laplace_theta) : :($arg = laplace_mean + laplace_theta)]
    terms = Symbol[]
    for s in kept
        if s.kind === :observed
            lp = Symbol(:laplace_lp, length(terms) + 1)
            push!(terms, lp)
            push!(body, Expr(:(=), lp, Expr(:call, _laplace_density, s.stmt.args[2], s.stmt.args[3])))
        else
            push!(body, s.stmt)
        end
    end
    push!(body, length(terms) == 1 ? only(terms) : Expr(:call, :+, terms...))
    block = Expr(:block, body...)
    captures = _closure_scope_captures(block, Set([:laplace_theta]); info)
    mean === nothing || (captures[:laplace_mean] = mean)
    _laplace_closure((:laplace_theta,), block, captures; info)
end

# A closure record built here rather than traced from a `->`: its captures are
# the values the generated function reads, under the names its body uses. Stan
# has no empty tuple, so a function that reads nothing takes an unused int.
_laplace_closure(arg_names, body, captures; info) = begin
    captures = OrderedDict{Symbol,Any}(captures...)
    isempty(stan_call_args(collect(values(captures)))) && (captures[:laplace_unused] = stan_expr(0))
    _closure_stan_expr((
        arg_names = Tuple(arg_names),
        vararg_name = nothing,
        body = Meta.isexpr(body, :block) ? body : Expr(:block, body),
        captures = captures,
        kwargs = (),
        source = something(_get_lnn(info), LineNumberNode(0, :none)),
        id = _next_closure_id(info),
        mod = get_module(info),
    ))
end

_laplace_options(name, options, n, mean; info) = begin
    hb = haskey(options, :hessian_block_size) ? forward!(options[:hessian_block_size]; info) : n
    (center_type(hb) <: types.int && stan_ndim(hb) == 0 && qual(hb) == :data) || error(
        "laplace_marginal: `hessian_block_size` of `", name, "` must be a data `int`; got `", hb, "`.")
    tol = any(in(keys(options)), _LAPLACE_TOL_OPTIONS)
    tol || return (; tol, hessian_block_size=hb, controls=())
    theta_init = if haskey(options, :theta_init)
        start = forward!(options[:theta_init]; info)
        mean === nothing ? start : _trace_stan_call(-, start, mean; info)
    else
        _trace_stan_call(builtin.rep_vector, stan_expr(0.0), n; info)
    end
    rest = map(k -> haskey(options, k) ? forward!(options[k]; info) : stan_expr(_LAPLACE_TOL_DEFAULTS[k]),
        _LAPLACE_TOL_OPTIONS[2:end])
    (; tol, hessian_block_size=hb, controls=(theta_init, rest...))
end

# The do block as it replays in generated quantities: the argument names the
# latent itself there.
_laplace_rename(x, from, to) = x
_laplace_rename(x::Symbol, from, to) = x === from ? to : x
_laplace_rename(x::Expr, from, to) =
    x.head === :kw ? Expr(:kw, x.args[1], _laplace_rename(x.args[2], from, to)) :
    x.head === :call ? Expr(:call, x.args[1], map(a -> _laplace_rename(a, from, to), x.args[2:end])...) :
    x.head === :. ? Expr(:., _laplace_rename(x.args[1], from, to), x.args[2:end]...) :
    Expr(x.head, map(a -> _laplace_rename(a, from, to), x.args)...)
_laplace_replay_block(name, arg, stmts) =
    Expr(:block, (_laplace_rename(s.stmt, arg, name) for s in stmts)...)
# A raw `lhs ~ laplace_marginal(…) do arg … end` statement with `f` applied to
# each statement of its do block; any other statement unchanged. Lets a pass
# over a plate body's raw statements reach the observations inside the block.
_laplace_map_body(f, s) = s
_laplace_map_body(f, s::Expr) = begin
    (s.head === :call && length(s.args) == 3 && s.args[1] === :~ && Meta.isexpr(s.args[3], :do)) || return s
    call, lambda = s.args[3].args
    (Meta.isexpr(call, :call) && call.args[1] === :laplace_marginal && Meta.isexpr(lambda, :->)) || return s
    lhs, body = lambda.args
    body = Meta.isexpr(body, :block) ? Expr(:block, map(f, body.args)...) : f(body)
    Expr(:call, :~, s.args[2], Expr(:do, call, Expr(:->, lhs, body)))
end
# Each replayed observation of the likelihood leaves the model block: its density
# is inside the approximation. It keeps its generated-quantities twins.
_laplace_mark_observations(x, observed) = x
_laplace_mark_observations(x::BlockExpr, observed) =
    remake(x, map(a -> _laplace_mark_observations(a, observed), x.args)...)
_laplace_mark_observations(x::SamplingExpr, observed) =
    qual(x.args[1]) == :data && _sampling_lhs_key(x.args[1]) in observed ?
        CanonicalExpr(_laplace_observation, x) : x

# Nothing informs the latent: draw it from its prior and replay the block as
# ordinary statements (its observations held out or simulated).
_forward_laplace_unconditioned!(name, spec, n, pargs, stmts; info) = begin
    prior = forward!(remake(spec.prior, spec.prior.args...); info)
    token = StanExpr(types.vector,
        StanType(types.tokenof{types.vector}, (n,); value=types.vector, qual=:data))
    draw = _trace_stan_expr(expr(rng_expr(token, prior)), info)
    out = Any[forward!(CanonicalExpr(:(=), name, draw); info)]
    _note_laplace_latent!(name; info)
    append!(out, forward!(canonical(_laplace_replay_block(name, spec.arg, stmts)); info).args)
    CanonicalExpr(:block, out...)
end

# `y ~ family(…)` inside the likelihood function: the log density the `~`
# itself selects (an observed bound truncates, as for an observation). The
# replayed `~` (`_laplace_replay_block`) validates the observation against the
# data itself; here the data is a function argument.
expand_inline_or_trace(x::CanonicalExpr{typeof(_laplace_density)}; info) = begin
    lhs, rhs = x.args
    rhs = _normalised_truncation(rhs; observed=true)
    lpmf = _dual_lpmf_call(lhs, rhs)
    density = isnothing(lpmf) ? lpxf_expr(lhs, rhs; validate=false) : lpmf
    _check_lpxf_resolves(density)
    density
end

tracetype(::CanonicalExpr{<:LaplaceMarginal}) = StanType(types.real)
tracetype(x::CanonicalExpr{<:LaplaceLatent}) = StanType(types.vector, (x.args[4],))

# The call Stan sees: (likelihood, (its captures), block size, covariance,
# (its captures)[, controls]). The latent's length (argument 4) only types the
# draw and the likelihood's latent argument.
_laplace_theta(n) = StanExpr(:_, StanType(types.vector, (n,)))
_laplace_ll_call(x) = CanonicalExpr(x.args[1], _laplace_theta(x.args[4]), _closure_captures(x.args[1])...)
_laplace_cov_call(x) = CanonicalExpr(x.args[3], _closure_captures(x.args[3])...)
struct _LaplaceTuple
    items::Vector{Any}
    _LaplaceTuple(items) = new(collect(Any, items))
end
Base.show(io::IO, t::_LaplaceTuple) =
    autoprint(io, "(", Join(t.items, ", "), length(t.items) == 1 ? ",)" : ")")
_laplace_captures_tuple(f) = _LaplaceTuple(stan_call_args(collect(_closure_captures(f))))
Base.show(io::IO, x::CanonicalExpr{<:LaplaceFunction}) = begin
    ll, hb, K = x.args[1], x.args[2], x.args[3]
    parts = Any[
        func_name(ll, _laplace_ll_call(x).args), _laplace_captures_tuple(ll), hb,
        func_name(K, _laplace_cov_call(x).args), _laplace_captures_tuple(K),
    ]
    length(x.args) > 4 && push!(parts, _LaplaceTuple(x.args[5:end]))
    autoprint(io, head(x), "(", Join(parts, ", "), ")")
end
fetch_functions!(x::CanonicalExpr{<:LaplaceFunction}; info) = begin
    fetch_functions!(_laplace_ll_call(x); info)
    fetch_functions!(_laplace_cov_call(x); info)
end

# The latents integrated out so far, by name (`_reject_draw_in_likelihood`).
_note_laplace_latent!(name; info) = begin
    context = _trace_context(info)
    context === nothing || push!(context.laplace_latents, name)
    nothing
end
_laplace_latent_reads(x; info) = begin
    context = _trace_context(info)
    (context === nothing || isempty(context.laplace_latents)) && return Symbol[]
    found = Symbol[]
    _laplace_latent_reads!(found, x, context.laplace_latents)
    unique!(found)
end
_laplace_latent_reads!(found, x, latents) = nothing
_laplace_latent_reads!(found, x::StanExpr, latents) = _laplace_latent_reads!(found, expr(x), latents)
_laplace_latent_reads!(found, x::Symbol, latents) = (x in latents && push!(found, x); nothing)
_laplace_latent_reads!(found, x::CanonicalExpr, latents) =
    foreach(a -> _laplace_latent_reads!(found, a, latents), x.args)

# The model-block term.
distribution_blocks(::CanonicalExpr{typeof(_laplace_target)}; info) = (:model,)
Base.show(io::IO, x::CanonicalExpr{typeof(_laplace_target)}) = print(io, "target += ", only(x.args))
# Whether a block holds a Laplace term, at its top level or in a (plate) loop.
_has_laplace_term(x) = false
_has_laplace_term(x::CanonicalExpr{typeof(_laplace_target)}) = true
_has_laplace_term(x::AbstractVector) = any(_has_laplace_term, x)
_has_laplace_term(x::StanBlock) = _has_laplace_term(content(x))
_has_laplace_term(x::BlockExpr) = _has_laplace_term(x.args)
_has_laplace_term(x::DocumentExpr) = _has_laplace_term(x.args[2])
_has_laplace_term(x::StanExpr{<:ForExpr}) = _has_laplace_term(expr(x).args[2])

# A replayed observation: generated quantities only, at model scope and in a
# compiler-owned (plate) loop alike.
backward!(x::CanonicalExpr{typeof(_laplace_observation)}; info) =
    remake(x, backward!(only(x.args); info))
distribution_blocks(::CanonicalExpr{typeof(_laplace_observation)}; info) = (:generated_quantities,)
_loop_distribution_stmts(x::CanonicalExpr{typeof(_laplace_observation)}, v::Val{:generated_quantities}; info) =
    _loop_distribution_stmts(only(x.args), v; info)
_obs_twin_targets(x::CanonicalExpr{typeof(_laplace_observation)}; info) = _obs_twin_targets(only(x.args); info)
distribute!(x::CanonicalExpr{typeof(_laplace_observation)}; info) = begin
    stmt = only(x.args)
    _push_expr!(info, stmt)
    push!(block(info, :generated_quantities), _ragged_scoped(:generated_quantities, stmt; info); info)
    _pop_expr!(info)
end
