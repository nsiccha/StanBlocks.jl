# `maybe_index(x, idx)` is a trace-time rewriting hook inserted by the
# partly-missing-vector body pre-pass.  After `forward!` resolves the head
# symbol to the bare function `builtin.maybe_index` (via canonical's
# StanExpr2{<:types.func} rewrite), this dispatch intercepts before
# `stan_expr` would need a `tracetype` for `maybe_index`:
#   ndim 0 (scalar) → return `x` unchanged (drop `idx`)
#   ndim 1 (vector) → return `x[idx]`
#   otherwise       → loud error (matrix / array not supported)
expand_inline_or_trace(x::CanonicalExpr{typeof(builtin.maybe_index)}; info) = begin
    nd = stan_ndim(x.args[1])
    nd == 0 && return x.args[1]
    nd == 1 && return stan_call(getindex, x.args[1], x.args[2])
    error(
        "maybe_index: only scalar (rank 0) and vector (rank 1) distribution args are " *
        "supported for partly-missing-vector imputation; got rank-$(nd) for " *
        "$(expr(x.args[1])). Matrix or inherently-joint distribution args cannot be " *
        "element-wise indexed. Use an explicit obs/mis split model instead."
    )
end
# A traced call's `StanExpr` type names only its head (types.jl), so the
# `dims(X)[i]` shape is matched on the getindex's runtime arguments.
fold_shape_query(x::StanExpr{<:CanonicalExpr{typeof(getindex)}}) = _fold_dims_index(x, expr(x).args...)
_fold_dims_index(x, _args...) = x
_fold_dims_index(x, query::StanExpr{<:CanonicalExpr{typeof(builtin.dims)}}, index::StanExpr{<:Integer}) = begin
    arg = expr(query).args[1]::StanExpr
    i = expr(index)
    sz = stan_size(arg)
    if 1 <= i <= length(sz)
        candidate = sz[i]
        qual(candidate) == :data ? fold_shape_query(candidate) : x
    else
        x
    end
end
fetch_data!(x::StanType{<:types.tup}; info) = fetch_data!((stan_size(x), x.info.arg_types); info)

# A distribution HOF has one base-family token among its arguments and owns a
# continuous/discrete pair of aggregate + pointwise wrappers. Registration is
# deliberately generic: future combinators can opt in without adding new
# sampling, likelihood, or RNG lowering methods.
_distribution_hof(::Any) = nothing
_distribution_hof(::typeof(builtin.weighted)) = (
    name = :weighted,
    family_arg = 1,
    data_args = (2,),
    requires_data_lhs = true,
    # The family slot may also hold one of these bounded combinators, whose own
    # arguments are then this HOF's trailing ones; in the token form its bounds
    # are written on `weighted` and forwarded to it.
    nestable = (:truncated, :censored, :interval_censored),
    forwarded_kwargs = (:lower, :upper),
    lpdf = builtin.weighted_lpdf,
    lpmf = builtin.weighted_lpmf,
    lpdfs = builtin.weighted_lpdfs,
    lpmfs = builtin.weighted_lpmfs,
    rng = builtin.weighted_rng,
)
_distribution_hof(::typeof(builtin.truncated)) = (
    name = :truncated,
    family_arg = 1,
    data_args = (),
    requires_data_lhs = false,
    optional_kwargs = (:lower, :upper),
    variants = (
        lower = (
            head = builtin.lower_conditioning,
            lpdf = builtin.lower_conditioning_lpdf,
            lpmf = builtin.lower_conditioning_lpmf,
            lpdfs = builtin.lower_conditioning_lpdfs,
            lpmfs = builtin.lower_conditioning_lpmfs,
            rng = builtin.lower_conditioning_rng,
        ),
        upper = (
            head = builtin.upper_conditioning,
            lpdf = builtin.upper_conditioning_lpdf,
            lpmf = builtin.upper_conditioning_lpmf,
            lpdfs = builtin.upper_conditioning_lpdfs,
            lpmfs = builtin.upper_conditioning_lpmfs,
            rng = builtin.upper_conditioning_rng,
        ),
        both = (
            head = builtin.conditioning,
            lpdf = builtin.conditioning_lpdf,
            lpmf = builtin.conditioning_lpmf,
            lpdfs = builtin.conditioning_lpdfs,
            lpmfs = builtin.conditioning_lpmfs,
            rng = builtin.conditioning_rng,
        ),
    ),
)
_distribution_hof(::typeof(builtin.censored)) = (
    name = :censored,
    family_arg = 1,
    data_args = (),
    # A censored (clamped) law has threshold atoms and is therefore not a valid
    # continuously-parameterised HMC prior. It is an observation model.
    requires_data_lhs = true,
    optional_kwargs = (:lower, :upper),
    variants = (
        lower = (
            head = builtin.lower_clamping,
            lpdf = builtin.lower_clamping_lpdf,
            lpmf = builtin.lower_clamping_lpmf,
            lpdfs = builtin.lower_clamping_lpdfs,
            lpmfs = builtin.lower_clamping_lpmfs,
            rng = builtin.lower_clamping_rng,
        ),
        upper = (
            head = builtin.upper_clamping,
            lpdf = builtin.upper_clamping_lpdf,
            lpmf = builtin.upper_clamping_lpmf,
            lpdfs = builtin.upper_clamping_lpdfs,
            lpmfs = builtin.upper_clamping_lpmfs,
            rng = builtin.upper_clamping_rng,
        ),
        both = (
            head = builtin.clamping,
            lpdf = builtin.clamping_lpdf,
            lpmf = builtin.clamping_lpmf,
            lpdfs = builtin.clamping_lpdfs,
            lpmfs = builtin.clamping_lpmfs,
            rng = builtin.clamping_rng,
        ),
    ),
)
_distribution_hof(::typeof(builtin.interval_censored)) = (
    name = :interval_censored,
    family_arg = 1,
    data_args = (),
    requires_data_lhs = true,
    lpdf = builtin.interval_evidence_impl_lpdf,
    lpmf = builtin.interval_evidence_impl_lpmf,
    lpdfs = builtin.interval_evidence_impl_lpdfs,
    lpmfs = builtin.interval_evidence_impl_lpmfs,
    rng = builtin.interval_evidence_impl_rng,
)
_distribution_hof(::typeof(builtin.interval_evidence_impl)) =
    _distribution_hof(builtin.interval_censored)
_fixed_distribution_hof(name, variant; requires_data_lhs = false) = (
    name,
    family_arg = 1,
    data_args = (),
    requires_data_lhs,
    lpdf = variant.lpdf,
    lpmf = variant.lpmf,
    lpdfs = variant.lpdfs,
    lpmfs = variant.lpmfs,
    rng = variant.rng,
)
_distribution_hof(::typeof(builtin.lower_conditioning)) = _fixed_distribution_hof(
    :truncated, _distribution_hof(builtin.truncated).variants.lower,
)
_distribution_hof(::typeof(builtin.upper_conditioning)) = _fixed_distribution_hof(
    :truncated, _distribution_hof(builtin.truncated).variants.upper,
)
_distribution_hof(::typeof(builtin.conditioning)) = _fixed_distribution_hof(
    :truncated, _distribution_hof(builtin.truncated).variants.both,
)
_distribution_hof(::typeof(builtin.lower_clamping)) = _fixed_distribution_hof(
    :censored, _distribution_hof(builtin.censored).variants.lower;
    requires_data_lhs = true,
)
_distribution_hof(::typeof(builtin.upper_clamping)) = _fixed_distribution_hof(
    :censored, _distribution_hof(builtin.censored).variants.upper;
    requires_data_lhs = true,
)
_distribution_hof(::typeof(builtin.clamping)) = _fixed_distribution_hof(
    :censored, _distribution_hof(builtin.censored).variants.both;
    requires_data_lhs = true,
)

_family_function(x::StanExpr2{<:types.func}) = type(x).info.value
_family_function(x) = error(
    "distribution HOF: expected a base distribution function token, got $(type(x))"
)
# The HOF spec of a resolved call head, or `nothing` for any other head.
_hof_spec_of(x::StanExpr2{<:types.func}) = _distribution_hof(_family_function(x))
_hof_spec_of(x) = nothing
_is_compile_time_nothing(x) = x === nothing || x === :nothing ||
    (x isa QuoteNode && x.value === nothing)
_forward_call_kwargs(
    x::CanonicalExpr,
    resolved_head::StanExpr2{<:types.func};
    info,
) = begin
    spec = _distribution_hof(_family_function(resolved_head))
    optional = isnothing(spec) ? () :
        (get(spec, :optional_kwargs, ())..., get(spec, :forwarded_kwargs, ())...)
    isempty(optional) && return forward!(x.kwargs; info)
    # A distribution HOF owns these kwargs at trace time. Explicit `nothing`
    # is equivalent to omission and must disappear before generic symbol/type
    # resolution; present bounds remain on the canonical call so `autotype`
    # can also use them as parameter constraints.
    kept = (;[
        key => value
        for (key, value) in pairs(x.kwargs)
        if !(key in optional && _is_compile_time_nothing(value))
    ]...)
    forward!(kept; info)
end
# The position of a HOF call's BASE family. A nestable combinator in the family
# slot (`weighted(lower_clamping, w, normal, lo, mu, sigma)`) takes this HOF's
# trailing arguments as its own, so kind, carrier and RNG checks look through it.
_hof_trailing_arg(spec) = spec.family_arg + length(spec.data_args) + 1
_check_nestable(spec, inner) = begin
    nestable = get(spec, :nestable, ())
    inner.name in nestable || error(
        "$(spec.name): the family slot takes a base distribution family",
        isempty(nestable) ? "" : " or one of `$(join(nestable, "`, `"))`",
        "; `$(inner.name)` cannot be nested here",
    )
    nothing
end
# Bounds reach a nested combinator from exactly one place; a key given on both
# the HOF and its family must not silently resolve to either.
_disjoint_kwargs(spec, outer::NamedTuple, inner::NamedTuple) = begin
    duplicate = intersect(keys(outer), keys(inner))
    isempty(duplicate) || error(
        "$(spec.name): `", join(duplicate, "=`, `"), "=` is given both on `$(spec.name)` ",
        "and on the combinator in its family slot; write it once",
    )
    (; outer..., inner...)
end
_hof_base_family_index(spec, args) = begin
    i = spec.family_arg
    length(args) >= i || error("distribution HOF: missing family argument $i")
    inner = _distribution_hof(_family_function(args[i]))
    isnothing(inner) && return i
    _check_nestable(spec, inner)
    j = _hof_trailing_arg(spec)
    j - 1 + _hof_base_family_index(inner, args[j:end])
end
_distribution_hof_family(spec, rhs::CanonicalExpr) =
    _family_function(rhs.args[_hof_base_family_index(spec, rhs.args)])
_validate_distribution_hof(spec, lhs, rhs::CanonicalExpr; unbound=false) = begin
    if spec.requires_data_lhs && qual(lhs) != :data && !unbound
        error(
            "$(spec.name): observations must be data-qualified; $(spec.name) priors are not supported"
        )
    end
    for i in spec.data_args
        length(rhs.args) >= i || error("distribution HOF: missing data argument $i")
        qual(rhs.args[i]) == :data || error(
            "$(spec.name): likelihood control arguments must be data-qualified; " *
            "parameter-dependent likelihood weights are not supported"
        )
    end
    nothing
end

# Stan emits every parameter declaration before its transformed-parameters
# block. Consequently a parameter constraint may depend on data (including a
# transformed-data binding) or an earlier genuine parameter, but never on a
# parameter-dependent assignment. Qualifiers alone cannot express that
# distinction: both sampled parameters and transformed-parameter values are
# `:parameter`-qualified, so `forward!` records their declaration provenance in
# `decl_role` and this sampling chokepoint validates the resolved leaves.
_constraint_dependencies!(deps, x::StanExpr{Symbol}) = (push!(deps, x); deps)
_constraint_dependencies!(deps, x::StanExpr) = _constraint_dependencies!(deps, expr(x))
_constraint_dependencies!(deps, x::CanonicalExpr) = begin
    _constraint_dependencies!(deps, x.args)
    _constraint_dependencies!(deps, values(x.kwargs))
end
_constraint_dependencies!(deps, x::Union{Tuple,AbstractArray}) = begin
    foreach(v -> _constraint_dependencies!(deps, v), x)
    deps
end
_constraint_dependencies!(deps, x::NamedTuple) = _constraint_dependencies!(deps, values(x))
_constraint_dependencies!(deps, x) = deps

_validate_parameter_constraint_scope(lhs) = begin
    for (key, constraint) in pairs(constraints(type(lhs)))
        dependencies = StanExpr[]
        _constraint_dependencies!(dependencies, constraint)
        for dependency in dependencies
            dependency_qual = qual(dependency)
            dependency_role = _decl_role(dependency)
            in_scope = dependency_qual === :data ||
                (dependency_qual === :parameter && dependency_role in (:sampled, :unfilled))
            in_scope && continue
            error(
                "Parameter `$(expr(lhs))` has `$(key)=$(expr(constraint))`, but its " *
                "constraint depends on `$(expr(dependency))`, a transformed-parameter " *
                "value. Stan parameter declarations precede transformed parameters; " *
                "a constraint may reference only data or an earlier sampled parameter."
            )
        end
    end
    nothing
end

validate_sampling_rhs(lhs, rhs::StanExpr{<:CanonicalExpr}; info) = begin
    unbound = _declared_observation_lhs(lhs; info)
    unbound || _validate_parameter_constraint_scope(lhs)
    canonical = expr(rhs)
    spec = _distribution_hof(head(canonical))
    isnothing(spec) || _validate_distribution_hof(spec, lhs, canonical; unbound)
    _check_matrix_variate_args(lhs, canonical)
    nothing
end
# A `matrix` variate of an elementwise family (`z::matrix[m, n] ~ normal(…)`, or
# matrix data) resolves through the matrix overloads in builtin.jl, which take
# each argument as a scalar or as a matrix of the variate's shape (snag
# matrix-parameter-94e1f96c). Stan has no other argument form for it: a vector
# location would reach stanc as an ill-typed `z ~ normal(mu_vec, 1)`, which
# neither the call's type (`real`, the native stub's) nor `_check_lpxf_resolves`
# (the name is native-backed) flags. The families opt in in builtin.jl.
_matrix_variate_family(f) = false
_check_matrix_variate_args(lhs, rhs::CanonicalExpr) = begin
    family = head(rhs)
    _matrix_variate_family(family) || return
    center_type(lhs) <: types.matrix || return
    for (i, arg) in enumerate(rhs.args)
        (stan_ndim(arg) == 0 || center_type(arg) <: types.matrix) && continue
        error(
            "`", _base_lhs_symbol(lhs), " ~ ", nameof(family), "(…)` samples a `matrix`, but ",
            "argument ", i, " of `", nameof(family), "` is a `", sigtype(arg), "`. A matrix ",
            "variate takes each argument of an elementwise family as a scalar or as a `matrix` ",
            "of the variate's own shape; there is no `vector`, `row_vector` or array argument ",
            "form. Pass a matrix: `rep_matrix(mu, n)` repeats a `vector[m]` location `mu` ",
            "across the `n` columns, and `rep_matrix(mu, m)` repeats a `row_vector[n]` one ",
            "down the `m` rows."
        )
    end
end
_check_matrix_variate_args(lhs, rhs) = nothing
_declared_observation_lhs(lhs; info) = begin
    key = _sampling_lhs_key(lhs)
    key === nothing && return false
    key in _model_observations(info) && return true
    # The simulated trajectory of an observed `@scan` array is that observation's
    # draw (forward.jl, `_observed_scan_shadow`).
    key in keys(info) && (get(type(info[key]).info, :unbound_observation, nothing) !== nothing ||
        _obs_shadow_of(info[key]) !== nothing)
end
# Forward tracing cannot yet know whether a declared missing response is a draw
# or a latent parameter. Recheck at block placement, after activity analysis.
validate_sampled_rhs(lhs, rhs::StanExpr{<:CanonicalExpr}; info) = begin
    _validate_parameter_constraint_scope(lhs)
    spec = _distribution_hof(head(expr(rhs)))
    isnothing(spec) || _validate_distribution_hof(spec, lhs, expr(rhs))
    nothing
end
_distribution_sample_shape(at, rhs::StanExpr{<:CanonicalExpr}) = _hof_sample_shape(at, expr(rhs))
_hof_sample_shape(at, rhs::CanonicalExpr) = begin
    spec = _distribution_hof(head(rhs))
    isnothing(spec) && return at
    # Likelihood weights do not introduce a predictive sample axis. A weighted
    # base family's RNG already exposes its shape; a nested combinator's
    # controls shape the draw exactly as they would without the weights.
    if spec.name === :weighted
        family = _family_function(rhs.args[spec.family_arg])
        isnothing(_distribution_hof(family)) && return at
        return _hof_sample_shape(at, CanonicalExpr(family, rhs.args[_hof_trailing_arg(spec):end]...))
    end
    shaped = filter(a -> stan_ndim(a) > 0, rhs.args)
    isempty(shaped) && return at
    shape = stan_size(first(shaped))
    length(shape) == 1 || return at
    ct = _probability_kind(_distribution_hof_family(spec, rhs)) === :lpmf ?
        types.int : types.vector
    StanType(ct, shape, info(at))
end
_probability_kind(family) = begin
    probability = lpxf_expr(family)
    name = string(nameof(probability))
    endswith(name, "_lpdf") && return :lpdf
    endswith(name, "_lpmf") && return :lpmf
    error(
        "distribution HOF: family $(nameof(family)) resolves to $name; " *
        "expected an _lpdf or _lpmf family"
    )
end
_hof_variant(spec, rhs::CanonicalExpr) = begin
    optional = get(spec, :optional_kwargs, ())
    isempty(optional) && return spec
    present = Tuple(key for key in optional if key in keys(rhs.kwargs))
    mode = present == (:lower,) ? :lower :
        present == (:upper,) ? :upper :
        present == (:lower, :upper) ? :both : error(
            "$(spec.name): provide at least one of `lower=` or `upper=`"
        )
    getproperty(spec.variants, mode)
end
_hof_call_args(spec, rhs::CanonicalExpr) = begin
    optional = get(spec, :optional_kwargs, ())
    isempty(optional) && return rhs.args
    i = spec.family_arg
    bounds = Any[rhs.kwargs[key] for key in optional if key in keys(rhs.kwargs)]
    Tuple(vcat(collect(rhs.args[1:i]), bounds, collect(rhs.args[i+1:end])))
end
_hof_probability(spec, family, rhs) = begin
    variant = _hof_variant(spec, rhs)
    _probability_kind(family) === :lpdf ? variant.lpdf : variant.lpmf
end
_hof_pointwise(spec, family, rhs) = begin
    variant = _hof_variant(spec, rhs)
    _probability_kind(family) === :lpdf ? variant.lpdfs : variant.lpmfs
end
_hof_rng(spec, rhs) = _hof_variant(spec, rhs).rng

const CdfCapableBuiltinFamily = Union{typeof.((
    builtin.std_normal, builtin.normal, builtin.student_t, builtin.cauchy,
    builtin.beta, builtin.beta_proportion, builtin.lognormal, builtin.exponential,
    builtin.gamma, builtin.inv_gamma, builtin.weibull, builtin.uniform,
    builtin.chi_square, builtin.inv_chi_square, builtin.scaled_inv_chi_square,
    builtin.frechet, builtin.rayleigh, builtin.von_mises,
    builtin.double_exponential, builtin.logistic, builtin.gumbel,
    builtin.skew_normal, builtin.exp_mod_normal, builtin.skew_double_exponential,
    builtin.pareto, builtin.pareto_type_2,
    builtin.bernoulli, builtin.binomial, builtin.beta_binomial,
    builtin.neg_binomial, builtin.neg_binomial_2, builtin.poisson,
    builtin.discrete_range,
))...}

# Keep optional-bound mode in the call head/positional args, not only in
# `CanonicalExpr.kwargs`: later compiler passes intentionally rebuild call ASTs
# without kwargs after `autotype` has consumed declaration constraints. The
# rewrite therefore preserves the bounds twice for exactly one stage: as
# positional probability/RNG inputs, and as support constraints for autotype.
const OptionalBoundDistributionHOF = Union{
    typeof(builtin.truncated), typeof(builtin.censored)
}
# A distribution CALL in the family position — `censored(normal(mu, sigma);
# lower=lo)` — desugars to the token form `censored(normal, mu, sigma;
# lower=lo)` before the variant rewrite below. The family call's arguments are
# appended AFTER the HOF's own remaining positionals, matching each combinator's
# token-form layout (`weighted(fam, w, args...)`, `interval_censored(fam, lo, hi,
# args...)`), so data/control argument positions are unchanged by the splice and
# every downstream check (sampling validation, likelihood, RNG) sees exactly the
# token form. A family that is neither a function token nor a call keeps the
# loud `_family_function` refusal here instead of crashing in the re-trace's
# `tracetype` (snag hof-call-form-fa-b2bfce08).
_hof_call_family_splice(spec, x::CanonicalExpr) = begin
    i = spec.family_arg
    length(x.args) >= i || _distribution_hof_family(spec, x)
    fam = x.args[i]
    fam isa StanExpr2{<:types.func} && return nothing
    inner = expr(fam)
    inner isa CanonicalExpr || _distribution_hof_family(spec, x)
    fam_head = head(inner)
    fam_fn = fam_head isa StanExpr2{<:types.func} ? type(fam_head).info.value : fam_head
    fam_fn isa Function || _distribution_hof_family(spec, x)
    nested = _distribution_hof(fam_fn)
    if !isnothing(nested)
        _check_nestable(spec, nested)
        # A combinator call (`weighted(censored(normal(mu, s); lower=lo), w)`)
        # has already traced to its implementation call, whose bounds are
        # positional; its kwargs only repeat them for a parameter's `autotype`.
        # A public head still owns its bounds, for the nested token-form rewrite.
        bounds = haskey(nested, :optional_kwargs) ? inner.kwargs : (;)
        return CanonicalExpr(
            head(x),
            x.args[1:i-1]...,
            stan_expr(fam_fn),
            x.args[i+1:end]...,
            inner.args...;
            _disjoint_kwargs(spec, x.kwargs, bounds)...,
        )
    end
    isempty(inner.kwargs) || error(
        "$(spec.name): the family call `$(nameof(fam_fn))(...)` takes only " *
        "positional arguments; keyword argument(s) " *
        "`$(join(keys(inner.kwargs), ", "))` belong on `$(spec.name)` itself"
    )
    CanonicalExpr(
        head(x),
        x.args[1:i-1]...,
        stan_expr(fam_fn),
        x.args[i+1:end]...,
        inner.args...;
        x.kwargs...,
    )
end
# The implementation call a bounded combinator lowers to: optional bounds select
# the lower/upper/two-sided variant and become positional, `interval_censored`
# renames to its implementation head, and a variant head is already lowered.
_hof_impl_call(x::CanonicalExpr{<:OptionalBoundDistributionHOF}; kwargs...) = begin
    spec = _distribution_hof(head(x))
    CanonicalExpr(_hof_variant(spec, x).head, _hof_call_args(spec, x)...; kwargs...)
end
_hof_impl_call(x::CanonicalExpr{typeof(builtin.interval_censored)}; kwargs...) =
    CanonicalExpr(builtin.interval_evidence_impl, x.args...; kwargs...)
_hof_impl_call(x::CanonicalExpr; kwargs...) = CanonicalExpr(head(x), x.args...; kwargs...)
expand_inline_or_trace(
    x::CanonicalExpr{<:Union{OptionalBoundDistributionHOF,typeof(builtin.interval_censored)}};
    info,
) = begin
    spliced = _hof_call_family_splice(_distribution_hof(head(x)), x)
    spliced !== nothing && return forward!(spliced; info)
    forward!(_hof_impl_call(x; x.kwargs...); info)
end
# Token form of a nested combinator — `weighted(censored, w, normal, mu, sigma;
# lower=lo)` — lowers its family to the implementation call, exactly as the
# combinator alone would: `weighted(lower_clamping, w, normal, lo, mu, sigma)`.
# The forwarded bounds belong to that combinator, so a base family refuses them
# rather than silently dropping them.
_nested_hof_call(spec, x::CanonicalExpr) = begin
    i = spec.family_arg
    family = _family_function(x.args[i])
    forwarded = get(spec, :forwarded_kwargs, ())
    bounds = (; (k => v for (k, v) in pairs(x.kwargs) if k in forwarded)...)
    rest = (; (k => v for (k, v) in pairs(x.kwargs) if !(k in forwarded))...)
    nested = _distribution_hof(family)
    if isnothing(nested)
        isempty(bounds) || error(
            "$(spec.name): `", join(keys(bounds), "=`, `"), "=` bounds a combinator ",
            "in the family slot — `$(spec.name)(censored($(nameof(family))(…); lower=lo), w)` ",
            "or `$(spec.name)(censored, w, $(nameof(family)), …; lower=lo)`; ",
            "`$(nameof(family))` alone is a base family",
        )
        return nothing
    end
    haskey(nested, :optional_kwargs) || isempty(bounds) || error(
        "$(spec.name): `", join(keys(bounds), "=`, `"), "=` on `$(spec.name)` has no ",
        "`truncated`/`censored` token to bound — a combinator call carries its own ",
        "bounds, and `interval_censored` takes them positionally",
    )
    j = _hof_trailing_arg(spec)
    inner = CanonicalExpr(family, x.args[j:end]...; bounds...)
    impl = _hof_impl_call(something(_hof_call_family_splice(nested, inner), inner))
    head(impl) === family && return nothing
    CanonicalExpr(
        head(x),
        x.args[1:i-1]...,
        stan_expr(head(impl)),
        x.args[i+1:j-1]...,
        impl.args...;
        rest...,
    )
end
# Companion selection (`lpxf_expr`/`likelihood_expr`/`rng_expr`) bypasses this
# expansion, so a call built directly — a ragged group's twin — lowers its
# nested combinator token here.
_normalized_hof_call(rhs::CanonicalExpr) = begin
    spec = _distribution_hof(head(rhs))
    isnothing(spec) && return rhs
    something(_nested_hof_call(spec, rhs), rhs)
end
expand_inline_or_trace(
    x::CanonicalExpr{typeof(builtin.weighted)};
    info,
) = begin
    spec = _distribution_hof(head(x))
    spliced = _hof_call_family_splice(spec, x)
    spliced !== nothing && return forward!(spliced; info)
    nested = _nested_hof_call(spec, x)
    nested !== nothing && return forward!(nested; info)
    invoke(expand_inline_or_trace, Tuple{CanonicalExpr}, x; info)
end
_family_probability_companion(family, suffix::Symbol) = begin
    if parentmodule(family) === builtin && !(family isa CdfCapableBuiltinFamily)
        error(
            "distribution HOF: built-in family $(nameof(family)) has no Stan " *
            "`$(nameof(family))$(suffix)` companion; choose a CDF-capable family"
        )
    end
    probability = lpxf_expr(family)
    probability_name = string(nameof(probability))
    probability_suffix = _probability_kind(family) === :lpdf ? "_lpdf" : "_lpmf"
    base_name = probability_name[1:end-length(probability_suffix)]
    companion_name = Symbol(base_name, suffix)
    mod = parentmodule(probability)
    isdefined(mod, companion_name) || error(
        "distribution HOF: family $(nameof(family)) requires companion " *
        "`$companion_name`, but it is not defined in $mod"
    )
    getfield(mod, companion_name)
end
logcdf_expr(family) = _family_probability_companion(family, :_lcdf)
logccdf_expr(family) = _family_probability_companion(family, :_lccdf)
# `normal` overrides its Stan-native companions with the erfc reformulation
# (builtin.jl): Stan Math's `normal_lcdf`/`normal_lccdf` reverse-mode AD rules
# are inaccurate (~1e-6) and `normal_lccdf` diverges past ~9σ, so every
# `censored`/`truncated`/`interval_censored(normal, …)` tail mass would inherit
# a wrong gradient. The erfc form `log(0.5·erfc(∓z/√2))` autodiffs exactly and
# is value-identical. See snag censored-normal-410c4e35.
logcdf_expr(::typeof(builtin.normal)) = builtin.normal_lcdf_stable
logccdf_expr(::typeof(builtin.normal)) = builtin.normal_lccdf_stable

lpxf_expr(lhs, rhs::StanExpr) = lpxf_expr(lhs, expr(rhs))
lpxf_expr(lhs, rhs::CanonicalExpr) = begin
    spec = _distribution_hof(head(rhs))
    isnothing(spec) && return stan_call(lpxf_expr(head(rhs)), lhs, rhs.args...)
    _validate_distribution_hof(spec, lhs, rhs)
    family = _distribution_hof_family(spec, rhs)
    stan_call(_hof_probability(spec, family, rhs), lhs, _hof_call_args(spec, rhs)...)
end
for lpxf_rhs in (
    :dummy_lpdf,
    :truncated_normal_lpdf,
    :truncated_student_t_lpdf,
    :flat_lpdf, :std_normal_lpdf, :normal_lpdf, :student_t_lpdf, :cauchy_lpdf,
    :beta_lpdf, :beta_proportion_lpdf, :lognormal_lpdf, :exponential_lpdf, :gamma_lpdf,
    :inv_gamma_lpdf, :weibull_lpdf, :uniform_lpdf,
    :chi_square_lpdf, :inv_chi_square_lpdf, :scaled_inv_chi_square_lpdf,
    :frechet_lpdf, :rayleigh_lpdf, :loglogistic_lpdf, :von_mises_lpdf,
    :double_exponential_lpdf, :logistic_lpdf, :gumbel_lpdf,
    :skew_normal_lpdf, :exp_mod_normal_lpdf, :skew_double_exponential_lpdf,
    :pareto_lpdf, :pareto_type_2_lpdf, :wiener_lpdf,
    :dirichlet_lpdf, :multi_normal_lpdf, :multi_normal_prec_lpdf, :multi_normal_cholesky_lpdf,
    :multi_gp_lpdf, :multi_gp_cholesky_lpdf,
    :multi_student_t_lpdf, :multi_student_t_cholesky_lpdf,
    :gaussian_dlm_obs_lpdf,
    :lkj_corr_lpdf, :lkj_corr_cholesky_lpdf,
    :wishart_lpdf, :inv_wishart_lpdf, :inv_wishart_cholesky_lpdf, :wishart_cholesky_lpdf,
    :normal_id_glm_lpdf,
    :bernoulli_lpmf, :bernoulli_logit_lpmf, :bernoulli_logit_glm_lpmf,
    :binomial_lpmf, :binomial_logit_lpmf, :beta_binomial_lpmf,
    :neg_binomial_lpmf, :neg_binomial_2_lpmf, :neg_binomial_2_log_lpmf,
    :neg_binomial_2_log_glm_lpmf,
    :poisson_lpmf, :poisson_log_lpmf, :poisson_log_glm_lpmf,
    :discrete_range_lpmf, :hypergeometric_lpmf, :multinomial_lpmf,
    :categorical_lpmf, :categorical_logit_lpmf,
    :ordered_logistic_lpmf,
)
    base_rhs = Symbol(string(lpxf_rhs)[1:end-5])
    rng_rhs = Symbol(base_rhs, "_rng")
    lpxfs_rhs = Symbol(lpxf_rhs, "s")

    @eval lpxf_expr(::typeof(builtin.$base_rhs)) = builtin.$lpxf_rhs
    @eval rng_expr(::typeof(builtin.$base_rhs)) = builtin.$rng_rhs
    @eval likelihood_expr(::typeof(builtin.$base_rhs)) = builtin.$lpxfs_rhs
end

# Dual-kind closure families: one bare `~` name serving BOTH real and integer
# observations. Stan resolves a bare `y ~ foo(…)` to the `_lpdf`/`_lpmf`
# specialization by the variate type itself (stanc: "`~` should refer to a
# distribution without its suffix"), so the model emission needs no variant
# selection — but the DEFINITIONS must match: a closure-specialised `_lpdf`
# wrapper with an int first arg is illegal Stan, so for int lhs the fetch must
# pull the `_lpmf` specialization INSTEAD of the `_lpdf` one. `_dual_lpmf_call`
# builds that call; `fetch_functions!(::SamplingExpr)` consults it. The GQ
# likelihood twin (`_lpdfs`, a plain function) and the rng draw (token
# dispatch to the `int[T]` overload) need no selection. Families opt in by
# defining the trait; everything else falls through to `nothing` (zero
# behaviour change outside opt-in families).
_dual_lpmf_variant(family) = nothing
_dual_lpmf_variant(::typeof(builtin.hmm_forward)) = builtin.hmm_forward_lpmf
_dual_lpmf_call(lhs, rhs::StanExpr) = _dual_lpmf_call(lhs, expr(rhs))
_dual_lpmf_call(lhs, rhs::CanonicalExpr) = begin
    variant = _dual_lpmf_variant(head(rhs))
    isnothing(variant) && return nothing
    center_type(lhs) <: types.int || return nothing
    stan_call(variant, lhs, rhs.args...)
end
_dual_lpmf_call(lhs, rhs) = nothing

lpxf_expr(x) = error("$x is missing `lpxf_expr`")
likelihood_expr(lhs, rhs::StanExpr) = likelihood_expr(lhs, expr(rhs))
likelihood_expr(lhs, rhs::CanonicalExpr) = begin
    spec = _distribution_hof(head(rhs))
    isnothing(spec) && return stan_call(likelihood_expr(head(rhs)), lhs, rhs.args...)
    _validate_distribution_hof(spec, lhs, rhs)
    family = _distribution_hof_family(spec, rhs)
    stan_call(_hof_pointwise(spec, family, rhs), lhs, _hof_call_args(spec, rhs)...)
end
likelihood_expr(rhs) = error("$rhs is missing `likelihood_expr`")
# gq `~` synthesis: `rng_expr(token, rhs)` builds either `rng_fn(args...)` (for
# scalar tokens — matches Stan's native rng signatures) or `rng_fn(token, args...)`
# (for sized tokens — dispatched into per-shape `*_rng` @deffun overloads).
# The token is a `tokenof{T}` StanExpr carrying the wanted output shape.
rng_expr(token, rhs::StanExpr) = rng_expr(token, expr(rhs))
# `std_normal` is Stan's one ZERO-argument family, and every distribution HOF's
# rng body reaches its base draw as `predictive(family, args...)` — which with no
# trailing args is the SELECTOR's token-returning form, not a call (the
# `DistributionFamilySelector` contract below). So `lower_conditioning_rng(std_normal, lo)`
# traced its draw as a `func`, and re-drawing a truncated `std_normal` prior died
# in `fetch_functions!`/`sig_expr` with "NamedTuple has no field value". For the
# RNG call only, spell the family as `normal(0, 1)` — the same law; the density
# and pointwise companions keep emitting the `std_normal` forms they always did.
# Family args are the TRAILING positional args of every HOF rng
# (`<variant>_rng(family, bounds…, args...)`), so appending is layout-safe.
_hof_rng_args(spec, rhs::CanonicalExpr) = begin
    args = _hof_call_args(spec, rhs)
    i = _hof_base_family_index(spec, args)
    _family_function(args[i]) === builtin.std_normal || return args
    (args[1:i-1]..., stan_expr(builtin.normal), args[i+1:end]..., stan_expr(0.0), stan_expr(1.0))
end
# Scalar token path: native Stan rng, no token forwarding.
rng_expr(token::StanExpr2{<:types.tokenof,0}, rhs::CanonicalExpr) = begin
    spec = _distribution_hof(head(rhs))
    isnothing(spec) && return stan_call(rng_expr(head(rhs)), rhs.args...)
    _distribution_hof_family(spec, rhs)
    stan_call(_hof_rng(spec, rhs), _hof_rng_args(spec, rhs)...)
end
# Sized token path: prepend token so per-shape @deffun overloads dispatch.
rng_expr(token::StanExpr2{<:types.tokenof}, rhs::CanonicalExpr) = begin
    spec = _distribution_hof(head(rhs))
    isnothing(spec) && return stan_call(rng_expr(head(rhs)), token, rhs.args...)
    _distribution_hof_family(spec, rhs)
    stan_call(_hof_rng(spec, rhs), token, _hof_rng_args(spec, rhs)...)
end
rng_expr(x) = begin
    spec = _distribution_hof(x)
    isnothing(spec) && error("$x is missing `rng_expr`")
    spec.rng
end
# A sampled symbol's own USER bound — `sigma ~ normal(0, 1; lower=0)`, the
# half-normal — is a truncation of the family (`stanblocks-use` §8). Bounds the
# family already implies (`exponential` ⇒ `lower=0`) are its support, not a
# truncation: a bound that merely restates one is skipped, a different one
# (`exponential(1; lower=0.5)`) truncates. `offset=`/`multiplier=` are
# reparameterisations, never truncations.
_bound_value(x::StanExpr) = _bound_value(expr(x))
_bound_value(x) = x
_restates_implied_bound(user, implied) =
    implied !== nothing && isequal(_bound_value(user), _bound_value(implied))
_truncation_bounds(rhs::CanonicalExpr) = begin
    isnothing(_distribution_hof(head(rhs))) || return (;)   # a HOF owns its own bounds
    implied = autokwargs(rhs)
    (; [
        key => rhs.kwargs[key]
        for key in BOUND_KEYS
        if key in keys(rhs.kwargs) && !_restates_implied_bound(rhs.kwargs[key], get(implied, key, nothing))
    ]...)
end
# The implementation call an authored `truncated(family, args...; bounds...)`
# lowers to (`expand_inline_or_trace`), built from a base-family call.
_truncated_impl_call(rhs::CanonicalExpr, bounds; kwargs...) = begin
    spec = _distribution_hof(builtin.truncated)
    call = CanonicalExpr(builtin.truncated, stan_expr(head(rhs)), rhs.args...; bounds...)
    CanonicalExpr(_hof_variant(spec, call).head, _hof_call_args(spec, call)...; kwargs...)
end
# When that symbol is RE-DRAWN in generated quantities — prior-only lowering with
# no likelihood, or a cv-flipped re-draw — the draw must come from the truncated
# prior: `normal_rng(0, 1)` for a `<lower=0>` half-normal is a negative scale half
# the time, silently (snag prior-predictive-7e463983). Route such a re-draw
# through the `truncated` HOF's rejection sampler, exactly as an authored
# `sigma ~ truncated(normal, 0, 1; lower=0)` would draw. A bounded observation
# samples from the `truncated` call itself (`_normalised_truncation`), so its
# `<obs>_gen` twin draws from it through `rng_expr`.
_user_bounded_redraw(rhs::CanonicalExpr) = begin
    bounds = _truncation_bounds(rhs)
    isempty(bounds) && return rhs
    rng_expr(head(rhs))   # a family with no rng at all fails here, with its own message
    _truncated_impl_call(rhs, bounds)
end
redraw_rng_expr(token, rhs::StanExpr) = redraw_rng_expr(token, expr(rhs))
redraw_rng_expr(token, rhs::CanonicalExpr) = rng_expr(token, _user_bounded_redraw(rhs))
# The FITTED density carries the truncation too. The model block spells the bound
# as the parameter's `<lower=0>` declaration plus its `~` statement, and a bare
# `sigma ~ normal(0, s)` is the truncated density only because the normaliser it
# drops, `-log(1 - F(0 | 0, s))`, is a constant. With a parameter location —
# `x ~ normal(mu, 1; lower=0)` — it is not: the bare family silently gave `mu`
# the prior `N(mu | 0, 1)·Φ(mu)` while the re-draw above drew from the truncated
# law (todo 1l1odrx). Such a statement samples from the call an authored
# `truncated(normal, mu, 1; lower=0)` lowers to; a provably constant normaliser
# keeps the lean native statement. `support` adds the bound a declaration implies
# without a kwarg (`positive_ordered`, `_center_support`).
#
# An `observed` LHS is truncated whatever its normaliser: data carry no declared
# bound, so only the `truncated` call checks the support, normalises the
# pointwise `<obs>_likelihood` and draws `<obs>_gen` from the truncated law. A
# bare call silently ignored the bound in all three.
_normalised_truncation(rhs::StanExpr; support=(;), observed=false) = begin
    c = expr(rhs)
    c isa CanonicalExpr || return rhs
    bounds = _truncation_bounds(remake(c, c.args...; support..., c.kwargs...))
    isempty(bounds) && return rhs
    !observed && _constant_truncation_normaliser(c, bounds) && return rhs
    StanExpr(_truncated_impl_call(c, bounds; c.kwargs...), type(rhs))
end
_normalised_truncation(rhs; support=(;), observed=false) = rhs
# The normaliser is constant when it reads no parameter: every family argument and
# bound is data. A family symmetric about its location and cut once, at that
# location, keeps half its mass whatever its other arguments: `normal(0, s;
# lower=0)` is the half-normal for every `s`, normaliser `log 2`.
_constant_truncation_normaliser(rhs::CanonicalExpr, bounds) =
    all(a -> qual(a) == :data, (rhs.args..., values(bounds)...)) ||
    _cut_at_symmetry_centre(head(rhs), rhs.args, bounds)
_symmetry_centre(family) = nothing
_symmetry_centre(::Union{typeof.((
    builtin.normal, builtin.cauchy, builtin.logistic, builtin.double_exponential,
))...}) = 1
_symmetry_centre(::typeof(builtin.student_t)) = 2
_cut_at_symmetry_centre(family, args, bounds) = begin
    i = _symmetry_centre(family)
    (i === nothing || length(bounds) != 1 || length(args) < i) && return false
    _same_bound(only(values(bounds)), args[i])
end

# Base-family companion selectors are compile-time calls. With no trailing
# arguments they return the selected function token; with trailing arguments
# they immediately trace a call to that function. In either form the selector
# itself is absent from emitted Stan.
const DistributionFamilySelector = Union{
    typeof(builtin.density), typeof(builtin.pointwise), typeof(builtin.predictive),
    typeof(builtin.logcdf), typeof(builtin.logccdf)
}
_family_selector_target(::typeof(builtin.density), family) = lpxf_expr(family)
_family_selector_target(::typeof(builtin.pointwise), family) = likelihood_expr(family)
_family_selector_target(::typeof(builtin.predictive), family) = rng_expr(family)
_family_selector_target(::typeof(builtin.logcdf), family) = logcdf_expr(family)
_family_selector_target(::typeof(builtin.logccdf), family) = logccdf_expr(family)
_family_selector_target(selector, family, rest) = _family_selector_target(selector, family)
# A nested combinator (`weighted`'s `density(lower_clamping, y, normal, lo, mu,
# sigma)`) has no fixed companion: its `_lpdf`/`_lpmf` twin follows its own base
# family, which trails the observation. Select it exactly as its `~` would.
_family_selector_target(
    selector::Union{typeof(builtin.density),typeof(builtin.pointwise)}, family, rest,
) = begin
    spec = _distribution_hof(family)
    isnothing(spec) && return _family_selector_target(selector, family)
    call = CanonicalExpr(family, rest[2:end]...)
    base = _distribution_hof_family(spec, call)
    selector === builtin.density ? _hof_probability(spec, base, call) :
        _hof_pointwise(spec, base, call)
end
expand_inline_or_trace(
    x::CanonicalExpr{<:DistributionFamilySelector,<:Tuple{<:StanExpr2{<:types.func},Vararg{Any}}};
    info,
) = begin
    family = _family_function(x.args[1])
    rest = x.args[2:end]
    selected = _family_selector_target(head(x), family, rest)
    isempty(rest) ? forward!(selected; info) :
        forward!(CanonicalExpr(selected, rest...); info)
end
