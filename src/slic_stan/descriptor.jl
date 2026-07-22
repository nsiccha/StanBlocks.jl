# --- The descriptor: one reflectable, executable model declaration -----------
#
# A `@slic` model is already the single authoritative declaration of a Stan
# program — the tracer knows every data input, every parameter, every generated
# quantity, and (since the `_gen`/`_likelihood` expansion) which generated
# quantity is the predictive twin of which observation. Until now none of that
# was reachable from outside: a consumer wanting to render a model as a UI, or
# to offer "fit"/"predict" buttons, had to maintain its own parallel list of
# operations and its own idea of what each output meant.
#
# `stan_descriptor(model)` exposes that knowledge as data:
#
#   * a stable, content-derived `id` (the same hash `instantiate` caches by, so
#     the descriptor id and the compiled artifact agree by construction),
#   * `inputs`  — the data block, with declared type, size, constraints, whether
#     a value is bound, whether it is observed, and whether it is cv-held-out,
#   * `outputs` — parameters / transformed parameters / generated quantities,
#     each tagged with its GENERATIVE ROLE (`:posterior`, `:draw`,
#     `:pointwise_loglik`, `:derived`) and, for a twin, the observation it
#     derives from,
#   * `operations` — DERIVED from the two above, never hand-listed. A model with
#     no likelihood term offers no `:fit`; a model with no `_gen` twin offers no
#     `:predict`. Each operation carries its own executor.
#
# Everything here is read-only reflection over an already-traced `StanModel`;
# nothing in this file participates in tracing or emission.

"""
    ModelInput

One entry of a descriptor's data block — see [`stan_descriptor`](@ref
StanBlocks.stan_descriptor).

# Fields
- `name::Symbol` — the Stan data variable name.
- `type::Symbol` — the declared Stan center type (`:real`, `:int`, `:vector`,
  `:matrix`, `:simplex`, …).
- `size::Tuple` — the declared size expressions, outermost first. Entries are
  *emitted expressions*: a literal, a `Symbol` naming another input, or a larger
  expression. Resolve a symbolic entry against the other inputs' `value`.
- `constraints::NamedTuple` — the `lower`/`upper`/`offset`/`multiplier` subset
  actually spelled on the declaration. Empty does **not** mean unconstrained:
  a natively constrained center (`simplex`, `cholesky_factor_corr`, …) carries
  its support in `type`.
- `value` — the bound Julia value. Always present: a symbol with no value never
  reaches the data block (an unresolvable one fails at *tracing* time, not here).
- `inlined::Bool` — `true` for a value the emitter folds into the generated
  source rather than passing as data (functions, closures, 0-dim type tokens).
- `derived::Bool` — `true` when this input is another input's declared size
  (`y_n` for `vector[y_n] y`). Re-binding the container re-derives it, so a
  consumer must never ask for it separately.
  Neither an `inlined` nor a `derived` input is part of
  [`required_inputs`](@ref StanBlocks.required_inputs), and neither belongs in a
  generated form.
- `observed::Bool` — this variable appears on the left of a `~` (it is data the
  model conditions on), as opposed to a covariate or a size.
- `held_out::Bool` — this variable was marked via `StanBlocks.stan.maybecv`, so
  its likelihood contribution is dropped and it is re-drawn in generated
  quantities (the cross-validation flag).
"""
struct ModelInput
    name::Symbol
    type::Symbol
    size::Tuple
    constraints::NamedTuple
    value::Any
    inlined::Bool
    derived::Bool
    observed::Bool
    held_out::Bool
end

"""
    ModelOutput

One quantity the model produces — see [`stan_descriptor`](@ref
StanBlocks.stan_descriptor).

# Fields
- `name::Symbol` — the Stan variable name (matches BridgeStan's `param_names`
  prefix; a container appears there as `name.1`, `name.2`, …).
- `kind::Symbol` — which Stan block declares it: `:parameter`,
  `:transformed_parameter`, or `:generated_quantity`.
- `type::Symbol`, `size::Tuple`, `constraints::NamedTuple` — as for
  [`ModelInput`](@ref StanBlocks.ModelInput).
- `generative::Symbol` — what the quantity MEANS:
  - `:posterior` — a sampled parameter or a deterministic function of one.
  - `:draw` — a predictive draw of an observation (the compiler-owned
    `<obs>_gen` twin). `source` names the observation.
  - `:pointwise_loglik` — the per-element log-likelihood of an observation
    (the compiler-owned `<obs>_likelihood` companion). `source` names it.
  - `:derived` — any other generated quantity: a prior-only sample, a
    cv-flipped re-draw of a latent, a model `return` value.
- `source::Union{Symbol,Nothing}` — the observation a `:draw` /
  `:pointwise_loglik` derives from; `nothing` otherwise.
"""
struct ModelOutput
    name::Symbol
    kind::Symbol
    type::Symbol
    size::Tuple
    constraints::NamedTuple
    generative::Symbol
    source::Union{Symbol,Nothing}
end

"""
    ModelOperation

One executable operation DERIVED from a model declaration — never a
consumer-maintained entry. See [`stan_descriptor`](@ref
StanBlocks.stan_descriptor) for the derivation rules and
[`stan_execute`](@ref StanBlocks.stan_execute) to run one.

# Fields
- `name::Symbol` — `:transpile`, `:instantiate`, `:fit`, `:predict`, or
  `:pointwise_loglik`.
- `title::String` — a human label a UI can render without a lookup table.
- `inputs::Tuple{Vararg{Symbol}}` — descriptor input names this operation
  consumes (empty for `:transpile`, which needs no data).
- `outputs::Tuple{Vararg{Symbol}}` — descriptor output names it produces.
- `run` — the executor, called as `run(descriptor; kwargs...)`.
"""
struct ModelOperation
    name::Symbol
    title::String
    inputs::Tuple{Vararg{Symbol}}
    outputs::Tuple{Vararg{Symbol}}
    run::Any
end

"""
    ModelDescriptor

The reflectable, executable declaration of one Stan model — the value returned
by [`stan_descriptor`](@ref StanBlocks.stan_descriptor).

# Fields
- `id::String` — stable identity, derived from the generated Stan source (the
  same key [`instantiate`](@ref StanBlocks.instantiate) caches the compiled
  artifact under). Two models with byte-identical Stan share an `id`; changing
  the model changes it. Independent of the process, of tracing order, and of
  the gensym'd `name`.
- `name::Symbol` — the model's name. Informational only — `@slic begin … end`
  produces a gensym; pass `name=` to [`stan_descriptor`](@ref
  StanBlocks.stan_descriptor) to set it.
- `docstring::String` — the model's leading docstring, if any.
- `model::StanModel` — the traced model the descriptor reflects.
- `inputs::Tuple{Vararg{ModelInput}}`
- `outputs::Tuple{Vararg{ModelOutput}}`
- `operations::Tuple{Vararg{ModelOperation}}`
"""
struct ModelDescriptor
    id::String
    name::Symbol
    docstring::String
    model::StanModel
    inputs::Tuple{Vararg{ModelInput}}
    outputs::Tuple{Vararg{ModelOutput}}
    operations::Tuple{Vararg{ModelOperation}}
end

# --- reflection helpers ------------------------------------------------------

# `Base.show(::Type{<:types.anything})` prints `T.name.name`; the descriptor
# reports that same Symbol so `type` round-trips against the emitted Stan.
_descriptor_type(x::StanExpr) = _descriptor_type(center_type(x))
_descriptor_type(T::Type) = T.name.name
_descriptor_type(::Any) = :anything

# Size entries are EMITTED expressions, not values: a literal, a `Symbol` naming
# another declaration, or a larger expression. Unwrap the StanExpr wrapper so a
# consumer sees the expression itself.
_descriptor_size(x::StanExpr) = map(expr, stan_size(x))

# Every bare name a size expression mentions — `y_n` in `vector[y_n]`, both of
# `m` and `n` in `matrix[m, n]`, and `n` in a composite `n - 1`.
_size_symbols(x, acc) = acc
_size_symbols(x::Symbol, acc) = (push!(acc, x); acc)
_size_symbols(x::StanExpr, acc) = _size_symbols(expr(x), acc)
_size_symbols(x::CanonicalExpr, acc) = (foreach(a -> _size_symbols(a, acc), x.args); acc)

_descriptor_constraints(x::StanExpr) = _descriptor_constraints(info(type(x)))
_descriptor_constraints(i) = (; [
    key => i[key] for key in (:lower, :upper, :offset, :multiplier) if haskey(i, key)
]...)

# --- observations ------------------------------------------------------------
#
# Which data variables the model CONDITIONS on takes TWO passes, because neither
# alone is complete:
#
#  (1) the model block's `~` statements. Authoritative, and the only signal for
#      a RAGGED observation, which has no declarable Stan shape and therefore no
#      `_gen` twin at all.
#  (2) the `<stem>_gen` / `<stem>_likelihood` declarations in generated
#      quantities, WHERE `<stem>` IS ITSELF A DATA INPUT. Needed because a
#      cv-held-out observation routes to generated quantities ONLY — and by then
#      it is an ASSIGNMENT (`y_gen = normal_vector_rng(…)`), not a `~`, so pass
#      (1) cannot see it.
#
# Pass (2) is a name match, but not a guess: SLIC mints those two names from
# nowhere except a data-qualified sampling LHS, and anchoring the stem in the
# data block is what makes it unambiguous. A user variable that happens to
# collide is caught by `_assert_distinct_names`, not silently mislabelled.
_observed_bases(x, acc) = acc
_observed_bases(x::AbstractVector, acc) = (foreach(s -> _observed_bases(s, acc), x); acc)
_observed_bases(x::AbstractDict, acc) = (foreach(s -> _observed_bases(s, acc), values(x)); acc)
_observed_bases(x::StanBlock, acc) = _observed_bases(content(x), acc)
_observed_bases(x::BlockExpr, acc) = _observed_bases(x.args, acc)
_observed_bases(x::DocumentExpr, acc) = _observed_bases(x.args[2], acc)
_observed_bases(x::StanExpr{<:ForExpr}, acc) = _observed_bases(expr(x).args[2], acc)
_observed_bases(x::StanExpr{<:CanonicalExpr}, acc) = _observed_bases(expr(x), acc)
_observed_bases(x::SamplingExpr, acc) = begin
    lhs = x.args[1]
    qual(lhs) == :data || return acc
    # A sampling LHS may be an arbitrary data-qualified expression
    # (`(sum(a) + n) ~ normal(…)` is legal), so accept only a bare Symbol or a
    # genuine `base[i…]` getindex chain — the same strictness `_gen` twin
    # retargeting uses.
    k = expr(lhs) isa Symbol ? expr(lhs) : _getindex_chain_base(lhs)
    k isa Symbol && push!(acc, k)
    acc
end

# --- outputs -----------------------------------------------------------------

# `parameters` is declarative (an OrderedDict keyed by name); the transformed
# and generated blocks are imperative statement lists, so a declaration there is
# whatever an assignment binds.
_output_symbols(x, acc) = acc
_output_symbols(x::AbstractVector, acc) = (foreach(s -> _output_symbols(s, acc), x); acc)
_output_symbols(x::AbstractDict, acc) = (foreach(s -> _output_symbols(s, acc), values(x)); acc)
_output_symbols(x::StanBlock, acc) = _output_symbols(content(x), acc)
_output_symbols(x::BlockExpr, acc) = _output_symbols(x.args, acc)
_output_symbols(x::DocumentExpr, acc) = _output_symbols(x.args[2], acc)
# A compiler-owned loop declares nothing at block scope: the base it fills is
# declared separately (`_push_obs_gen_decl!`), so descending would double-count
# the loop index instead.
_output_symbols(::StanExpr{<:ForExpr}, acc) = acc
_output_symbols(x::StanExpr{Symbol}, acc) = (get!(acc, expr(x), x); acc)
_output_symbols(x::StanExpr{<:DeclExpr}, acc) = acc
_output_symbols(x::AssignmentExpr, acc) = begin
    lhs = x.args[1]
    expr(lhs) isa Symbol && get!(acc, expr(lhs), lhs)
    acc
end
_output_symbols(x::SamplingExpr, acc) = begin
    lhs = x.args[1]
    expr(lhs) isa Symbol && get!(acc, expr(lhs), lhs)
    acc
end

# Generative role of a generated quantity: the compiler-owned twin of `stem`
# when the name is `<stem>_gen` / `<stem>_likelihood` AND `stem` is a data input
# of this model (pass (2) above), `:derived` otherwise.
_generative_role(name::Symbol, datanames) = begin
    s = string(name)
    for (suffix, role) in (("_likelihood", :pointwise_loglik), ("_gen", :draw))
        endswith(s, suffix) || continue
        stem = Symbol(s[1:end-length(suffix)])
        stem in datanames && return (role, stem)
    end
    (:derived, nothing)
end

_block_outputs(m::StanModel, blockname::Symbol, kind::Symbol, datanames) = begin
    acc = OrderedDict{Symbol,Any}()
    _output_symbols(block(m, blockname), acc)
    rv = ModelOutput[]
    for (name, x) in pairs(acc)
        always_inline(x) && continue
        generative, source = kind === :generated_quantity ?
            _generative_role(name, datanames) : (:posterior, nothing)
        push!(rv, ModelOutput(
            name, kind, _descriptor_type(x), _descriptor_size(x),
            _descriptor_constraints(x), generative, source,
        ))
    end
    rv
end

# --- the descriptor ----------------------------------------------------------

"""
    stan_descriptor(model; name=nothing) -> ModelDescriptor

Reflect `model` (a [`SlicModel`](@ref StanBlocks.SlicModel) or an
already-traced [`StanModel`](@ref StanBlocks.StanModel)) as one
descriptor-bearing, executable declaration: stable identity, inputs, outputs
with their generative semantics, and the operations DERIVED from them.

A `SlicModel` is traced first (via [`stan_model`](@ref
StanBlocks.stan_model)); prefer passing a `StanModel` when reflecting the same
model repeatedly, since tracing is the expensive step.

`name` overrides the model's own (gensym'd, for an anonymous `@slic begin … end`)
name. It is informational — `id` is derived from the generated Stan source, not
from the name.

# Derived operations

Nothing here is a hand-maintained list; each operation appears exactly when the
traced model supports it.

| operation | offered when |
| --- | --- |
| `:transpile` | always |
| `:instantiate` | always (fails loudly at run time if a required input is unbound) |
| `:fit` | the model has ≥1 parameter **and** ≥1 likelihood term — i.e. a non-held-out observation reaches the `model` block. A pure prior-predictive model, or one whose every observation is cv-held-out, offers no `:fit` |
| `:predict` | ≥1 output with `generative == :draw` |
| `:pointwise_loglik` | ≥1 output with `generative == :pointwise_loglik` |

Ask for one with [`stan_operation`](@ref StanBlocks.stan_operation) and run it
with [`stan_execute`](@ref StanBlocks.stan_execute); both fail closed on an
operation the model does not offer.

# Example

```julia
m = @slic begin
    sigma ~ exponential(1.)
    mu ~ normal(0., 10.)
    y ~ normal(mu, sigma)
end (; y)

d = stan_descriptor(m; name=:simple)
d.id                                        # "…" — stable, content-derived
required_inputs(d)                          # (:y,) — not the derived size `y_n`
[i.name for i in d.inputs if i.observed]    # [:y]
[o.name for o in d.outputs if o.generative == :draw]   # [:y_gen]
[op.name for op in d.operations]            # [:transpile, :instantiate, :fit, :predict, :pointwise_loglik]

stan_execute(d, :transpile)                 # the Stan source
prob = stan_execute(d, :fit)                # a BridgeStan-backed StanProblem
stan_execute(d, :predict; problem=prob, draws=theta_unc, seed=1)   # (; y_gen = […])
```

See also [`stan_operation`](@ref StanBlocks.stan_operation),
[`stan_execute`](@ref StanBlocks.stan_execute),
[`required_inputs`](@ref StanBlocks.required_inputs).
"""
stan_descriptor(x::SlicModel; kwargs...) = stan_descriptor(stan_model(x); kwargs...)
stan_descriptor(m::StanModel; name=nothing) = begin
    code = stan_code(m)
    data = content(block(m, :data))
    datanames = Set{Symbol}(keys(data))

    outputs = vcat(
        _block_outputs(m, :parameters, :parameter, datanames),
        _block_outputs(m, :transformed_parameters, :transformed_parameter, datanames),
        _block_outputs(m, :generated_quantities, :generated_quantity, datanames),
    )
    # Observations: pass (1) the model block's `~`, pass (2) the twins just
    # classified. See the two-pass note above `_observed_bases`.
    observed = _observed_bases(block(m, :model), Set{Symbol}())
    for o in outputs
        o.source === nothing || push!(observed, o.source)
    end

    # A data variable that is another data variable's declared SIZE (`y_n` for
    # `vector[y_n] y`) is not something a consumer supplies: re-binding the
    # container re-derives it (`_record_size_kwargs!`). Asking for it in a
    # generated form is both noise and an opportunity to contradict the data.
    sizes = Set{Symbol}()
    for (_, x) in pairs(data), s in stan_size(x)
        _size_symbols(expr(s), sizes)
    end
    inputs = ModelInput[]
    for (key, x) in pairs(data)
        push!(inputs, ModelInput(
            key, _descriptor_type(x), _descriptor_size(x), _descriptor_constraints(x),
            getvalue(x), always_inline(x), key in sizes,
            key in observed, cv(x),
        ))
    end

    d = ModelDescriptor(
        string(hash(code); base=16),
        something(name, meta(m).name),
        get(meta(m), :docstring, ""),
        m, Tuple(inputs), Tuple(outputs), (),
    )
    _assert_distinct_names(d)
    ModelDescriptor(d.id, d.name, d.docstring, d.model, d.inputs, d.outputs,
        _derive_operations(d))
end

# Fail closed on a name that is both an input and an output, or that appears
# twice on one side. Reachable in practice: a data variable literally named
# `y_gen` alongside an observed `y` collides with the compiler-owned twin, and
# every downstream consumer (a form, a result target, a BridgeStan
# `param_names` lookup) keys on the name. Better a loud descriptor error than a
# UI that silently binds the wrong quantity.
_assert_distinct_names(d::ModelDescriptor) = begin
    _assert_no_repeat(x -> x.name, d.inputs, "input")
    _assert_no_repeat(x -> x.name, d.outputs, "output")
    ins = Set(i.name for i in d.inputs)
    clash = [o.name for o in d.outputs if o.name in ins]
    isempty(clash) || error(
        "Model descriptor `", d.name, "`: name(s) ", join(clash, ", "),
        " are declared as BOTH a data input and a model output. Downstream ",
        "consumers key operations, generated forms and BridgeStan draws by name, ",
        "so this cannot be resolved automatically — rename the data variable ",
        "(a compiler-owned `<obs>_gen` / `<obs>_likelihood` twin of an observed ",
        "variable is the usual source of the collision)."
    )
    d
end
_assert_no_repeat(f, xs, what) = begin
    seen = Set{Symbol}()
    for x in xs
        k = f(x)
        k in seen && error(
            "Model descriptor: duplicate ", what, " name `", k, "`. Each ", what,
            " must be uniquely addressable by name."
        )
        push!(seen, k)
    end
end

"""
    required_inputs(d::ModelDescriptor) -> Tuple{Vararg{Symbol}}

The input names a consumer must actually supply — every input that is neither
`inlined` (folded into the generated source; it never reaches the data JSON) nor
`derived` (another input's declared size, re-derived when that container is
re-bound). This is the set a generated form should collect, and it is what each
operation's `inputs` field reports.

The data block itself is wider than this: `d.inputs` also carries the derived
sizes, so a consumer that wants the full JSON payload can still see them.
"""
required_inputs(d::ModelDescriptor) =
    Tuple(i.name for i in d.inputs if !i.inlined && !i.derived)

_outputs_with(d::ModelDescriptor, role::Symbol) =
    Tuple(o.name for o in d.outputs if o.generative === role)

# A model has a likelihood iff some observation actually reaches the `model`
# block. A cv-held-out observation routes to generated quantities ONLY, so a
# model whose every observation is held out has parameters but nothing to
# condition on — it can be instantiated, but calling it a "fit" would be a lie.
_has_likelihood(d::ModelDescriptor) =
    !isempty(_observed_bases(block(d.model, :model), Set{Symbol}()))
_has_parameters(d::ModelDescriptor) = any(o -> o.kind === :parameter, d.outputs)

# --- operations --------------------------------------------------------------

_derive_operations(d::ModelDescriptor) = begin
    req = required_inputs(d)
    params = Tuple(o.name for o in d.outputs if o.kind === :parameter)
    ops = ModelOperation[
        ModelOperation(:transpile, "Show Stan code", (), (), _run_transpile),
        ModelOperation(:instantiate, "Compile", req, (), _run_instantiate),
    ]
    _has_parameters(d) && _has_likelihood(d) && push!(ops,
        ModelOperation(:fit, "Fit", req, params, _run_instantiate))
    draws = _outputs_with(d, :draw)
    isempty(draws) || push!(ops,
        ModelOperation(:predict, "Predict", req, draws, _run_generated(:draw)))
    lls = _outputs_with(d, :pointwise_loglik)
    isempty(lls) || push!(ops,
        ModelOperation(:pointwise_loglik, "Pointwise log-likelihood", req, lls,
            _run_generated(:pointwise_loglik)))
    _assert_no_repeat(op -> op.name, ops, "operation")
    Tuple(ops)
end

"""
    stan_operation(d::ModelDescriptor, name::Symbol) -> ModelOperation

The operation `name` of descriptor `d`, or a loud error naming the operations
`d` actually offers.

Fails closed by design: which operations exist is DERIVED from the model (see
[`stan_descriptor`](@ref StanBlocks.stan_descriptor)), so asking for one the
model does not support is a statement about the model — a prior-predictive model
has no `:fit`, a model with no `_gen` twin has no `:predict` — and silently
returning `nothing` would push that discovery into the consumer.
"""
stan_operation(d::ModelDescriptor, name::Symbol) = begin
    for op in d.operations
        op.name === name && return op
    end
    error(
        "Model `", d.name, "` (", d.id, ") has no `", name, "` operation. It offers: ",
        join(("`" * string(op.name) * "`" for op in d.operations), ", "),
        ". Operations are DERIVED from the declaration — see `stan_descriptor` for ",
        "the rule that gates each one."
    )
end

"""
    stan_execute(d::ModelDescriptor, name::Symbol; kwargs...)

Run operation `name` of descriptor `d`. Errors loudly if `d` does not offer it
(see [`stan_operation`](@ref StanBlocks.stan_operation)).

# Per-operation contract

- `:transpile` → `String`. Takes no keyword arguments.
- `:instantiate`, `:fit` → a `StanLogDensityProblems.StanProblem` implementing
  the `LogDensityProblems` interface. Data keyword arguments re-bind inputs
  before compiling (`stan_execute(d, :fit; y=newy)`); reserved
  [`instantiate`](@ref StanBlocks.instantiate) keywords (`path`,
  `nan_on_error`, `make_args`, `warn`) are forwarded to it. Refuses, naming
  them, if any required input is still unbound.
- `:predict`, `:pointwise_loglik` → a `NamedTuple` mapping each of the
  operation's `outputs` to its drawn values. Required keyword arguments:
  - `draws` — one unconstrained parameter vector (`Vector{Float64}`), or a
    `Matrix` whose COLUMNS are such vectors, or a vector of them.
  - `seed::Int` — seeds BridgeStan's RNG for the generated-quantities draw.
  Optional: `problem` — a `StanProblem` from a previous `:fit` / `:instantiate`,
  to skip recompilation. Any other keyword argument is forwarded to the
  compilation step exactly as for `:fit`.
  For a single draw each entry is a `Vector{Float64}` (length 1 for a scalar
  output); for several, a `Matrix` with one column per draw.
"""
stan_execute(d::ModelDescriptor, name::Symbol; kwargs...) =
    stan_operation(d, name).run(d; kwargs...)

_run_transpile(d::ModelDescriptor; kwargs...) = begin
    isempty(kwargs) || error(
        "`stan_execute(d, :transpile)` takes no keyword arguments (got ",
        join(keys(kwargs), ", "), "). Re-bind data on the model, or use ",
        "`:instantiate` / `:fit`, which accept data keywords."
    )
    stan_code(d.model)
end

# `instantiate`'s own keywords vs. data re-binding: everything not reserved is a
# data keyword, matching `(::StanModel)(; kwargs...)`.
_is_instantiate_kwarg(k::Symbol) = k in (:path, :nan_on_error, :make_args, :warn)

_run_instantiate(d::ModelDescriptor; kwargs...) = begin
    data = (; [k => v for (k, v) in pairs(kwargs) if !_is_instantiate_kwarg(k)]...)
    opts = (; [k => v for (k, v) in pairs(kwargs) if _is_instantiate_kwarg(k)]...)
    instantiate(isempty(data) ? d.model : d.model(; data...); opts...)
end

# `:predict` and `:pointwise_loglik` differ only in WHICH generated quantities
# they select, so they share one executor parameterised by the generative role.
_run_generated(role::Symbol) = (d::ModelDescriptor; draws=nothing, seed=nothing,
        problem=nothing, kwargs...) -> begin
    draws === nothing && error(
        "`stan_execute(d, :", role === :draw ? "predict" : "pointwise_loglik",
        "; draws=…, seed=…)` requires `draws`: one unconstrained parameter vector, ",
        "a matrix whose COLUMNS are such vectors, or a vector of them. Get them from ",
        "a sampler run against the `StanProblem` that `:fit` returns."
    )
    seed === nothing && error(
        "`stan_execute(d, :", role === :draw ? "predict" : "pointwise_loglik",
        "; draws=…, seed=…)` requires `seed::Int` — generated quantities are RNG ",
        "draws, and an unseeded draw would not be reproducible."
    )
    prob = problem === nothing ? _run_instantiate(d; kwargs...) : problem
    wanted = _outputs_with(d, role)
    _draw_generated(prob, wanted, _draw_columns(draws), seed)
end

_draw_columns(x::AbstractVector{<:Real}) = [collect(Float64, x)]
_draw_columns(x::AbstractMatrix{<:Real}) = [collect(Float64, view(x, :, j)) for j in axes(x, 2)]
_draw_columns(x::AbstractVector{<:AbstractVector{<:Real}}) = [collect(Float64, v) for v in x]
_draw_columns(x) = error(
    "`draws` must be a Vector{<:Real} (one unconstrained draw), a Matrix whose ",
    "columns are draws, or a vector of such vectors — got ", typeof(x), "."
)

# BridgeStan flattens a container into `name.1`, `name.2`, …; group by the stem
# so the descriptor's names address whole quantities.
_bridgestan_stem(s::AbstractString) = Symbol(first(split(s, '.')))

_draw_generated(prob, wanted, columns, seed::Integer) = begin
    sm = prob.model
    names = BridgeStan.param_names(sm; include_tp=true, include_gq=true)
    idx = OrderedDict{Symbol,Vector{Int}}()
    for (i, nm) in enumerate(names)
        push!(get!(() -> Int[], idx, _bridgestan_stem(nm)), i)
    end
    for w in wanted
        haskey(idx, w) || error(
            "Generated quantity `", w, "` is declared by the model descriptor but ",
            "absent from the compiled model's `param_names` (which has: ",
            join(sort!(collect(string.(keys(idx)))), ", "),
            "). The descriptor and the compiled artifact are out of sync — ",
            "re-reflect the model that was actually compiled."
        )
    end
    rng = BridgeStan.StanRNG(sm, seed)
    cols = [BridgeStan.param_constrain(sm, θ; include_tp=true, include_gq=true, rng=rng)
            for θ in columns]
    (; [w => _stack_draws([c[idx[w]] for c in cols]) for w in wanted]...)
end
# NB `hcat` inside this module resolves to the SLIC builtin, not `Base.hcat`
# (builtin.jl shadows a long list of Base names) — build the matrix explicitly
# rather than reaching for a name the module has redefined.
_stack_draws(vs) = length(vs) == 1 ? only(vs) :
    [vs[j][i] for i in eachindex(first(vs)), j in eachindex(vs)]

# --- display -----------------------------------------------------------------

Base.show(io::IO, d::ModelDescriptor) =
    print(io, "ModelDescriptor(", d.name, ", ", d.id, ")")
Base.show(io::IO, ::MIME"text/plain", d::ModelDescriptor) = begin
    print(io, "ModelDescriptor `", d.name, "` (", d.id, ")")
    isempty(d.docstring) || print(io, "\n  ", d.docstring)
    print(io, "\n  inputs:")
    for i in d.inputs
        print(io, "\n    ", i.name, "::", i.type)
        isempty(i.size) || print(io, "[", join(i.size, ", "), "]")
        i.observed && print(io, "  observed")
        i.held_out && print(io, "  held-out")
        i.derived && print(io, "  derived")
        i.inlined && print(io, "  inlined")
    end
    print(io, "\n  outputs:")
    for o in d.outputs
        print(io, "\n    ", o.name, "::", o.type)
        isempty(o.size) || print(io, "[", join(o.size, ", "), "]")
        print(io, "  ", o.kind, "/", o.generative)
        o.source === nothing || print(io, " of ", o.source)
    end
    print(io, "\n  operations: ", join((op.name for op in d.operations), ", "))
end
Base.show(io::IO, op::ModelOperation) = print(io, "ModelOperation(", op.name, ")")
