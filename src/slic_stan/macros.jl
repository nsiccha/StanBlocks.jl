"""
Defines `SlicModel`s (see `test/slic.jl` for usage examples).

The defining module is captured automatically via `__module__`, so that `@deffun` functions
defined in the same module (e.g. a package extension) are found during symbol resolution.

A leading string literal inside the `begin ... end` block is captured as the model
docstring and rendered as a `// ...` comment header in the generated Stan code.
"""
macro slic(model)
    # `@slic f(args...) = body` — a NAMED sub-model function (see `_slic_fn`);
    # everything else is an anonymous `SlicModel` value.
    if Meta.isexpr(model, :(=)) && Meta.isexpr(model.args[1], :call)
        return _slic_fn(_inherit_source_file(model, __source__), __module__)
    end
    expanded = _inherit_source_file(
        lower_string_interp(slic_macroexpand(__module__, model)), __source__)
    doc, stripped = extract_leading_docstring(expanded)
    SlicModel(stripped, Dict{Symbol,Any}(:docstring => doc), __module__)
end
macro slic(data, model)
    mod = @__MODULE__
    expanded = _inherit_source_file(
        lower_string_interp(slic_macroexpand(__module__, model)), __source__)
    doc, stripped = extract_leading_docstring(expanded)
    qmodel = Meta.quot(stripped)
    if isempty(doc)
        esc(:($mod.SlicModel($qmodel, $data, $(__module__))))
    else
        esc(:($mod.SlicModel($qmodel, merge((;docstring=$doc), $data), $(__module__))))
    end
end

# `@slic f(args...) = body` — a NAMED sub-model function (the `@deffun`-analogue for
# models). Lowers to a proper Julia callable: binds `f = SubmodelFn{:f}()` and adds a
# call method `(::SubmodelFn{:f})(args...; kwargs...) = SlicModel(body, data, mod)` that
# binds the POSITIONAL args by name into the sub-model's data. Multiple `@slic f(...)=...`
# definitions add methods → native multiple-dispatch; other inputs still flow in by
# kwarg/scope. Contrast `@slic begin ... end`, which builds an anonymous `SlicModel`
# value. Typed args (`a::vector[n]`) drive native multiple-dispatch: the call-method
# arg gets `a::StanExpr2{<:types.vector, 1}`, so a sibling `@slic f(a::real) = …`
# defines a DISTINCT method. Untyped args match anything (positional-only).
#
# Translate one `@slic f(...)` arg spec → a method-signature fragment. Mirrors the
# `T[dims...] → StanExpr2{<:types.T, ndims}` shape of @deffun's `xsig_type` (a closure
# local to `deffun`, hence re-derived here rather than reused).
_slic_argsig(a::Symbol, fname) = a
_slic_argsig(a::Expr, fname) = begin
    (Meta.isexpr(a, :(::)) && a.args[1] isa Symbol) || error(
        "@slic ", fname, "(...): unsupported argument form `", a, "` — use `name` or `name::type`."
    )
    name, tann = a.args
    ref = Meta.isexpr(tann, :ref) ? tann : Expr(:ref, tann)   # `real` → `real[]`
    ct = ref.args[1]
    ndims = length(ref.args) - 1
    (ct isa Symbol && isdefined(types, ct)) || error(
        "@slic ", fname, "(...): unknown SLIC type `", ct, "` in argument `", a, "`."
    )
    ctval = getproperty(types, ct)
    constr = if ctval === types.anything && ndims == 0
        Expr(:curly, StanExpr2, Expr(:(<:), ctval))
    elseif ctval === types.matrix && ndims == 2
        # A `matrix[m,n]` cell argument ALSO admits Stan's constrained
        # square-matrix families (`cholesky_factor_corr`/`cholesky_factor_cov`,
        # `corr_matrix`, `cov_matrix` — the `<:square_matrix` subtypes). Those
        # carry a SINGLE declared size (`r_ndim(square_matrix)==1`, so
        # `stan_ndim==1`) even though they ARE matrices, so a plain
        # `StanExpr2{<:matrix, 2}` signature rejects them on the `ndims` type
        # parameter (value ndim 1 ≠ decl ndim 2) — the reported
        # `SubmodelFn(...) MethodError` for a top-level `cholesky_factor_corr[k]`
        # passed where `matrix[k,k]` is declared. Widen dispatch to accept both
        # shapes; the value flows into the sub-model verbatim, so its constraint
        # metadata is preserved at the caller. (No reverse admission: a plain
        # `matrix` value is NOT accepted where a constrained family is declared.)
        Expr(:curly, Union,
            Expr(:curly, StanExpr2, Expr(:(<:), ctval), ndims),
            Expr(:curly, StanExpr2, Expr(:(<:), types.square_matrix), 1))
    else
        Expr(:curly, StanExpr2, Expr(:(<:), ctval), ndims)
    end
    Expr(:(::), name, constr)
end
_slic_fn(model, mod) = begin
    call, body = model.args
    fname = call.args[1]
    fname isa Symbol || error(
        "@slic f(...) = ...: the function name must be a bare Symbol, got `", fname, "`."
    )
    argspecs = call.args[2:end]
    any(a -> Meta.isexpr(a, :parameters), argspecs) && error(
        "@slic ", fname, "(...) = ...: keyword parameters in the signature are not supported — ",
        "declare only the POSITIONAL argument names; other inputs flow in by kwarg or scope."
    )
    argnames = map(argspecs) do a
        a isa Symbol && return a
        (Meta.isexpr(a, :(::)) && a.args[1] isa Symbol) && return a.args[1]
        error("@slic ", fname, "(...): unsupported argument form `", a, "` — use `name` or `name::type`.")
    end
    expanded = lower_string_interp(slic_macroexpand(mod, body))
    doc, stripped = extract_leading_docstring(expanded)
    qbody = Meta.quot(stripped)
    data_pairs = [:($(QuoteNode(nm)) => $nm) for nm in argnames]
    ftype = Expr(:curly, SubmodelFn, QuoteNode(fname))
    sigargs = [_slic_argsig(a, fname) for a in argspecs]
    methsig = Expr(:call, Expr(:(::), ftype), Expr(:parameters, :(kwargs...)), sigargs...)
    methbody = :($SlicModel($qbody, merge(Dict{Symbol,Any}(:docstring => $doc, $(data_pairs...)), kwargs), $mod))
    esc(Expr(:block,
        Expr(:(=), fname, Expr(:call, ftype)),
        Expr(:(=), methsig, methbody),
    ))
end

"""
Utility macro to define function signatures (see `src/slic_stan/builtin.jl` for usage examples).

**Note:**

This macro is mainly useful for bulk built-in function signature definitions.
StanBlocks.jl users should generally prefer using @deffun.
"""
macro defsig(x)
    esc(defsig(x; source=__source__))
end

"""
    @deffun function_definition

Define a Stan-compatible function with type inference and dual code generation.

Parses a Julia-style function definition (with type-annotated arguments and return type),
generates the corresponding Stan function, and registers type-inference signatures so the
transpiler can propagate types through calls to this function. Eligible bodyful bare-symbol
definitions also install one callable Julia method from the original user-facing definition.
Signature-only/type-token glue, qualified or pre-existing function extensions, and
definitions whose own name is in the probability/RNG/ODE/`reduce_sum` family
(`*_lpdf`, `*_lpmf`, `*_lcdf`, `*_lccdf`, `*_cdf`, `*_rng`, the elementwise `*_lpdfs`/
`*_lpmfs`/… companions, and `ode_*`) skip the Julia target automatically.

The Julia target is a bounded deterministic compatibility layer: supported signatures,
symbolic dimension checks, typed locals, control flow/mutation, nested deterministic calls,
higher-order arguments, and varargs. In a *deterministically named* definition, a direct
probability/RNG/ODE/`reduce_sum` primitive call requires the explicit `@stanonly` opt-out
and otherwise errors at expansion time — the name says the function was meant to be
callable, so a silent skip would hide a mistake. A probability-family definition is
outside the layer by construction and needs no annotation.

For functions ending in `_lpdf`/`_lpmf`/`_lcdf`/`_lccdf`, the return type is automatically
set to `real` and companion `_lpdfs`/`_rng` stubs are generated for use in `generated_quantities`.

UDF bodies must not contain `~` sampling statements or `target +=` increments
— UDFs cannot introduce parameters or directly manipulate the log density.
The macro errors at expansion time if either is found.

Standard Julia macros inside the body are expanded against the calling module
before tracing, so `@views`, `@.`, `@inbounds`, and user-defined macros work
transparently.

# Inlining

Annotating with `@inline` (or giving the function a Julia-convention trailing `!`
in its name) causes every call to be expanded at the call site instead of
producing a Stan function:

```julia
@deffun @inline scale(x::vector[n], s::real)::vector[n] = x * s
@deffun set_first!(buf::vector[n])::vector[n] = (buf[1] = 42.; buf)
```

Inline UDFs do not appear in Stan's `functions {}` block. Multi-statement
bodies, vararg parameters, and higher-order function arguments are all
supported. Locals are renamed per call site, and pre-statements hoist into
the enclosing block.

`@inline` cannot be combined with `@lhs` / `@lpxf`.

# Example

```julia
@deffun @stanonly garch11_lpdf(y::vector[T], mu::real, alpha0::real, alpha1::real, beta1::real)::real = begin
    sigma2 = alpha0
    rv = 0.
    for t in 1:T
        rv += normal_lpdf(y[t], mu, sqrt(sigma2))
        sigma2 = alpha0 + alpha1 * square(y[t] - mu) + beta1 * sigma2
    end
    return rv
end
```

See `src/slic_stan/builtin.jl` for many more examples.
"""
macro deffun(x)
    expanded = _inherit_source_file(
        lower_string_interp(slic_macroexpand(__module__, x)), __source__)
    esc(deffun(expanded; source=__source__, def_mod=__module__))
end

"""
    @stanonly definition

Opt a bodyful `@deffun` definition out of its default Julia method while
retaining the existing SLIC/Stan lowering. Use it for deliberately Stan-only
semantics outside the bounded deterministic Julia compatibility layer:

```julia
@deffun @stanonly foo_rng(x::real)::real = stan_rng_primitive(x)
```

It may wrap one definition or a `begin ... end` group inside `@deffun`.
Signature-only and type-token compiler-glue definitions — and any definition
whose own name is in the probability/RNG/ODE/`reduce_sum` family — already skip
Julia emission automatically, so `@stanonly` is redundant on those (harmless,
but unnecessary). Reach for it when a *deterministically named* definition is
intentionally Stan-only.
"""
macro stanonly(x)
    error("@stanonly may only appear inside an @deffun block")
end

"""
    @lpxf foo_lpdf
    @lpxf begin foo_lpdf; bar_lpmf end

Register the three SLIC dispatch hooks (`lpxf_expr`, `rng_expr`, `likelihood_expr`)
for one or more user-defined log-probability functions.

The argument(s) must be bare symbols ending in `_lpdf`, `_lpmf`, `_lcdf`, or `_lccdf`.
For each `foo_lpdf` (or `_lpmf`/etc.), the macro emits the registrations:

    StanBlocks.lpxf_expr(::typeof(foo))       = foo_lpdf
    StanBlocks.rng_expr(::typeof(foo))        = foo_rng
    StanBlocks.likelihood_expr(::typeof(foo)) = foo_lpdfs

The companion `foo_rng` and `foo_lpdfs` (resp. `_lpmfs`/`_lcdfs`/`_lccdfs`) names
must already exist when the registrations execute. This macro does not parse
function bodies and does not wrap `@deffun`.
"""
macro lpxf(x)
    lpxf_register(x; source=__source__)
end

"""
    @lhs foo_lpdf(y::T, args...) = body

Inside a `@deffun` block, opt this method into base-level LHS inference. Without
`@lhs`, only the `_lpdf`-keyed tracetype is registered (so the method dispatches
when called explicitly), but `lhs ~ foo(args...)` cannot trace because the base
`foo` has no tracetype keyed on its argument signature. `@lhs` registers
`tracetype(::CanonicalExpr{<:typeof(foo), <:Tuple{lhs_type[2:end]...}})` so the
sampling form works.

Compose with `@lpxf` (any order — `@lhs @lpxf …` or `@lpxf @lhs …`) to also
register the dispatch hooks for `foo`/`foo_rng`/`foo_lpdfs`.

Standalone `@lhs` (outside `@deffun`) is not supported and errors immediately.
"""
macro lhs(x)
    error("@lhs may only appear inside a @deffun block")
end

"""
    @stan_assert cond
    @stan_assert cond message

Stan-compatible runtime assertion. Expands to `if !cond; reject(msg); end`,
where Stan's `reject` aborts the current MCMC proposal with the message.
Without an explicit message, a default `"assertion failed: <cond>"` is used.

Use inside `@deffun` bodies (control flow is not allowed in `@slic` model
bodies — wrap the check in a helper if needed at the model level).

# Example

```julia
@deffun safe_log(x::real)::real = begin
    @stan_assert x > 0 "safe_log: argument must be positive"
    return log(x)
end
```
"""
macro stan_assert(cond, msg=nothing)
    msg_expr = msg === nothing ? string("assertion failed: ", cond) : msg
    esc(:(if !($cond); reject($msg_expr); end))
end
