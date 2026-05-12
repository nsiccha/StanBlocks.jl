"""
Defines `SlicModel`s (see `test/slic.jl` for usage examples).

The defining module is captured automatically via `__module__`, so that `@deffun` functions
defined in the same module (e.g. a package extension) are found during symbol resolution.

A leading string literal inside the `begin ... end` block is captured as the model
docstring and rendered as a `// ...` comment header in the generated Stan code.
"""
macro slic(model)
    expanded = lower_string_interp(slic_macroexpand(__module__, model))
    doc, stripped = extract_leading_docstring(expanded)
    SlicModel(stripped, Dict{Symbol,Any}(:docstring => doc), __module__)
end
macro slic(data, model)
    mod = @__MODULE__
    expanded = lower_string_interp(slic_macroexpand(__module__, model))
    doc, stripped = extract_leading_docstring(expanded)
    qmodel = Meta.quot(stripped)
    if isempty(doc)
        esc(:($mod.SlicModel($qmodel, $data, $(__module__))))
    else
        esc(:($mod.SlicModel($qmodel, merge((;docstring=$doc), $data), $(__module__))))
    end
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

Define a Stan-compatible function with type inference and code generation.

Parses a Julia-style function definition (with type-annotated arguments and return type),
generates the corresponding Stan function, and registers type-inference signatures so the
transpiler can propagate types through calls to this function.

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
@deffun garch11_lpdf(y::vector[T], mu::real, alpha0::real, alpha1::real, beta1::real)::real = begin
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
    esc(deffun(lower_string_interp(slic_macroexpand(__module__, x)); source=__source__))
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
