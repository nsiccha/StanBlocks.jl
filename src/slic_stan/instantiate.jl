"""
    stan_code(model) -> String

Return the generated Stan source for `model` (a [`SlicModel`](@ref
StanBlocks.SlicModel) or [`StanModel`](@ref StanBlocks.StanModel)) as a
plain `String`.

For a `SlicModel`, tracing runs first via [`stan_model`](@ref
StanBlocks.stan_model); for an already-traced `StanModel`, only the
rendering pass runs. Output covers the full Stan program — `data`,
`transformed_data`, `parameters`, `transformed_parameters`, `model`, and
`generated_quantities` blocks — with automatic block placement applied.

Pair with [`transpiles`](@ref StanBlocks.transpiles) for boolean smoke
tests and [`stan_instantiate`](@ref StanBlocks.stan_instantiate) to
compile the generated code via BridgeStan.
"""
function stan_code end

stan_code(x::StanModel) = begin 
    buf = IOBuffer()
    show(buf, x)
    String(take!(buf))
end
stan_code(x::SlicModel) = stan_code(stan_model(x))
prepare_for_stan(x::Dict) = Dict(key => prepare_for_stan(value) for (key, value) in x)
prepare_for_stan(x::Number) = x
prepare_for_stan(x::AbstractVector{<:Number}) = x
prepare_for_stan(x::AbstractMatrix{<:Number}) = x'
prepare_for_stan(x::NamedTuple) = prepare_for_stan(values(x))
prepare_for_stan(x::Tuple) = prepare_for_stan(Dict(enumerate(x)))
bridgestan_data(x::Dict) = JSON.json(prepare_for_stan(x))
"""
    instantiate(model; nan_on_error=true, make_args=["STAN_THREADS=true"], warn=false, path=…) -> StanProblem
    stan_instantiate(model; ...) -> StanProblem

Compile `model` (a [`SlicModel`](@ref StanBlocks.SlicModel) or
[`StanModel`](@ref StanBlocks.StanModel)) via BridgeStan and return a
`StanLogDensityProblems.StanProblem`. The returned value implements the
`LogDensityProblems` interface — call `LogDensityProblems.dimension`,
`logdensity`, and `logdensity_and_gradient` on it.

`stan_instantiate` is an exported alias for `instantiate`.

# Keyword arguments

- `path::AbstractString` — where to write the `.stan` file. Defaults to
  `"tmp/<hash>.stan"`, so identical generated code is cached on disk.
- `nan_on_error::Bool = true` — make BridgeStan return `NaN` instead of
  throwing on evaluation failures.
- `make_args::Vector{String} = ["STAN_THREADS=true"]` — extra arguments
  forwarded to Stan's `make`.
- `warn::Bool = false` — forwarded to BridgeStan.

Errors during compilation are wrapped in a [`StanBlocksError`](@ref
StanBlocks.StanBlocksError) tagged with `phase = :compile`.
"""
instantiate(x::Union{SlicModel,StanModel}; nan_on_error=true, make_args=["STAN_THREADS=true"], warn=false, kwargs...) = begin
    sc = stan_code(x)
    stan_path = get(kwargs, :path, joinpath("tmp", string(hash(sc)) * ".stan"))
    mkpath(dirname(stan_path))
    if !isfile(stan_path)
        open(stan_path, "w") do fd
            write(fd, sc)
        end
    end
    StanLogDensityProblems.StanProblem(
        stan_path,
        bridgestan_data(stan_data(x));
        nan_on_error,
        make_args,
        warn
    )
end
debug_instantiate(x; kwargs...) = instantiate(x; nan_on_error=false, kwargs...)
passinstantiate(x; kwargs...) = (instantiate(x; kwargs...); x)
stan_data(x::SlicModel) = stan_data(stan_model(x))
stan_data(x::StanModel) = Dict([
    key=>getvalue(value) for (key, value) in pairs(content(block(x, :data)))
    if !always_inline(value)
])

_record_size_kwargs!(d, key, v::AbstractVector) = (d[Symbol(key, "_n")] = length(v); nothing)
_record_size_kwargs!(d, key, v::AbstractMatrix) =
    ((d[Symbol(key, "_m")], d[Symbol(key, "_n")]) = size(v); nothing)
_record_size_kwargs!(args...) = nothing

"StanModels can update the associated data (via `new_model = model(;x=new_x)`)."
(x::StanModel)(;kwargs...) = begin
    xkwargs = Dict{Symbol,Any}(pairs(kwargs))
    for (key, value) in pairs(kwargs)
        _record_size_kwargs!(xkwargs, key, value)
    end
    StanModel(x.meta, x.vars, merge(x.blocks, (;data=StanBlock(:data,OrderedDict([
        key=>StanExpr(expr(x), remake(type(x); value=get(xkwargs, key, getvalue(x))))
        for (key, x) in pairs(block(x, :data).content)
    ])))))
end
slic_expr(x::Expr) = x
