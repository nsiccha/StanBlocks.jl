"""
    stan_code(model) -> String

Return the generated Stan source for `model` (a [`SlicModel`](@ref
StanBlocks.SlicModel) or [`StanModel`](@ref StanBlocks.StanModel)) as a
plain `String`. This is the intentional source-inspection API: ordinary
terminal, Markdown, and HTML display shows a semantic model summary and never
calls this function automatically.

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

stan_code(x::StanModel) = _with_slic_read() do
    _stan_code(x)
end
_stan_code(x::StanModel) = begin
    try
        buf = IOBuffer()
        show(StanIO(buf), x)
        String(take!(buf))
    catch e
        _is_stanblocks_error(e) && rethrow()
        bt = catch_backtrace()
        lnn = get(meta(x), :_source_lnn, nothing)
        structured = _diagnostic_from_error(:slic_lowering_error, e, lnn)
        throw(StanBlocksError(:transpile, "model", (e, bt, Any[], structured)))
    end
end
stan_code(x::SlicModel) = stan_code(stan_model(x))
prepare_for_stan(x::Dict) = Dict(key => prepare_for_stan(value) for (key, value) in x)
prepare_for_stan(x::Number) = x
prepare_for_stan(x::AbstractVector{<:Number}) = x
prepare_for_stan(x::AbstractVector{T}) where {T >: Missing} = error(
    "prepare_for_stan: data vector has eltype $(eltype(x)) (contains Missing). " *
    "Partly-missing vectors are split by the SLIC tracer at tracing time; by the " *
    "time prepare_for_stan runs, the data block should contain only the observed " *
    "sub-vector (*_obs). If you see this error, the vector was not auto-detected — " *
    "check that it was passed as a keyword argument to the SlicModel."
)
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

- `path::AbstractString` — optional source alias. Relative paths resolve against
  the cwd. The file is updated atomically; identical bytes preserve its mtime,
  and changed bytes produce a warning. The native library always comes from an
  immutable artifact under `STANBLOCKS_BUILD_DIR`, or `tempdir()/stanblocks`
  when unset. It is keyed by source, ordered `make_args`, source basename,
  build root, BridgeStan version/location, CPU/compiler identities, build
  configuration, and the state of every toolchain path the build names.
  Every call resolves this identity, including identical-source calls after an
  earlier program was loaded. Published libraries are never overwritten.
- `nan_on_error::Bool = true` — make BridgeStan return `NaN` instead of
  throwing on evaluation failures.
- `make_args::Vector{String} = ["STAN_THREADS=true"]` — extra arguments
  forwarded to Stan's `make`. An argument whose value names an existing file
  or directory (`STAN=`, `MATH=`, `STANC=`, `USER_HEADER=`, …; relative to the
  BridgeStan home, where `make` runs) is identified by its resolved path and a
  stamp of every entry beneath it, so a link moved to another tree, or a tree
  rebuilt in place, selects a fresh library. Build environment variables that
  name paths count the same way, except the `PATH` and `LD_LIBRARY_PATH`
  search lists and `BRIDGESTAN`, which selects the home itself. The files a
  `USER_HEADER` reaches through quoted `#include` lines count too, as for
  [`stan_library`](@ref StanBlocks.stan_library).
- `warn::Bool = false` — forwarded to BridgeStan.

Concurrent calls own separate model instances. Artifact, source-alias, and
BridgeStan setup/build locks coordinate Julia tasks and cooperating processes;
independent tracing stays parallel. Only building the toolchain's shared files
(`stanc`, BridgeStan's bridge object, Stan Math's libraries) takes the
toolchain lock; distinct models then compile concurrently under their own
artifact locks. Native constructor calls are serialized within the process.
Compilation publishes only a completed library from a private staging
directory. Failures propagate and release the locks; partial build outputs are
discarded.
An abruptly killed process can leave a pidfile: confirm that its owner has
exited before removing the lock. A slow live compiler's lock never expires.

Keep inputs, environment, and the installed toolchain read-only during calls.
The identity covers the documented `make/local` override, compiler binaries,
and toolchain paths named by `make_args` or the build environment. The default
BridgeStan home's own trees are identified by its version and location; for
source edits there, or external compiler inputs not represented in that
configuration, use a new `STANBLOCKS_BUILD_DIR`. Direct external BridgeStan
builds must not mutate the same toolchain concurrently. This construction
contract does not grant shared ownership of the returned model's workspaces or
RNG, or make native evaluation without threading support thread-safe. Keep
BridgeStan's threading/ABI configuration consistent across libraries loaded in
one process. Stan enables threading for any nonempty `STAN_THREADS` value,
including `false`; an empty value disables it.
"""
instantiate(x::SlicModel; kwargs...) = instantiate(stan_model(x); kwargs...)
function instantiate(x::StanModel; nan_on_error=true, make_args=["STAN_THREADS=true"], warn=false, path=nothing)
    sc = stan_code(x)
    _guard_ragged_stan_version(sc)
    data = bridgestan_data(stan_data(x))
    _instantiate_artifact(sc, data; path, make_args, nan_on_error, warn)
end

"""
    stan_library(source; make_args=["STAN_THREADS=true"], stanc_args=String[], dependencies=String[]) -> String

Compile the complete Stan program text `source` into StanBlocks' immutable
artifact store and return the path of the published BridgeStan shared library.
Use it for Stan text that does not come from a SLIC model: an archived emitted
program, a hand-written `.stan` file's contents, or a program that calls
caller-owned C++ functions. Construct the model from the returned path, for
example `StanLogDensityProblems.StanProblem(library, stan_json)` or
`BridgeStan.StanModel(library, stan_json)`. [`stan_instantiate`](@ref
StanBlocks.stan_instantiate) uses the same store for SLIC models.

The library is `<build-dir>/<key>/model_<key>_model.so`, beside its source
`model_<key>.stan`, where `<build-dir>` is `\$STANBLOCKS_BUILD_DIR`, else
`tempdir()/stanblocks`. The key covers everything `stan_instantiate`'s does
(source text, ordered `make_args`, build root, BridgeStan version and
location, CPU and compiler identities, build environment, `make/local`, and
the state of every toolchain path the build names), plus the `stanc_args` and
the state of every dependency below. A change to any of them selects a new
library; a published library is never rebuilt or overwritten, so one already
loaded by a process stays valid. Identical calls return the same path without
compiling.

# Keyword arguments

- `make_args` — arguments forwarded to BridgeStan's `make`, in order. The words
  of a `STANCFLAGS=` entry are passed to `stanc` ahead of `stanc_args`.
- `stanc_args` — arguments for `stanc`, such as `"--allow-undefined"` for a
  program whose declared functions a C++ header implements. All of them reach
  `stanc`: StanBlocks hands BridgeStan one combined `STANCFLAGS` assignment,
  where `BridgeStan.compile_model` given several would keep only the last.
- `dependencies` — further files the build reads, such as a C++ header found
  through an `-I` path or a Stan file reached through `--include-paths`.
  Relative paths resolve against the current directory; each must exist.

The C++ header BridgeStan includes, named by `USER_HEADER=` in `make_args` or
in the environment, is a toolchain path like any other. make runs in the
BridgeStan home, so a relative `USER_HEADER` names a file there; pass an
absolute path. That header and every listed dependency are scanned for quoted
`#include "…"` lines, and each included file that exists relative to the
including file is a dependency too, transitively. Angle-bracket includes are
treated as part of the toolchain.

Keep the source, the dependencies, the environment, and the toolchain read-only
during the call. Compilation runs in a private staging directory under the
artifact's own build lock, so concurrent calls in this and other processes
publish each library once while distinct programs compile at the same time.
Only bringing the toolchain's shared files (`stanc`, BridgeStan's bridge
object, Stan Math's libraries) up to date takes the toolchain build lock
(`StanBlocks.with_toolchain_lock`). Unlike `stan_instantiate`,
this function constructs no native model: serializing construction and
evaluation is the caller's responsibility.
"""
function stan_library(source::AbstractString; make_args=["STAN_THREADS=true"],
                      stanc_args=String[], dependencies=String[])
    sc = String(source)
    _guard_ragged_stan_version(sc)
    files = String[abspath(path) for path in dependencies]
    for path in files
        isfile(path) || throw(ArgumentError("stan_library: dependency $(repr(path)) is not a file"))
    end
    _artifact_library(sc; make_args=String[arg for arg in make_args],
        stanc_args=String[arg for arg in stanc_args], dependencies=files)
end
"""
    _guard_ragged_stan_version(sc)

Loudly reject compiling `sc` when it uses a `*_jacobian` constraint transform
(as the ragged-constrained-parameter path does — `simplex_jacobian`,
`ordered_jacobian`, …) alongside a `functions` block, on Stan < 2.39
(BridgeStan < 2.9). On Stan 2.37 that combination *silently drops* the
jacobian adjustment: the compiled log-density is finite but omits the
constraint's change-of-variables term, yielding a WRONG posterior with no
error. Stan 2.39 fixes it. Native scalar constrained parameters declare via
the `simplex[N]` type (no `_jacobian` call), so this guard is targeted at the
ragged path and leaves non-ragged models untouched (decision 1kz3lmg).
"""
_guard_ragged_stan_version(sc) = begin
    (occursin("_jacobian(", sc) && occursin(r"functions\s*\{", sc)) || return sc
    v = pkgversion(BridgeStan)
    (v isa VersionNumber && v >= v"2.9") && return sc
    error(
        "This model uses a `*_jacobian` constraint transform (e.g. a ragged ",
        "constrained parameter like `p::simplex[Ks]`) together with a `functions` ",
        "block, which SILENTLY DROPS the jacobian adjustment on Stan < 2.39 ",
        "(BridgeStan < 2.9) — producing a finite but WRONG log-density (missing the ",
        "constraint's change-of-variables term) and hence an incorrect posterior. ",
        "Detected BridgeStan $(v isa VersionNumber ? v : "unknown"). ",
        "Upgrade BridgeStan to >= 2.9 (Stan 2.39): ",
        "`import Pkg; Pkg.add(Pkg.PackageSpec(name=\"BridgeStan\", version=\"2.9\"))`."
    )
end
debug_instantiate(x; kwargs...) = instantiate(x; nan_on_error=false, kwargs...)
stan_data(x::SlicModel) = stan_data(stan_model(x))
stan_data(x::StanModel) = Dict([
    key=>getvalue(value) for (key, value) in pairs(content(block(x, :data)))
    if !always_inline(value)
])

# Re-bind (`new_model = model(; x=new_x)`) MUST ingest a value the same way init
# does, or a value init normalizes — a table → its columns; a ragged
# vector-of-vectors → flat `mem`/`ends` — reaches `prepare_for_stan` un-normalized
# on re-bind and dies (and its derived sizes go stale). So re-bind routes every
# rebound DATA var back through the SAME `stan_type` chokepoint used at init and
# reads off the normalized value plus every derived size (`<x>_n`, a matrix's
# `<x>_m`/`<x>_n`, a table's shared `<x>_nrow`, a ragged carrier's
# `<x>_mem_n`/`<x>_ends_n`) from the resulting type — one path for every carrier,
# present and future. A kwarg that is NOT itself a data var (e.g. a partly-missing
# vector, split into other data vars at trace time) is left untouched here.
_collect_data_entry!(d, s::StanExpr) =
    (expr(s) isa Symbol && hasvalue(s) && (d[expr(s)] = getvalue(s)); nothing)
_collect_data_entry!(d, s) = nothing
_collect_derived_sizes!(d, st::StanType) = begin
    foreach(s -> _collect_data_entry!(d, s), stan_size(st))
    ats = get(info(st), :arg_types, nothing)
    ats === nothing || foreach(at -> _collect_derived_sizes!(d, at), values(ats))
    nothing
end
_rebind_data_entries!(d, key, value) = begin
    st = stan_type(key, value)
    d[key] = getvalue(st)
    _collect_derived_sizes!(d, st)
    nothing
end

"StanModels can update the associated data (via `new_model = model(;x=new_x)`)."
(x::StanModel)(;kwargs...) = begin
    xkwargs = Dict{Symbol,Any}(pairs(kwargs))
    datakeys = keys(block(x, :data).content)
    for (key, value) in pairs(kwargs)
        key in datakeys && _rebind_data_entries!(xkwargs, key, value)
    end
    StanModel(x.meta, x.vars, merge(x.blocks, (;data=StanBlock(:data,OrderedDict([
        key=>StanExpr(expr(x), remake(type(x); value=get(xkwargs, key, getvalue(x))))
        for (key, x) in pairs(block(x, :data).content)
    ])))))
end
