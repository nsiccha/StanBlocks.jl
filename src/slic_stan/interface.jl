# --- The interface: a traced program described for consumers outside Julia ---
#
# BridgeStan runs the emitted `.stan` file without Julia (its Python package,
# or nutpie on top of it); it is the runtime StanBlocks supports, CmdStan is not
# (user, decision `0i2b1wr`). Such a consumer needs the data as a Stan JSON data
# file and reads draws under the emitted names, so it otherwise re-implements
# StanBlocks' naming: the derived sizes of a container (`y_n`, a ragged input's
# `y_mem_n`/`y_ends_n`, a table's `df_nrow`), the tuple encoding of a ragged
# input, and the group layout of a ragged output. `stan_interface` publishes
# those facts as plain JSON data, projected from the descriptor (todo `0gr4e02`,
# user decision `0soyu1b`, prong `interface`); `stan_data_json` writes the exact
# data file a program receives. Nothing here traces or emits.

const _INTERFACE_FORMAT = "stanblocks-interface/1"
const _INTERFACE_BLOCKS = Dict(
    :parameter => "parameters",
    :transformed_parameter => "transformed_parameters",
    :generated_quantity => "generated_quantities",
)

"""
    stan_interface(model) -> OrderedDict{String,Any}

Describe a traced program for a consumer that runs its Stan code outside Julia,
through BridgeStan (from Python, for example, with nutpie for sampling). `model`
is a [`SlicModel`](@ref StanBlocks.SlicModel), a [`StanModel`](@ref
StanBlocks.StanModel) or a [`ModelDescriptor`](@ref StanBlocks.ModelDescriptor).
The result holds only strings, numbers, booleans, `nothing`, vectors and
dictionaries, so `JSON.json(stan_interface(m))` is the interface file; it is
regenerated with the program and pinned to it by `program_id`. CmdStan is not a
supported runtime; see "Running a program outside Julia" in the authoring guide.

Top level:

- `"format"` — `"stanblocks-interface/1"`.
- `"program_id"` — the descriptor `id`, a hash of the emitted Stan source.
- `"name"` — the model name.
- `"data"` — one entry per data-block variable, in data-block order (below).
- `"parameters"` — the parameter variables in the order BridgeStan's
  `param_unc_names` / `param_names` list their coordinates.
- `"outputs"` — one entry per parameter, transformed parameter and generated
  quantity (below).

A `"data"` entry:

- `"name"`, `"type"` (the descriptor's center type: `"real"`, `"int"`,
  `"vector"`, `"matrix"`, `"RaggedVector"`, `"ntup"`, …) and `"size"` (each
  declared size as Stan source text).
- `"supplied"` — `true` for a value the caller provides, `false` for one
  computed from another; `"computed"` then names the rule
  `{"rule": "size", "of": <input>, "component": [<field names>], "axis": <k>}`:
  the size along axis `k` (1-based) of that input, or of its named component.
- `"encoding"` — how the caller's value becomes JSON:
  - `"value"` — as is; a matrix is a list of rows.
  - `"ragged"` — a list of groups. The JSON is a Stan tuple: key `"1"` the
    groups' values concatenated in order, key `"2"` their inclusive 1-based
    ends. `"ragged"` names the element type (`"real"` or `"int"`) and the
    component keys.
  - `"tuple"` — a record. The JSON is an object keyed by component position,
    `"1"`, `"2"`, …; `"fields"` lists the components in that order.
- `"fields"` — the components of a tuple-typed input (`[]` otherwise), each
  with `"key"`, `"name"`, `"type"`, `"size"`, `"encoding"` and its own
  `"fields"`.
- `"observed"`, `"held_out"` and `"constraints"` (each declared bound's value).

An `"outputs"` entry: `"name"` (the Stan name; a container's coordinates are
`name.i.j…`, first index fastest), `"block"`, `"type"`, `"size"` (Stan text),
`"dims"` (resolved for the bound data), `"generative"` and `"source"` as for
[`ModelOutput`](@ref StanBlocks.ModelOutput), `"segments"` (a ragged output's
inclusive group ends for the bound data, else `nothing`), and
`"segments_rule"` — where those ends come from, so a consumer can rebuild them
for other data:

- `{"rule": "groups_of", "of": <input>}` — the ends of a ragged data input.
- `{"rule": "lengths", "of": <input>}` — group `g` has `<input>[g]` values, so
  the ends are their cumulative sums.
- `{"rule": "expression", "cell_size": <Stan text>, "index": <name>}` — group
  `index` has `cell_size` values.
- `nothing` for a dense output.

A tuple-typed output also lists its components in `"fields"`.

The JSON a program receives is [`stan_data_json`](@ref
StanBlocks.stan_data_json); a consumer that encodes its own values from this
interface produces the same file.
"""
function stan_interface end
stan_interface(x::Union{SlicModel,StanModel}; kwargs...) = stan_interface(stan_descriptor(x; kwargs...))
stan_interface(d::ModelDescriptor) = begin
    inputs = [i for i in d.inputs if !i.inlined]
    rules = _interface_size_rules(inputs)
    ragged = Set{Symbol}(i.name for i in inputs if i.type === :RaggedVector)
    ints = Set{Symbol}(i.name for i in inputs if i.type === :int && length(i.size) == 1)
    OrderedDict{String,Any}(
        "format" => _INTERFACE_FORMAT,
        "program_id" => d.id,
        "name" => string(d.name),
        "data" => Any[_interface_input(i, rules) for i in inputs],
        "parameters" => String[string(o.name) for o in d.outputs if o.kind === :parameter],
        "outputs" => Any[_interface_output(d, o, ragged, ints) for o in d.outputs],
    )
end

"""
    stan_data_json(model) -> String

The data `model` binds, as the Stan JSON data file its program reads — the same
bytes [`instantiate`](@ref StanBlocks.instantiate) hands BridgeStan. Matrices
are written as lists of rows and Stan tuples (a ragged input, a table) as
objects keyed by component position. `model` is a
[`SlicModel`](@ref StanBlocks.SlicModel), a [`StanModel`](@ref
StanBlocks.StanModel) or a [`ModelDescriptor`](@ref StanBlocks.ModelDescriptor);
write it with `write(path, stan_data_json(model))` and pass the path to
BridgeStan's `StanModel`.
[`stan_interface`](@ref StanBlocks.stan_interface) describes how each entry is
encoded.
"""
function stan_data_json end
stan_data_json(x::Union{SlicModel,StanModel}) = bridgestan_data(stan_data(x))
stan_data_json(d::ModelDescriptor) = stan_data_json(d.model)

# A size or bound as the Stan source the program spells it with. A name or a
# literal is its own spelling (the Stan printer shows a bare `Symbol` quoted).
_interface_text(x) = sprint(io -> show(StanIO(io), x))
_interface_text(x::Union{Symbol,Integer}) = string(x)

# Every derived input is a declared size of a supplied one (`ModelInput.derived`):
# record, for each such size, which input and component declares it and along
# which axis. The first declaration wins when several share one (a table's
# columns all read `df_nrow`).
_interface_size_rules(inputs) = begin
    rules = Dict{Symbol,Any}()
    for i in inputs
        i.derived || _interface_size_rules!(rules, i.name, String[], i.size, i.fields)
    end
    for i in inputs
        i.derived && !haskey(rules, i.name) && error(
            "stan_interface: `", i.name, "` is a derived input, but no supplied input ",
            "declares it as a size.")
    end
    rules
end
_interface_size_rules!(rules, of::Symbol, path, size, fields) = begin
    for (axis, s) in enumerate(size)
        (s isa Symbol && !haskey(rules, s)) || continue
        rules[s] = OrderedDict{String,Any}(
            "rule" => "size", "of" => string(of), "component" => copy(path), "axis" => axis)
    end
    for (key, f) in pairs(fields)
        _interface_size_rules!(rules, of, [path; string(key)], f.size, f.fields)
    end
end

_interface_encoding(type::Symbol, fields) =
    type === :RaggedVector ? "ragged" : isempty(fields) ? "value" : "tuple"

# The components of a tuple-typed entry, in Stan tuple order. A ragged component
# (a `NamedTuple` field holding groups) carries `"ragged"` like a ragged input.
_interface_fields(fields) = Any[
    _with_ragged!(OrderedDict{String,Any}(
        "key" => string(k), "name" => string(name), "type" => string(f.type),
        "size" => String[_interface_text(s) for s in f.size],
        "encoding" => _interface_encoding(f.type, f.fields),
        "fields" => _interface_fields(f.fields),
    ), f.type, f.fields)
    for (k, (name, f)) in enumerate(pairs(fields))
]
_with_ragged!(entry, type::Symbol, fields) =
    (type === :RaggedVector && (entry["ragged"] = _interface_ragged(fields)); entry)

# A ragged input's element type and component keys: its `mem` row is `vector`
# for real values, `int` for integers (`ModelInput.fields`).
_interface_ragged(fields) = OrderedDict{String,Any}(
    "element" => fields.mem.type === :int ? "int" : "real",
    "values" => string(findfirst(==(:mem), keys(fields))),
    "ends" => string(findfirst(==(:ends), keys(fields))),
)

_interface_input(i::ModelInput, rules) = _with_ragged!(OrderedDict{String,Any}(
        "name" => string(i.name),
        "type" => string(i.type),
        "size" => String[_interface_text(s) for s in i.size],
        "encoding" => _interface_encoding(i.type, i.fields),
        "supplied" => !i.derived,
        "computed" => i.derived ? rules[i.name] : nothing,
        "observed" => i.observed,
        "held_out" => i.held_out,
        "constraints" => OrderedDict{String,Any}(string(k) => v for (k, v) in pairs(i.constraint_values)),
        "fields" => _interface_fields(i.fields),
    ), i.type, i.fields)

_interface_output(d::ModelDescriptor, o::ModelOutput, ragged, ints) = OrderedDict{String,Any}(
    "name" => string(o.name),
    "block" => _INTERFACE_BLOCKS[o.kind],
    "type" => string(o.type),
    "size" => String[_interface_text(s) for s in o.size],
    "dims" => collect(Int, o.dims),
    "generative" => string(o.generative),
    "source" => o.source === nothing ? nothing : string(o.source),
    "segments" => o.segments,
    "segments_rule" => _interface_segments_rule(d, o, ragged, ints),
    "fields" => Any[
        OrderedDict{String,Any}(
            "key" => string(k), "name" => string(name), "type" => string(f.type),
            "size" => String[_interface_text(s) for s in f.size], "dims" => collect(Int, f.dims),
        )
        for (k, (name, f)) in enumerate(pairs(o.fields))
    ],
)

# Where a ragged output's group ends come from. An observation's twin has the
# observed input's groups. A ragged plate value records, on its flat-memory
# declaration, the length of each cell as a data expression over the plate
# index (`ragged_plate_layout`); two shapes of it name an input directly.
_interface_segments_rule(d, o, ragged, ints) = begin
    o.segments === nothing && return nothing
    o.source !== nothing && o.source in ragged && return _groups_of(o.source)
    x = get(vars(d.model), o.name, nothing)
    layout = x === nothing ? nothing : get(info(type(x)), :ragged_plate_layout, nothing)
    layout === nothing && return nothing
    rule = _cell_size_rule(layout.cell_size, layout.index, ragged, ints)
    rule !== nothing && return rule
    OrderedDict{String,Any}(
        "rule" => "expression", "cell_size" => _interface_text(layout.cell_size),
        "index" => string(layout.index))
end
_groups_of(name) = OrderedDict{String,Any}("rule" => "groups_of", "of" => string(name))
_lengths_of(name) = OrderedDict{String,Any}("rule" => "lengths", "of" => string(name))

# `1 + (ragged_end(y, i) - ragged_start(y, i))`, the length of group `i` of the
# ragged input `y` (spelled `y.2` when a do-form positional slices it), and
# `K[i]` for an integer array input `K`. Any other cell size has no rule here.
_cell_size_rule(x, index, ragged, ints) = nothing
_cell_size_rule(x::StanExpr, index, ragged, ints) = _cell_size_rule(expr(x), index, ragged, ints)
_cell_size_rule(x::CanonicalExpr{typeof(+)}, index, ragged, ints) = begin
    (length(x.args) == 2 && _literal_one(x.args[1])) || return nothing
    diff = expr(x.args[2])
    (diff isa CanonicalExpr{typeof(-)} && length(diff.args) == 2) || return nothing
    stop, start = expr(diff.args[1]), expr(diff.args[2])
    (stop isa CanonicalExpr{typeof(builtin.ragged_end)} &&
        start isa CanonicalExpr{typeof(builtin.ragged_start)}) || return nothing
    a, b = _ragged_group_ref(stop, index), _ragged_group_ref(start, index)
    (a !== nothing && a === b && a in ragged) ? _groups_of(a) : nothing
end
_cell_size_rule(x::CanonicalExpr{typeof(getindex)}, index, ragged, ints) = begin
    length(x.args) == 2 || return nothing
    base, at = expr(x.args[1]), expr(x.args[2])
    (base isa Symbol && base in ints && at === index) ? _lengths_of(base) : nothing
end
_literal_one(x) = expr(x) == 1

# The ragged input a `ragged_end(…, index)` / `ragged_start(…, index)` call reads.
_ragged_group_ref(call, index) = begin
    (length(call.args) == 2 && expr(call.args[2]) === index) || return nothing
    _ragged_input_name(expr(call.args[1]))
end
_ragged_input_name(x) = nothing
_ragged_input_name(x::Symbol) = x
_ragged_input_name(x::CanonicalExpr{typeof(getfield)}) =
    (length(x.args) == 2 && expr(x.args[2]) == 2) ? _ragged_input_name(expr(x.args[1])) : nothing
