# Rich, non-transpiling display for model declarations.
#
# A display request is observation, not compilation: showing a `SlicModel` must
# never call `stan_model`/`stan_code`, and showing an already-traced `StanModel`
# must never run the Stan emitter. The explicit `stan_code(model)` function is
# the intentional model-to-Stan-source entry point.

_display_docstring(x::SlicModel) = begin
    d = data(x)
    haskey(d, :docstring) ? string(d[:docstring]) : ""
end
_display_docstring(x::StanModel) = string(get(meta(x), :docstring, ""))

_display_slic_inputs(x::SlicModel) = [
    (name=key, declaration=sprint(summary, value), semantics="bound")
    for (key, value) in pairs(data(x)) if key !== :docstring
]

_display_statement_count(x) = 1
_display_statement_count(x::Expr) = x.head === :block ?
    count(stmt -> !(stmt isa LineNumberNode), x.args) : 1

_display_source(x::SlicModel) = begin
    body = deepcopy(model(x))
    body isa Expr && Base.remove_linenums!(body)
    sprint(Base.show_unquoted, body)
end

_display_block_counts(x::StanModel) = [
    (name=key, count=length(content(value))) for (key, value) in pairs(blocks(x))
]

_display_size_text(size) = isempty(size) ? "" : string("[", join(size, ", "), "]")
_display_constraint_text(constraints) = isempty(constraints) ? "" : string(
    " (", join((string(key, " = ", value) for (key, value) in pairs(constraints)), ", "), ")"
)
_display_declaration(type, size, constraints) = string(
    type, _display_size_text(size), _display_constraint_text(constraints)
)

_display_input_semantics(input::ModelInput) = begin
    flags = String[]
    input.observed && push!(flags, "observed")
    input.held_out && push!(flags, "held-out")
    input.derived && push!(flags, "derived size")
    input.inlined && push!(flags, "inlined")
    isempty(flags) ? "data" : join(flags, ", ")
end

_display_output_semantics(output::ModelOutput) = begin
    role = string(replace(string(output.kind), "_" => " "), " / ",
        replace(string(output.generative), "_" => " "))
    output.source === nothing || (role = string(role, " of ", output.source))
    # A ragged observation's twins are group-shaped; say so, since the declared
    # size alone (a flat `num_elements`) does not.
    output.segments === nothing ? role :
        string(role, ", ", length(output.segments), " ragged groups")
end

# The semantic half of `stan_descriptor`, intentionally excluding its stable id:
# deriving that id requires `stan_code`. These helpers inspect only the already-
# traced blocks and are therefore safe for default display.
_display_traced_parts(x::StanModel) = begin
    datacontent = content(block(x, :data))
    # `_block_outputs` takes the data CONTENT, not just its names: a ragged
    # observation's twins carry the carrier's group boundaries, which live on
    # the data value (`ModelOutput.segments`).
    outputs = vcat(
        _block_outputs(x, :parameters, :parameter, datacontent),
        _block_outputs(x, :transformed_parameters, :transformed_parameter, datacontent),
        _block_outputs(x, :generated_quantities, :generated_quantity, datacontent),
    )

    observed = _observed_bases(block(x, :model), Set{Symbol}())
    for output in outputs
        output.source === nothing || push!(observed, output.source)
    end

    sizes = Set{Symbol}()
    for (_, value) in pairs(datacontent), size in stan_size(value)
        _size_symbols(expr(size), sizes)
    end

    inputs = ModelInput[]
    for (key, value) in pairs(datacontent)
        push!(inputs, ModelInput(
            key, _descriptor_type(value), _descriptor_size(value),
            _descriptor_constraints(value), getvalue(value), always_inline(value),
            key in sizes, key in observed, cv(value),
        ))
    end

    operations = Symbol[:transpile, :instantiate]
    any(output -> output.kind === :parameter, outputs) &&
        !isempty(_observed_bases(block(x, :model), Set{Symbol}())) &&
        push!(operations, :fit)
    any(output -> output.generative === :draw, outputs) && push!(operations, :predict)
    any(output -> output.generative === :pointwise_loglik, outputs) &&
        push!(operations, :pointwise_loglik)

    (; inputs, outputs, operations, blocks=_display_block_counts(x))
end

_display_model_name(x::StanModel) = begin
    name = get(meta(x), :name, nothing)
    name isa Symbol && Base.isgensym(name) ? nothing : name
end

function _display_print_indented(io::IO, text, prefix)
    lines = split(string(text), '\n'; keepempty=true)
    for (i, line) in enumerate(lines)
        print(io, prefix, line)
        i == length(lines) || print(io, '\n')
    end
end

function _display_print_docstring(io::IO, docstring)
    isempty(docstring) && return
    print(io, "\n  documentation:\n")
    _display_print_indented(io, docstring, "    ")
end

function _display_print_slic_plain(io::IO, x::SlicModel)
    inputs = _display_slic_inputs(x)
    source = _display_source(x)
    print(io, "SlicModel (untraced)\n  module: ", x.mod)
    _display_print_docstring(io, _display_docstring(x))
    print(io, "\n  bound inputs: ", length(inputs))
    for input in inputs
        print(io, "\n    ", input.name, " :: ", input.declaration)
    end
    print(io, "\n  declaration: ", _display_statement_count(model(x)), " top-level statement(s)\n")
    _display_print_indented(io, source, "    ")
    print(io, "\n  Stan source: call stan_code(model) explicitly (this will trace the declaration)")
end

function _display_print_stan_plain(io::IO, x::StanModel)
    parts = _display_traced_parts(x)
    name = _display_model_name(x)
    print(io, "StanModel")
    name === nothing || print(io, " `", name, "`")
    print(io, " (traced)")
    _display_print_docstring(io, _display_docstring(x))
    print(io, "\n  inputs: ", length(parts.inputs))
    for input in parts.inputs
        print(io, "\n    ", input.name, " :: ",
            _display_declaration(input.type, input.size, input.constraints),
            "  ", _display_input_semantics(input))
    end
    print(io, "\n  outputs: ", length(parts.outputs))
    for output in parts.outputs
        print(io, "\n    ", output.name, " :: ",
            _display_declaration(output.type, output.size, output.constraints),
            "  ", _display_output_semantics(output))
    end
    print(io, "\n  operations: ", join(parts.operations, ", "))
    print(io, "\n  blocks:")
    for item in parts.blocks
        print(io, "\n    ", replace(string(item.name), "_" => " "), ": ",
            item.count, " item(s)")
    end
    print(io, "\n  Stan source: call stan_code(model) explicitly")
end

Base.show(io::IO, x::SlicModel) = begin
    inputs = _display_slic_inputs(x)
    print(io, "SlicModel(untraced; inputs=", length(inputs), ", statements=",
        _display_statement_count(model(x)), ")")
end
Base.show(io::IO, ::MIME"text/plain", x::SlicModel) = _display_print_slic_plain(io, x)

Base.show(io::IO, x::StanModel) = begin
    parts = _display_traced_parts(x)
    name = _display_model_name(x)
    print(io, "StanModel(")
    name === nothing || print(io, name, "; ")
    print(io, "traced, inputs=", length(parts.inputs), ", outputs=",
        length(parts.outputs), ")")
end
Base.show(io::IO, ::MIME"text/plain", x::StanModel) = _display_print_stan_plain(io, x)

_display_markdown_cell(x) = replace(string(x), "\\" => "\\\\", "|" => "\\|", "\n" => "<br>")

function _display_markdown_fence(source)
    longest = 0
    current = 0
    for char in source
        if char === '`'
            current += 1
            longest = max(longest, current)
        else
            current = 0
        end
    end
    repeat("`", max(3, longest + 1))
end

function _display_print_markdown_doc(io::IO, docstring)
    isempty(docstring) && return
    print(io, "\n#### Documentation\n\n")
    for line in split(docstring, '\n'; keepempty=true)
        print(io, "> ", line, '\n')
    end
end

function _display_print_slic_markdown(io::IO, x::SlicModel)
    inputs = _display_slic_inputs(x)
    source = _display_source(x)
    fence = _display_markdown_fence(source)
    print(io, "### SlicModel\n\n`untraced` · module `", x.mod, "`\n")
    _display_print_markdown_doc(io, _display_docstring(x))
    print(io, "\n#### Bound inputs\n\n| Name | Julia binding |\n| --- | --- |\n")
    for input in inputs
        print(io, "| `", _display_markdown_cell(input.name), "` | `",
            _display_markdown_cell(input.declaration), "` |\n")
    end
    isempty(inputs) && print(io, "| — | _none_ |\n")
    print(io, "\n#### Model declaration\n\n", fence, "julia\n", source, "\n", fence,
        "\n\n_Stan source is generated only when you call `stan_code(model)` explicitly; ",
        "that call traces this declaration._\n")
end

function _display_print_stan_markdown(io::IO, x::StanModel)
    parts = _display_traced_parts(x)
    name = _display_model_name(x)
    print(io, "### StanModel")
    name === nothing || print(io, " `", _display_markdown_cell(name), "`")
    print(io, "\n\n`traced`\n")
    _display_print_markdown_doc(io, _display_docstring(x))
    print(io, "\n#### Inputs\n\n| Name | Declaration | Semantics |\n| --- | --- | --- |\n")
    for input in parts.inputs
        print(io, "| `", _display_markdown_cell(input.name), "` | `",
            _display_markdown_cell(_display_declaration(input.type, input.size, input.constraints)),
            "` | ", _display_markdown_cell(_display_input_semantics(input)), " |\n")
    end
    isempty(parts.inputs) && print(io, "| — | — | _none_ |\n")
    print(io, "\n#### Outputs\n\n| Name | Declaration | Semantics |\n| --- | --- | --- |\n")
    for output in parts.outputs
        print(io, "| `", _display_markdown_cell(output.name), "` | `",
            _display_markdown_cell(_display_declaration(output.type, output.size, output.constraints)),
            "` | ", _display_markdown_cell(_display_output_semantics(output)), " |\n")
    end
    isempty(parts.outputs) && print(io, "| — | — | _none_ |\n")
    print(io, "\n#### Operations\n\n",
        join((string("`", operation, "`") for operation in parts.operations), " · "),
        "\n\n#### Stan blocks\n\n| Block | Items |\n| --- | ---: |\n")
    for item in parts.blocks
        print(io, "| ", _display_markdown_cell(replace(string(item.name), "_" => " ")),
            " | ", item.count, " |\n")
    end
    print(io, "\n_Stan source is generated only when you call `stan_code(model)` explicitly._\n")
end

Base.show(io::IO, ::MIME"text/markdown", x::SlicModel) = _display_print_slic_markdown(io, x)
Base.show(io::IO, ::MIME"text/markdown", x::StanModel) = _display_print_stan_markdown(io, x)

_display_html_escape(x) = replace(string(x),
    "&" => "&amp;", "<" => "&lt;", ">" => "&gt;", "\"" => "&quot;", "'" => "&#39;")
_display_html_doc(x) = replace(_display_html_escape(x), "\n" => "<br>\n")

function _display_print_html_header(io::IO, kind, stage, title, subtitle)
    print(io, "<article class=\"stanblocks-model stanblocks-", kind,
        "\" data-stanblocks-kind=\"", kind, "\" data-stanblocks-stage=\"", stage, "\">",
        "<header class=\"stanblocks-model-header\"><h3>", _display_html_escape(title),
        "</h3><p><span class=\"stanblocks-stage\">", _display_html_escape(stage),
        "</span>", subtitle, "</p></header>")
end

function _display_print_html_doc(io::IO, docstring)
    isempty(docstring) && return
    print(io, "<section class=\"stanblocks-documentation\"><h4>Documentation</h4><p>",
        _display_html_doc(docstring), "</p></section>")
end

function _display_print_slic_html(io::IO, x::SlicModel)
    inputs = _display_slic_inputs(x)
    _display_print_html_header(io, "slic-model", "untraced", "SlicModel",
        string(" · module <code>", _display_html_escape(x.mod), "</code>"))
    _display_print_html_doc(io, _display_docstring(x))
    print(io, "<section class=\"stanblocks-inputs\"><h4>Bound inputs</h4>")
    if isempty(inputs)
        print(io, "<p class=\"stanblocks-empty\">None</p>")
    else
        print(io, "<table><thead><tr><th>Name</th><th>Julia binding</th></tr></thead><tbody>")
        for input in inputs
            print(io, "<tr data-stanblocks-input=\"", _display_html_escape(input.name),
                "\"><th scope=\"row\"><code>", _display_html_escape(input.name),
                "</code></th><td><code>", _display_html_escape(input.declaration),
                "</code></td></tr>")
        end
        print(io, "</tbody></table>")
    end
    print(io, "</section><section class=\"stanblocks-declaration\"><h4>Model declaration</h4>",
        "<pre><code class=\"language-julia\">", _display_html_escape(_display_source(x)),
        "</code></pre></section><footer class=\"stanblocks-model-footer\">Stan source is ",
        "generated only by an explicit <code>stan_code(model)</code> call; that call traces ",
        "this declaration.</footer></article>")
end

function _display_print_stan_html(io::IO, x::StanModel)
    parts = _display_traced_parts(x)
    name = _display_model_name(x)
    _display_print_html_header(io, "stan-model", "traced", "StanModel",
        name === nothing ? "" : string(" · <code>", _display_html_escape(name), "</code>"))
    _display_print_html_doc(io, _display_docstring(x))
    print(io, "<section class=\"stanblocks-inputs\"><h4>Inputs</h4>")
    if isempty(parts.inputs)
        print(io, "<p class=\"stanblocks-empty\">None</p>")
    else
        print(io, "<table><thead><tr><th>Name</th><th>Declaration</th><th>Semantics</th></tr></thead><tbody>")
        for input in parts.inputs
            print(io, "<tr data-stanblocks-input=\"", _display_html_escape(input.name),
                "\"><th scope=\"row\"><code>", _display_html_escape(input.name),
                "</code></th><td><code>",
                _display_html_escape(_display_declaration(input.type, input.size, input.constraints)),
                "</code></td><td>", _display_html_escape(_display_input_semantics(input)),
                "</td></tr>")
        end
        print(io, "</tbody></table>")
    end
    print(io, "</section><section class=\"stanblocks-outputs\"><h4>Outputs</h4>")
    if isempty(parts.outputs)
        print(io, "<p class=\"stanblocks-empty\">None</p>")
    else
        print(io, "<table><thead><tr><th>Name</th><th>Declaration</th><th>Semantics</th></tr></thead><tbody>")
        for output in parts.outputs
            print(io, "<tr data-stanblocks-output=\"", _display_html_escape(output.name),
                "\"><th scope=\"row\"><code>", _display_html_escape(output.name),
                "</code></th><td><code>",
                _display_html_escape(_display_declaration(output.type, output.size, output.constraints)),
                "</code></td><td>", _display_html_escape(_display_output_semantics(output)),
                "</td></tr>")
        end
        print(io, "</tbody></table>")
    end
    print(io, "</section><section class=\"stanblocks-operations\"><h4>Operations</h4><ul>")
    for operation in parts.operations
        print(io, "<li data-stanblocks-operation=\"", _display_html_escape(operation),
            "\"><code>", _display_html_escape(operation), "</code></li>")
    end
    print(io, "</ul></section><section class=\"stanblocks-blocks\"><h4>Stan blocks</h4><dl>")
    for item in parts.blocks
        print(io, "<dt>", _display_html_escape(replace(string(item.name), "_" => " ")),
            "</dt><dd>", item.count, " item(s)</dd>")
    end
    print(io, "</dl></section><footer class=\"stanblocks-model-footer\">Stan source is ",
        "generated only by an explicit <code>stan_code(model)</code> call.</footer></article>")
end

Base.show(io::IO, ::MIME"text/html", x::SlicModel) = _display_print_slic_html(io, x)
Base.show(io::IO, ::MIME"text/html", x::StanModel) = _display_print_stan_html(io, x)

# Internal Quarto hook: callers that explicitly ask for it receive the same
# semantic Markdown that notebook display uses.
quarto(x::Union{SlicModel,StanModel}) = sprint(show, MIME"text/markdown"(), x)
