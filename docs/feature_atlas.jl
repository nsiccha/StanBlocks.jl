module FeatureAtlasDocs

using Markdown
using StanBlocks

const EXAMPLE_MODULES = Dict{Symbol,Module}()

"""Return one stable evaluation module for all executable blocks on a page."""
example_module(name::Symbol) = get!(EXAMPLE_MODULES, name) do
    Module(name)
end

function validate_template(path::AbstractString)
    source = read(path, String)
    occursin(r"(?m)^```stan\s*$", source) && error(
        "feature-atlas.md contains a hand-written Stan fence; " *
        "use FeatureAtlasDocs.comparison so the build emits it from executable source",
    )
    occursin("Main.FeatureAtlasDocs.comparison", source) ||
        error("feature-atlas.md contains no generated comparisons")
    return nothing
end

function validate_generated_templates(paths)
    for path in paths
        source = read(path, String)
        occursin(r"(?m)^```stan\s*$", source) && error(
            "$(basename(path)) contains a hand-written Stan fence; " *
            "generated examples must emit from their displayed Julia source",
        )
        occursin("Main.FeatureAtlasDocs.", source) || error(
            "$(basename(path)) contains no build-generated example",
        )
        generated = length(collect(eachmatch(r"Main\.FeatureAtlasDocs\.comparisons?\(", source)))
        wrappers = length(collect(eachmatch(r"data-atlas-comparison", source)))
        wrappers == generated || error(
            "$(basename(path)) has $(generated) generated comparison(s) but " *
            "$(wrappers) atlas UI wrapper(s)",
        )
    end
    return nothing
end

function evaluate_source(mod::Module, displayed::AbstractString)
    parsed = Meta.parseall(displayed; filename="feature-atlas-example.jl")
    expressions = parsed.head === :toplevel ? parsed.args : Any[parsed]
    for expression in expressions
        expression isa LineNumberNode && continue
        Core.eval(mod, expression)
    end
    return nothing
end

"""Evaluate and display source that defines a reusable building block."""
function source(mod::Module, code::AbstractString)
    displayed = strip(code, '\n')
    Core.eval(mod, :(using StanBlocks))
    evaluate_source(mod, displayed)
    return Markdown.MD([Markdown.Code("julia", displayed)])
end

"""Execute one displayed example and render its source plus complete Stan program."""
function comparison(mod::Module, code::AbstractString, model_name::Symbol)
    displayed = strip(code, '\n')
    Core.eval(mod, :(using StanBlocks))
    # Evaluate top-level definitions one at a time so a following `@slic`
    # observes freshly registered `@deffun` and named-submodel methods. A single
    # include-string thunk otherwise carries an older world age.
    evaluate_source(mod, displayed)
    model = Core.eval(mod, model_name)
    emitted = strip(Base.invokelatest(StanBlocks.stan_code, model), '\n')
    return Markdown.MD([
        Markdown.Code("julia", displayed),
        Markdown.Code("stan", emitted),
    ])
end

"""
Execute one displayed source block and render complete Stan programs for a named
model family. Each `models` entry is `label => expression`; the expression is
evaluated in the same page module after the displayed source.
"""
function render_comparisons(displayed::AbstractString, models)
    blocks = Any[Markdown.Code("julia", displayed)]
    for (label, model) in models
        emitted = strip(Base.invokelatest(StanBlocks.stan_code, model), '\n')
        push!(blocks, Markdown.Header([string(label)], 3))
        push!(blocks, Markdown.Code("stan", emitted))
    end
    return Markdown.MD(blocks)
end

function comparisons(mod::Module, code::AbstractString, models)
    displayed = strip(code, '\n')
    Core.eval(mod, :(using StanBlocks))
    evaluate_source(mod, displayed)
    evaluated = [string(label) => Core.eval(mod, model_expression) for
                 (label, model_expression) in models]
    return render_comparisons(displayed, evaluated)
end

function flatten_models!(models, value, prefix::AbstractString="")
    if value isa NamedTuple
        for (name, child) in pairs(value)
            label = isempty(prefix) ? string(name) : string(prefix, " / ", name)
            flatten_models!(models, child, label)
        end
    else
        push!(models, prefix => value)
    end
    return models
end

"""Like `comparisons`, deriving labels recursively from a named model family."""
function comparisons(mod::Module, code::AbstractString, model_family::Symbol)
    displayed = strip(code, '\n')
    Core.eval(mod, :(using StanBlocks))
    evaluate_source(mod, displayed)
    models = flatten_models!(Pair{String,Any}[], Core.eval(mod, model_family))
    return render_comparisons(displayed, models)
end

end
