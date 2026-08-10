module FeatureAtlasDocs

using Markdown
using StanBlocks

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

function evaluate_source(mod::Module, displayed::AbstractString)
    parsed = Meta.parseall(displayed; filename="feature-atlas-example.jl")
    expressions = parsed.head === :toplevel ? parsed.args : Any[parsed]
    for expression in expressions
        expression isa LineNumberNode && continue
        Core.eval(mod, expression)
    end
    return nothing
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

end
