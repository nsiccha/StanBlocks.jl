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
        generated = length(collect(eachmatch(r"Main\.FeatureAtlasDocs\.\w*comparisons?\(", source)))
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

# ── Grey-seal IPM — vendored SlicTranspiler spotlight, assembled via a bundle ──
# The "most complete" grey-seal source is 31 `@deffun` UDF cards + two anonymous
# observation-stream submodels + one parent `@slic` body, kept as inert strings
# and assembled with `compile_slic_bundle`. Vendored verbatim in
# `grey_seal_ipm.jl`; the `seal_ipm` example fixture is inlined here.
_bundle_part(slug, title, source; kind = :submodel, data = nothing) =
    (; slug, title, kind, source = strip(source), data)
include("grey_seal_ipm.jl")   # GREY_SEAL_IPM_SOURCE + GREY_SEAL_IPM_PARTS

const GREY_SEAL_IPM_DATA = (;
    n_age = 3, n_state_years = 3, n_demo = 6, population_burn_in = 2,
    population_init = [0.08, 0.17, 0.25, 0.08, 0.17, 0.25],
    herring_index_1 = [-0.2, 0.0, 0.1, 0.2], herring_index_2 = [0.1, 0.0, -0.1, -0.2],
    hunting_quota_sweden = [40, 45, 50], hunting_quota_finland = [20, 22, 25],
    t_mate_to_preg = 0.4, t_birth_to_end_hunt = 0.75,
    ode_init_state = [0.0], ode_times = [0.75],
    obs_aerial_count = [850, 910, 980], aerial_year = [1, 2, 3],
    obs_hunting_bag_sweden = [38.0, 44.0, 48.0], hunting_bag_year_sweden = [1, 2, 3],
    obs_hunting_bag_finland = [18.0, 21.0, 24.0], hunting_bag_year_finland = [1, 2, 3],
    obs_hunting_comp_sweden = [3 4 5 2 3 1; 4 4 5 2 3 2; 4 5 5 2 3 2],
    hunting_comp_year_sweden = [1, 2, 3], hunting_comp_sample_size_sweden = [18, 20, 21],
    obs_hunting_comp_finland = [2 2 3 1 2 1; 2 3 3 1 2 1; 3 3 4 1 2 1],
    hunting_comp_year_finland = [1, 2, 3], hunting_comp_sample_size_finland = [11, 12, 14],
    obs_bycatch_comp = [1 2 3 1 2 1; 1 2 2 2 2 1; 2 2 3 2 2 1],
    bycatch_comp_year = [1, 2, 3], bycatch_comp_sample_size = [10, 10, 12],
    obs_pregnancy_count = [22, 24, 25], pregnancy_count_year = [1, 2, 3],
    pregnancy_sample_size = [30, 32, 34],
    obs_reproductive_signs_finland = [8 5 4 3; 9 5 4 4; 9 6 5 4],
    reproductive_signs_year = [1, 2, 3], reproductive_signs_sample_size = [20, 22, 24],
)

"""Assemble the vendored grey-seal IPM bundle into a traced model + Stan source."""
function grey_seal_model()
    body_expr = Meta.parse("begin\n" * GREY_SEAL_IPM_SOURCE * "\nend")
    udfs  = [p for p in GREY_SEAL_IPM_PARTS if p.kind == :udf]
    anons = [p for p in GREY_SEAL_IPM_PARTS if p.kind == :anonymous]
    udf_definitions = map(udfs) do p
        # The six observation-family DENSITY cards carry `# @slic markers:
        # stanonly, lhs, lpxf`; the rest are plain Stan-only helpers.
        markers = occursin("markers: stanonly, lhs, lpxf", p.source) ?
            (:stanonly, :lhs, :lpxf) : :stanonly
        (; definition = Meta.parse(p.source), markers)
    end
    anonymous_submodels = map(anons) do p
        ex = Meta.parse(p.source)      # `name = begin … end`
        ex.args[1] => ex.args[2]
    end
    return compile_slic_bundle(GREY_SEAL_IPM_DATA, Pair{String,Expr}[], "body" => body_expr;
        udf_definitions, anonymous_submodels, name = :grey_seal_ipm)
end

"""Render the grey-seal parent `@slic` body + named UDF cards + the generated Stan."""
function grey_seal_comparison(; show_udfs = String[])
    result = grey_seal_model()
    blocks = Any[Markdown.Code("julia", strip(GREY_SEAL_IPM_SOURCE, '\n'))]
    for slug in show_udfs
        part = only(p for p in GREY_SEAL_IPM_PARTS if p.slug == slug)
        push!(blocks, Markdown.Code("julia", part.source))
    end
    push!(blocks, Markdown.Code("stan", strip(result.code, '\n')))
    return Markdown.MD(blocks)
end

end
