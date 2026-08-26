using Documenter, DocumenterVitepress, StanBlocks

include("feature_atlas.jl")
FeatureAtlasDocs.validate_template(joinpath(@__DIR__, "src", "feature-atlas.md"))
FeatureAtlasDocs.validate_generated_templates(joinpath.(@__DIR__, "src", [
    "examples/golf-models.md",
    "examples/isba-2024.md",
    "examples/crowdsource.md",
    "examples/constraints.md",
    "examples/case-studies/golf.md",
    "examples/case-studies/motorcycle.md",
    "examples/case-studies/radon.md",
    "examples/case-studies/planets.md",
    "examples/case-studies/school.md",
    "examples/case-studies/species.md",
    "examples/case-studies/soil.md",
    "examples/case-studies/wastewater.md",
    "examples/case-studies/episewer.md",
    "examples/case-studies/grey-seal-ipm.md",
    "examples/monster.md",
]))

const DEVBRANCH = "devibe"
const STANCON_DIR = joinpath(dirname(@__DIR__), "presentations", "stancon-2026")
const STANCON_BUILD = joinpath(@__DIR__, "build", "stancon-2026")

# Theme files synced from HTMXObjects/assets/vitepress/ (htmxo-embed.ts,
# htmxo-gallery.css, htmxo-syntax.css) are tracked under
# docs/src/.vitepress/theme/. Re-sync them locally via
# `using HTMXObjects; HTMXObjects.vitepress_theme_install(joinpath(@__DIR__, "src", ".vitepress", "theme"))`
# when upstream changes — we don't do that on CI so the private
# HTMXObjects.jl doesn't need to be cloned during the docs build.

makedocs(
    sitename = "StanBlocks.jl",
    modules  = [StanBlocks],
    format   = DocumenterVitepress.MarkdownVitepress(
        repo = "github.com/nsiccha/StanBlocks.jl",
        devurl = "dev",
        # `devurl` is the URL PATH (`gh-pages/dev/`, which the root redirect
        # points at); `devbranch` is the git branch that deploys there. Only the
        # latter moves: `dev` had fallen 217 commits behind `devibe`, so the
        # published docs were stale by that much while the URL stayed correct.
        # Decision `0xyqj00`.
        devbranch = DEVBRANCH,
    ),
    pages = [
        "Home"              => "index.md",
        "Authoring support" => "authoring.md",
        "Activity analysis" => "activity-analysis.md",
        "Feature atlas"     => "feature-atlas.md",
        "Worked examples"   => [
            "Overview" => "worked-examples.md",
            "Model families" => [
                "Golf models" => "examples/golf-models.md",
                "PCR sensitivity versus time" => "examples/isba-2024.md",
                "Crowdsourced ratings" => "examples/crowdsource.md",
                "Reusable constraints" => "examples/constraints.md",
            ],
            "Case studies" => [
                "Golf putting" => "examples/case-studies/golf.md",
                "Motorcycle data" => "examples/case-studies/motorcycle.md",
                "Multilevel radon regression" => "examples/case-studies/radon.md",
                "Planetary motion" => "examples/case-studies/planets.md",
                "Disease transmission" => "examples/case-studies/school.md",
                "Species-site occupancy" => "examples/case-studies/species.md",
                "Soil carbon" => "examples/case-studies/soil.md",
                "Wastewater renewal model" => "examples/case-studies/wastewater.md",
                "EpiSewer composable library" => "examples/case-studies/episewer.md",
                "Grey-seal IPM" => "examples/case-studies/grey-seal-ipm.md",
                "Monster pharmacokinetics" => "examples/monster.md",
            ],
            "Design and historical material" => [
                "Original @slic overview" => "examples/slic-overview.md",
                "Simplex experiments" => "examples/simplex-transforms.md",
            ],
            "Maintained implementation references" => [
                "PosteriorDB implementations" => "examples/posteriordb-implementations.md",
            ],
        ],
        "Gallery"           => "gallery.md",
        "API"               => "api.md",
    ],
    checkdocs = :none,
    # Generated atlas comparisons must fail the build if their executable source
    # or Stan emission fails; silently publishing a stale hand-written excerpt is
    # precisely what the atlas is designed to prevent.
    warnonly = false,
)

# The presentation is a standalone Quarto site, not a VitePress page. Render it
# into its own deployment target after `makedocs` has cleaned `docs/build/`.
let quarto = Sys.which("quarto")
    isnothing(quarto) && error("Quarto is required to build the StanCon presentation")
    mkpath(STANCON_BUILD)
    run(Cmd(`$quarto render index.qmd --output-dir $STANCON_BUILD`; dir=STANCON_DIR))
    isfile(joinpath(STANCON_BUILD, "index.html")) ||
        error("Quarto did not produce $(joinpath(STANCON_BUILD, "index.html"))")
end

# Ensure a root index.html redirect exists for when no stable version is deployed
let redirect = joinpath(@__DIR__, "build", "index.html")
    isfile(redirect) || write(redirect, """
    <!DOCTYPE html>
    <html><head>
    <meta http-equiv="refresh" content="0; url=dev/">
    </head><body>Redirecting to <a href="dev/">dev</a>...</body></html>
    """)
end

DocumenterVitepress.deploydocs(
    repo = "github.com/nsiccha/StanBlocks.jl",
    devbranch = DEVBRANCH,
    push_preview = true,
)

# DocumenterVitepress deploys versioned documentation under `/dev/`, whereas
# the public presentation URL is intentionally stable and unversioned. Publish
# the independently rendered deck to `gh-pages/stancon-2026/` on devibe only.
if get(ENV, "GITHUB_REF", "") == "refs/heads/$DEVBRANCH"
    Documenter.deploydocs(
        root = @__DIR__,
        target = joinpath("build", "stancon-2026"),
        dirname = "stancon-2026",
        repo = "github.com/nsiccha/StanBlocks.jl",
        devbranch = DEVBRANCH,
        versions = nothing,
    )
end
