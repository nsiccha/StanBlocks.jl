using Documenter, DocumenterVitepress, StanBlocks

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
        devbranch = "dev",
    ),
    pages = [
        "Home"      => "index.md",
        "Gallery"   => "gallery.md",
        "API"       => "api.md",
    ],
    checkdocs = :none,
    warnonly = true,
)

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
    devbranch = "dev",
    push_preview = true,
)
