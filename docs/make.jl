using Documenter, DocumenterVitepress, StanBlocks, BridgeStan
using NodeJS_20_jll: node, npm

include("examples.jl")
include("lint.jl")

const DEVBRANCH = "devibe"
const PAGES = [
    "Start" => ["StanBlocks.jl" => "index.md"],
    "Guides" => [
        "Models" => "guides/models.md",
        "Composition" => "guides/composition.md",
        "Arrays and loops" => "guides/arrays.md",
        "Functions and densities" => "guides/functions.md",
        "Prediction" => "guides/prediction.md",
        "Execution" => "guides/execution.md",
    ],
    "Examples" => [
        "Golf putting" => "examples/golf.md",
        "Partial pooling" => "examples/pooling.md",
        "Epidemic dynamics" => "examples/epidemic.md",
    ],
    "Reference" => [
        "Model APIs" => "reference/models.md",
        "Execution APIs" => "reference/execution.md",
    ],
]
DocsLint.check(PAGES)

# The model-source panels run stanc before an example compiles a model, so
# BridgeStan's lazy compiler setup has not run yet on a fresh machine.
let path = BridgeStan.get_bridgestan_path(; download = true)
    StanBlocks.with_toolchain_lock(path) do
        target = Sys.iswindows() ? "bin/stanc.exe" : "bin/stanc"
        stanc = joinpath(path, target)
        isfile(stanc) || run(`make -C $path $target`)
        isfile(stanc) || error("BridgeStan setup has no stanc binary at $stanc")
    end
end

makedocs(
    sitename = "StanBlocks.jl",
    modules = [StanBlocks],
    format = DocumenterVitepress.MarkdownVitepress(
        repo = "github.com/nsiccha/StanBlocks.jl",
        devurl = "dev",
        devbranch = DEVBRANCH,
        build_vitepress = false,
    ),
    pages = PAGES,
    checkdocs = :exports,
    warnonly = false,
)

# Figures and their runtime are generated/copied only into the build tree.
DocsExamples.publish_assets(joinpath(@__DIR__, "build", ".documenter", "public"))
# VitePress builds here, after the figures are in place, not inside makedocs. Only
# makedocs' own build installs the npm packages, so install them for a clean checkout.
cd(@__DIR__) do
    node(; adjust_PATH = true, adjust_LIBPATH = true) do _
        run(`$(npm) install --no-audit --no-fund`)
    end
end
DocumenterVitepress.build_docs(joinpath(@__DIR__, "build"))
DocumenterVitepress.deploydocs(
    repo = "github.com/nsiccha/StanBlocks.jl",
    devbranch = DEVBRANCH,
    push_preview = true,
)
