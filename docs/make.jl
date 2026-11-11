using Documenter, DocumenterVitepress, StanBlocks

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
DocumenterVitepress.build_docs(joinpath(@__DIR__, "build"))
DocumenterVitepress.deploydocs(
    repo = "github.com/nsiccha/StanBlocks.jl",
    devbranch = DEVBRANCH,
    push_preview = true,
)
