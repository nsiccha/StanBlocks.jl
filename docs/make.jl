using Documenter
using StanBlocks

makedocs(;
    modules=[StanBlocks],
    sitename="StanBlocks.jl",
    authors="Nikolas Siccha",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://nsiccha.github.io/StanBlocks.jl",
    ),
    pages=[
        "Home" => "index.md",
        "API Reference" => "api.md",
    ],
    checkdocs=:none,
)

deploydocs(;
    repo="github.com/nsiccha/StanBlocks.jl",
    devbranch="main",
)
