module DocsLint
function check(pages)
    first.(pages) == ["Start", "Guides", "Examples", "Reference"] || error("Documentation navigation must have four ordered groups")
    paths = [last(page) for group in pages for page in last(group)]
    root = joinpath(@__DIR__, "src")
    actual = [relpath(joinpath(dir,file),root) for (dir,_,files) in walkdir(root) for file in files if endswith(file,".md")]
    Set(paths) == Set(actual) || error("Every documentation page must occur in the navigation")
    banned = r"\b(seamless|robust|powerful|leverage|delve|crucial|comprehensive|unlock|simply)\b|it's worth noting"i
    for path in paths
        source = read(joinpath(root,path),String)
        for block in eachmatch(r"(?ms)^```([^\n]*)\n(.*?)^```",source)
            kind = strip(block.captures[1])
            (startswith(kind,"@") || kind in ("sh","bash","shell")) || error("Static runnable fence in $path: $kind")
        end
        prose = replace(source, r"(?ms)^```.*?^```" => "")
        occursin(banned,prose) && error("Banned prose in $path: $(match(banned,prose).match)")
        words = length(split(prose))
        startswith(path,"guides/") && words >= 1500 && error("Guide exceeds 1500 words: $path")
        if startswith(path,"examples/")
            800 <= words <= 1500 || error("Example length outside 800–1500 words: $path ($words)")
            occursin("DocsExamples.model_block",source) || error("Example has no executed model: $path")
            occursin("DocsExamples.figure",source) || error("Example has no generated figure: $path")
            occursin("DocsExamples.reproduce",source) || error("Example has no reproduction receipt: $path")
        end
        println("docs lint: ",path," ",words," prose words")
    end
end
end
