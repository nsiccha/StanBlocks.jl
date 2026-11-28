# Command-line test selection for `test/runtests.jl`: parse the selectors,
# decide whether one test item is selected, and say what is wrong with a
# selection that would test nothing. It loads neither TestItemRunner nor
# StanBlocks, so `test/runner_selectors.jl` tests it on synthetic items.
#
# Selectors (each may be repeated; `--x value` works as well as `--x=value`):
#
# - `--file=<substring>`: the item's path contains it (case-insensitive).
# - `--name=<substring>`: the item's name contains it (case-insensitive).
# - `--tag=<tag>`: the item has this tag (a leading `:` is accepted).
# - `--skip-tag=<tag>`: the item does not have this tag.
# - `--htmxo-test=<file>::<name>`: exactly this one item (once only).
# - `--list`: print the selected items instead of running them.
#
# Repeated `--file` or `--name` values are ALTERNATIVES: `--file=a --file=b`
# selects the items of both files. Repeated `--tag` values must ALL be present,
# so `--tag=slic --tag=stanc` narrows, and the optional suites' runners rely on
# that to add their own `--tag` to the caller's. Different selectors combine
# with AND.
#
# A selection that selects no item, or any `--file`, `--name` or `--tag` value
# that matches no item at all, is refused before any item runs (snag
# `run-several-stan-d6eb9521`: repeated `--file` values used to be ANDed, so a
# five-file regression run tested nothing and exited 0 with `None`).
module StanBlocksTestSelectors

export parse_test_args, item_matches, SelectionTally, record!, selection_problems, selector_text

function option_value(args, i, name)
    i < length(args) || error("Missing value for $name")
    return args[i + 1], i + 1
end

function parse_test_args(args)
    names = String[]
    tags = Symbol[]
    skip_tags = Symbol[]
    files = String[]
    exact = nothing
    list = false

    i = 1
    while i <= length(args)
        arg = args[i]
        if arg == "--list"
            list = true
        elseif startswith(arg, "--name=")
            push!(names, split(arg, "="; limit=2)[2])
        elseif arg == "--name"
            value, i = option_value(args, i, arg)
            push!(names, value)
        elseif startswith(arg, "--tag=")
            push!(tags, Symbol(lstrip(split(arg, "="; limit=2)[2], ':')))
        elseif arg == "--tag"
            value, i = option_value(args, i, arg)
            push!(tags, Symbol(lstrip(value, ':')))
        # `--tag` is an AND-SELECTION, so it cannot express "everything except
        # X". CI needs exactly that: most of the suite is transpile-only and
        # runs anywhere, while `:bridgestan` / `:stanc` items need a Stan
        # toolchain the runner does not have. `--skip-tag` is the complement,
        # and like `--tag` it accepts a leading colon.
        elseif startswith(arg, "--skip-tag=")
            push!(skip_tags, Symbol(lstrip(split(arg, "="; limit=2)[2], ':')))
        elseif arg == "--skip-tag"
            value, i = option_value(args, i, arg)
            push!(skip_tags, Symbol(lstrip(value, ':')))
        elseif startswith(arg, "--file=")
            push!(files, replace(split(arg, "="; limit=2)[2], '\\' => '/'))
        elseif arg == "--file"
            value, i = option_value(args, i, arg)
            push!(files, replace(value, '\\' => '/'))
        elseif startswith(arg, "--htmxo-test=")
            isnothing(exact) || error("--htmxo-test may only be specified once")
            exact = split(arg, "="; limit=2)[2]
        elseif arg == "--htmxo-test"
            isnothing(exact) || error("--htmxo-test may only be specified once")
            exact, i = option_value(args, i, arg)
        else
            error("Unknown test selector: $arg")
        end
        i += 1
    end

    exact_parts = isnothing(exact) ? nothing : split(replace(exact, '\\' => '/'), "::"; limit=2)
    isnothing(exact_parts) || length(exact_parts) == 2 || error("--htmxo-test must be file::name")
    return (; names, tags, skip_tags, files, exact_parts, list)
end

contains_ci(haystack, needle) = occursin(lowercase(needle), lowercase(haystack))

"""
    item_matches(selection, path, name, tags)

Whether the test item at `path` (relative to the package root, `/`-separated)
called `name` with `tags` is selected by `selection`, ignoring the optional
suites' environment gates (`test/runtests.jl` applies those).
"""
function item_matches(selection, path, name, tags)
    return (isempty(selection.files) || any(file -> contains_ci(path, file), selection.files)) &&
           (isempty(selection.names) || any(n -> contains_ci(name, n), selection.names)) &&
           all(tag -> tag in tags, selection.tags) &&
           !any(tag -> tag in tags, selection.skip_tags) &&
           (isnothing(selection.exact_parts) ||
            (path == selection.exact_parts[1] && name == selection.exact_parts[2]))
end

"""
    SelectionTally(selection)

Counts, over every discovered test item, how many items each `--file`,
`--name` and `--tag` value matches on its own, how many items the whole
selection selects, and how many selected items an environment gate dropped.
"""
mutable struct SelectionTally
    file_hits::Vector{Int}
    name_hits::Vector{Int}
    tag_hits::Vector{Int}
    selected::Int
    gated::Dict{String,Int}
end

SelectionTally(selection) = SelectionTally(zeros(Int, length(selection.files)),
                                           zeros(Int, length(selection.names)),
                                           zeros(Int, length(selection.tags)),
                                           0, Dict{String,Int}())

"""
    record!(tally, selection, path, name, tags; gate=nothing)

Counts one discovered test item and returns whether it is selected. `gate`
names the environment variable that excludes the item from this run, or is
`nothing` when the item may run.
"""
function record!(tally::SelectionTally, selection, path, name, tags; gate=nothing)
    for (i, file) in enumerate(selection.files)
        contains_ci(path, file) && (tally.file_hits[i] += 1)
    end
    for (i, n) in enumerate(selection.names)
        contains_ci(name, n) && (tally.name_hits[i] += 1)
    end
    for (i, tag) in enumerate(selection.tags)
        tag in tags && (tally.tag_hits[i] += 1)
    end
    item_matches(selection, path, name, tags) || return false
    if !isnothing(gate)
        tally.gated[gate] = get(tally.gated, gate, 0) + 1
        return false
    end
    tally.selected += 1
    return true
end

"""
    selector_text(selection)

The selection as command-line selectors, `--list` left out.
"""
function selector_text(selection)
    parts = String[]
    append!(parts, "--file=" .* selection.files)
    append!(parts, "--name=" .* selection.names)
    append!(parts, "--tag=" .* string.(selection.tags))
    append!(parts, "--skip-tag=" .* string.(selection.skip_tags))
    isnothing(selection.exact_parts) || push!(parts, "--htmxo-test=" * join(selection.exact_parts, "::"))
    return isempty(parts) ? "(no selectors)" : join(parts, " ")
end

"""
    selection_problems(selection, tally)

Why the selection tested by `tally` is refused, one line per reason: each
`--file`, `--name` or `--tag` value that matches no discovered item, and a
selection that selects no item. Empty when the selection is fine.
"""
function selection_problems(selection, tally::SelectionTally)
    problems = String[]
    for (file, hits) in zip(selection.files, tally.file_hits)
        hits == 0 && push!(problems, "`--file=$file` matches no test file.")
    end
    for (name, hits) in zip(selection.names, tally.name_hits)
        hits == 0 && push!(problems, "`--name=$name` matches no test item's name.")
    end
    for (tag, hits) in zip(selection.tags, tally.tag_hits)
        hits == 0 && push!(problems, "`--tag=$tag`: no test item has this tag.")
    end
    if tally.selected == 0
        line = "The selection `$(selector_text(selection))` selects no test item."
        for (gate, count) in sort!(collect(tally.gated))
            line *= " It matches $count item(s) that run only when $gate=true."
        end
        push!(problems, line)
    end
    return problems
end

end
