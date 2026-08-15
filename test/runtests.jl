using TestItemRunner
# Load the package without copying its exports into `Main`. Fresh-workspace
# integration tests must not pass through `forward!`'s last-resort Main lookup.
import StanBlocks

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

const TEST_SELECTION = parse_test_args(ARGS)

test_path(ti) = replace(relpath(ti.filename, pkgdir(StanBlocks)), '\\' => '/')

function selected(ti)
    path = test_path(ti)
    selection = TEST_SELECTION
    matches = all(name -> occursin(lowercase(name), lowercase(ti.name)), selection.names) &&
              all(tag -> tag in ti.tags, selection.tags) &&
              !any(tag -> tag in ti.tags, selection.skip_tags) &&
              all(file -> occursin(lowercase(file), lowercase(path)), selection.files) &&
              (isnothing(selection.exact_parts) ||
               (path == selection.exact_parts[1] && ti.name == selection.exact_parts[2]))

    if selection.list && matches
        tag_text = isempty(ti.tags) ? "" : " [" * join(string.(ti.tags), ", ") * "]"
        println(path, "::", ti.name, tag_text)
        return false
    end
    return matches
end

function annotate_github_failure(err)
    get(ENV, "GITHUB_ACTIONS", "") == "true" || return

    # The public Actions API exposes annotations even when GitHub withholds the
    # raw job log. A TestSetException's normal display contains only aggregate
    # counts, while `errors_and_fails` carries the actionable nested exception
    # and backtrace. Preserve a few of those in one queryable annotation;
    # workflow-command messages require percent/newline escaping. Keep below
    # GitHub's annotation payload ceiling.
    report = sprint() do io
        showerror(io, err)
        hasproperty(err, :errors_and_fails) || return
        for (i, failure) in enumerate(Iterators.take(err.errors_and_fails, 3))
            println(io, "\n\nFailure $i:")
            show(io, failure)
        end
    end
    message = first(report, min(length(report), 60_000))
    escaped = replace(message, '%' => "%25", '\r' => "%0D", '\n' => "%0A")
    println("::error title=StanBlocks test failure::", escaped)
end

try
    @run_package_tests filter=selected verbose=true
catch err
    annotate_github_failure(err)
    rethrow()
end
