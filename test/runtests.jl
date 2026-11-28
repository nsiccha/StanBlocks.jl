using TestItemRunner
import Test
# Load the package without copying its exports into `Main`. Fresh-workspace
# integration tests must not pass through `forward!`'s last-resort Main lookup.
import StanBlocks

# The selectors, what they mean and why an empty selection is refused:
# `test/selectors.jl`.
include("selectors.jl")
using .StanBlocksTestSelectors

const TEST_SELECTION = parse_test_args(ARGS)
const TEST_TALLY = SelectionTally(TEST_SELECTION)
const SELECTION_CHECKED = Ref(false)

test_path(ti) = replace(relpath(ti.filename, pkgdir(StanBlocks)), '\\' => '/')

# The environment variable an item needs set to "true" to run, or `nothing`.
function test_gate(tags)
    # The optional integration environment installs RK's unregistered
    # dependencies explicitly; ordinary Pkg.test needs only registered deps.
    :reactivekernels in tags && get(ENV, "STANBLOCKS_TEST_REACTIVEKERNELS", "") != "true" &&
        return "STANBLOCKS_TEST_REACTIVEKERNELS"
    # Likewise Symbolics.jl, which only the `jacobian(f)` items need.
    :symbolics in tags && get(ENV, "STANBLOCKS_TEST_SYMBOLICS", "") != "true" &&
        return "STANBLOCKS_TEST_SYMBOLICS"
    return nothing
end

function selected(ti)
    path = test_path(ti)
    matches = record!(TEST_TALLY, TEST_SELECTION, path, ti.name, ti.tags; gate=test_gate(ti.tags))
    if TEST_SELECTION.list && matches
        tag_text = isempty(ti.tags) ? "" : " [" * join(string.(ti.tags), ", ") * "]"
        println(path, "::", ti.name, tag_text)
        return false
    end
    return matches
end

# Refuse a selection that tests nothing, or names a file, item name or tag no
# item has. `--list` only warns: an empty listing is an honest answer.
function check_selection()
    SELECTION_CHECKED[] && return
    SELECTION_CHECKED[] = true
    problems = selection_problems(TEST_SELECTION, TEST_TALLY)
    isempty(problems) && return
    listing = join("  " .* problems, "\n")
    TEST_SELECTION.list && return @warn "A run of this StanBlocks test selection would be refused:\n" * listing
    error("StanBlocks test selection refused before any test item ran:\n" * listing *
          "\nRun with --list to see which items a selection picks.")
end

# `@run_package_tests` calls the filter on every discovered item, then builds
# the root test set with this constructor, and only then runs items. Checking
# on that first call refuses a bad selection in seconds, not after a run.
function checked_testset(name; kwargs...)
    check_selection()
    return Test.DefaultTestSet(name; kwargs...)
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
    @run_package_tests filter=selected verbose=true testset=checked_testset
    # In case a TestItemRunner version builds no test set before running.
    check_selection()
catch err
    annotate_github_failure(err)
    rethrow()
end
