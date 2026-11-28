# Snag run-several-stan-d6eb9521: `test/runtests.jl` ANDed repeated `--file`
# values, so a run naming five files selected no item, printed
# `StanBlocks | None` and exited 0, which read as a pass.

@testitem "test selection: repeated --file and --name values are alternatives, --tag values all apply" tags=[:regression] begin
    include(joinpath(pkgdir(StanBlocks), "test", "selectors.jl"))
    using .StanBlocksTestSelectors

    items = [("test/a_scans.jl", "first a", [:slic, :scan]),
             ("test/a_scans.jl", "second a", [:slic]),
             ("test/b_plates.jl", "first b", [:slic, :plate]),
             ("test/c_other.jl", "first c", [:slic, :bridgestan])]
    picked(args...) = [name for (path, name, tags) in items
                       if item_matches(parse_test_args(collect(String, args)), path, name, tags)]

    @test picked() == ["first a", "second a", "first b", "first c"]
    @test picked("--file=test/a_scans.jl") == ["first a", "second a"]
    @test picked("--file=test/a_scans.jl", "--file=test/b_plates.jl") == ["first a", "second a", "first b"]
    @test picked("--file", "A_SCANS", "--file", "b_plates") == ["first a", "second a", "first b"]
    @test picked("--name=first A", "--name=first b") == ["first a", "first b"]
    # Different selectors combine with AND.
    @test picked("--file=a_scans", "--file=b_plates", "--name=first") == ["first a", "first b"]
    @test picked("--file=a_scans", "--tag=scan") == ["first a"]
    # Every `--tag` value must be present.
    @test picked("--tag=slic", "--tag=:scan") == ["first a"]
    @test picked("--tag=scan", "--tag=plate") == String[]
    @test picked("--skip-tag=bridgestan", "--skip-tag=scan") == ["second a", "first b"]
    @test picked("--htmxo-test=test/b_plates.jl::first b") == ["first b"]
end

@testitem "test selection: a selection that tests nothing is refused, naming the selectors" tags=[:regression] begin
    include(joinpath(pkgdir(StanBlocks), "test", "selectors.jl"))
    using .StanBlocksTestSelectors

    items = [("test/a_scans.jl", "first a", [:slic, :scan]),
             ("test/a_scans.jl", "second a", [:slic]),
             ("test/b_plates.jl", "first b", [:slic, :plate]),
             ("test/c_other.jl", "first c", [:slic, :gated])]
    function problems(args...; gate=tags -> :gated in tags ? "STANBLOCKS_TEST_GATED" : nothing)
        selection = parse_test_args(collect(String, args))
        tally = SelectionTally(selection)
        for (path, name, tags) in items
            record!(tally, selection, path, name, tags; gate=gate(tags))
        end
        return selection_problems(selection, tally)
    end

    @test isempty(problems())
    @test isempty(problems("--file=a_scans", "--file=b_plates"))
    @test isempty(problems("--skip-tag=no_item_has_this_tag"))
    # A misspelled value among several: the other values still select items.
    p = problems("--file=a_scans", "--file=b_plate5")
    @test p == ["`--file=b_plate5` matches no test file."]
    p = problems("--name=no such item")
    @test length(p) == 2
    @test occursin("`--name=no such item` matches no test item's name", p[1])
    @test occursin("`--name=no such item` selects no test item", p[2])
    p = problems("--tag=nonexistent", "--file=a_scans")
    @test occursin("`--tag=nonexistent`: no test item has this tag", p[1])
    @test occursin("`--file=a_scans --tag=nonexistent` selects no test item", p[2])
    # Each value matches something, the combination nothing.
    @test problems("--file=b_plates", "--tag=scan") ==
          ["The selection `--file=b_plates --tag=scan` selects no test item."]
    # Every item the selection picks is excluded by its environment gate.
    @test problems("--file=c_other") == ["The selection `--file=c_other` selects no test item. " *
                                          "It matches 1 item(s) that run only when STANBLOCKS_TEST_GATED=true."]
    @test isempty(problems("--file=c_other"; gate=_ -> nothing))
    @test problems(; gate=_ -> "STANBLOCKS_TEST_GATED") ==
          ["The selection `(no selectors)` selects no test item. " *
           "It matches 4 item(s) that run only when STANBLOCKS_TEST_GATED=true."]
end

@testitem "test selection: the runner lists two named files' items and exits nonzero, before any item runs, on a selection that tests nothing" tags=[:regression] begin
    runner = joinpath(pkgdir(StanBlocks), "test", "runtests.jl")
    project = dirname(Base.active_project())
    function run_runner(args...)
        out, err = IOBuffer(), IOBuffer()
        cmd = addenv(`$(Base.julia_cmd()) --startup-file=no --project=$project $runner $args`,
                     "GITHUB_ACTIONS" => "false")
        process = run(pipeline(ignorestatus(cmd); stdout=out, stderr=err))
        return (; ok=success(process), out=String(take!(out)), err=String(take!(err)))
    end

    listing = run_runner("--list", "--file=test/runner_selectors.jl", "--file=test/empty_data.jl")
    @test listing.ok
    listed = filter(line -> startswith(line, "test/"), split(listing.out, '\n'))
    @test any(startswith("test/runner_selectors.jl::"), listed)
    @test any(startswith("test/empty_data.jl::"), listed)
    @test all(line -> startswith(line, "test/runner_selectors.jl::") ||
                      startswith(line, "test/empty_data.jl::"), listed)

    refused = run_runner("--file=test/runner_selectors.jl", "--name=no test item is called this")
    @test !refused.ok
    @test occursin("StanBlocks test selection refused before any test item ran", refused.err)
    @test occursin("`--name=no test item is called this` matches no test item's name", refused.err)
    # Refused before the root test set exists, so nothing ran or was summarized.
    @test !occursin("Test Summary", refused.out)

    # `--list` only warns: listing nothing is an answer, not a failed run.
    empty_listing = run_runner("--list", "--name=no test item is called this")
    @test empty_listing.ok
    @test occursin("would be refused", empty_listing.err)
    @test occursin("`--name=no test item is called this` matches no test item's name", empty_listing.err)
end
