# Exercise the production artifact layer in separate processes without loading
# the unrelated DSL/compiler. All inputs name this test's own worktree/env.
using SHA, BridgeStan, LogDensityProblems
using FileWatching: Pidfile, watch_file
const JSON = Base.require(Base.PkgId(Base.UUID("682c06a0-de6a-54ab-a142-c8b1cf79cde6"), "JSON"))
const StanLogDensityProblems = Base.require(Base.PkgId(
    Base.UUID("a545de4d-8dba-46db-9d34-4e41d3f07807"), "StanLogDensityProblems"))
include(ARGS[1])
directory, index = ARGS[2], parse(Int, ARGS[3])
ENV["STANBLOCKS_BUILD_DIR"] = directory
write(joinpath(directory, "ready-$index"), "ready")
while !isfile(joinpath(directory, "start"))
    sleep(0.01)
end
source = "data { real y; } parameters { real theta; } model { theta ~ normal(0, 1); y ~ normal(theta, 1); }"
problem = _instantiate_artifact(source, JSON.json((; y=Float64(index)));
    path=nothing, make_args=["STAN_THREADS=true"], nan_on_error=true, warn=false)
@assert BridgeStan.param_names(problem.model) == ["theta"]
println("RESULT ", LogDensityProblems.logdensity(problem, [0.0]))
