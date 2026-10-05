# A cooperating build-lock waiter in its own process, without loading the DSL:
# after the start gate it takes StanBlocks' build lock and prints the interval
# it held it. ARGS: build.jl, lock path, directory for ready/start files, index.
const FileWatching = Base.require(Base.PkgId(
    Base.UUID("7b1f6079-737a-58dc-b8bc-7a2ca5c1b5ee"), "FileWatching"))
using .FileWatching: Pidfile, watch_file
include(ARGS[1])
path, directory, index = ARGS[2], ARGS[3], ARGS[4]
write(joinpath(directory, "ready-$index"), "ready")
while !isfile(joinpath(directory, "start"))
    sleep(0.01)
end
_with_build_lock(path; dead_age=1, poll=0.25) do
    entered = time()
    sleep(0.5)
    println("INTERVAL ", entered, " ", time())
end
