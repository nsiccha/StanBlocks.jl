# The reverse-mode tape memory of the model libraries StanBlocks builds, and
# handing it back to the operating system. The C side is `tape_memory.cpp`,
# which `_artifact_library` (build.jl) links into every library.

# The library handle of a model, a problem, or the path of a library this
# process has loaded, with whether the caller must close it again.
_tape_library(model::BridgeStan.StanModel) = (model.lib, false)
_tape_library(problem::StanLogDensityProblems.StanProblem) = _tape_library(problem.model)
function _tape_library(path::AbstractString)
    # Only a library already loaded has tape memory; never load one here.
    handle = Libc.Libdl.dlopen(path, Libc.Libdl.RTLD_LAZY | Libc.Libdl.RTLD_NOLOAD; throw_error=false)
    handle === nothing ? (nothing, false) : (handle, true)
end
_tape_library(x) = throw(ArgumentError(string("StanBlocks tape memory: expected what ",
    "`stan_instantiate` returns (a `StanLogDensityProblems.StanProblem`), a ",
    "`BridgeStan.StanModel`, or the path of a loaded model library; got a ", typeof(x))))

_tape_function(lib, name) = Libc.Libdl.dlsym(lib, name; throw_error=false)

# Julia's own threads, interactive and default. `Threads.maxthreadid()` also
# counts threads adopted from C, which run no Julia tasks: a task pinned to
# one of those would never start.
_tape_threads() = 1:(Threads.nthreads(:interactive) + Threads.nthreads(:default))

# Run `f()` on Julia thread `tid`, once whatever runs there now has yielded.
function _on_thread(f, tid)
    task = Task(f)
    task.sticky = true
    ccall(:jl_set_task_tid, Cint, (Any, Cint), task, tid - 1) == 1 ||
        error("StanBlocks could not pin a task to Julia thread $tid")
    schedule(task)
end

# glibc keeps memory that was freed, such as the storage the tape's pointer
# lists outgrew, in its per-thread heaps until it is asked to trim them.
function _trim_freed_memory()
    Sys.islinux() || return
    try
        ccall(:malloc_trim, Cint, (Csize_t,), 0)
    catch err
        # Another C library (musl) has no malloc_trim; nothing then to ask.
        err isa ErrorException || rethrow()
    end
    nothing
end

"""
    StanBlocks.release_tape_memory(model) -> Union{Task,Nothing}

Hand back to the operating system the reverse-mode tape memory that Julia's
threads hold for `model`'s library, typically once a fit is over. `model` is
what [`stan_instantiate`](@ref StanBlocks.stan_instantiate) returns (a
`StanLogDensityProblems.StanProblem`), a `BridgeStan.StanModel`, or the path
of a library this process has loaded ([`stan_library`](@ref
StanBlocks.stan_library)'s result).

Stan Math records each gradient on the evaluating thread's arena and keeps
the arena's memory for as long as the thread lives, about twice the largest
gradient that thread has evaluated. Each library has its own copy of Stan
Math, so a process holds one arena per thread and library. Destroying model
instances, Julia's garbage collector and `malloc_trim` return none of it.

The release runs on every Julia thread. A thread busy with other work runs
it once that work yields, so a gradient in progress there finishes first; a
thread's tape in use is never touched. Nothing is freed: the next gradient on
a thread writes into the same memory again, and evaluates exactly as before.
Only `model`'s library is released; other libraries' arenas on the same
threads are untouched. On Linux the memory is gone from the process at once,
and glibc is then asked to return what the gradients freed (`malloc_trim`).
On macOS it leaves the process's footprint; call
[`reuse_tape_memory`](@ref StanBlocks.reuse_tape_memory) on a thread before
its next gradient on the library, so that macOS charges that memory to the
process again once it is written.

Returns a task, already running, whose result is the number of threads
`(; released, busy, unused)`: released; whose tape was in use (left as it
was); and that never evaluated a gradient with this library. `fetch` it to
wait for the release, or let it run. Returns `nothing` when the library cannot
release its tape memory: a library built by a StanBlocks without this function
or by BridgeStan directly, a path no loaded library has, and any system other
than Linux and macOS. A library built without threading (`STAN_THREADS`) has one
tape for the whole process, released once from one thread.

Arenas of threads Julia does not run, such as the TBB workers of a model's
`reduce_sum` or `map_rect`, are not reached.
"""
function release_tape_memory(model)
    (Sys.islinux() || Sys.isapple()) || return nothing
    lib, close_lib = _tape_library(model)
    lib === nothing && return nothing
    release = _tape_function(lib, :stanblocks_tape_release)
    per_thread = _tape_function(lib, :stanblocks_tape_per_thread)
    if release === nothing || per_thread === nothing
        close_lib && Libc.Libdl.dlclose(lib)
        return nothing
    end
    call() = Int(ccall(release, Cint, ()))
    tasks = if ccall(per_thread, Cint, ()) == 1
        [_on_thread(call, tid) for tid in _tape_threads()]
    else
        [Threads.@spawn call()]
    end
    Threads.@spawn begin
        try
            codes = fetch.(tasks)
            _trim_freed_memory()
            (; released=count(==(1), codes), busy=count(==(-1), codes), unused=count(==(0), codes))
        finally
            close_lib && Libc.Libdl.dlclose(lib)
        end
    end
end

"""
    StanBlocks.reuse_tape_memory(model)

Say that the calling thread's tape memory for `model`'s library, handed back
by [`release_tape_memory`](@ref StanBlocks.release_tape_memory), is in use
again. Call it on a thread before its first gradient with the library after a
release. Only macOS needs it: there, memory written again after a release is
not charged to the process until it is told, so the process would look smaller
than it is. Elsewhere it does nothing, and on macOS a call without a release
since the last one does nothing either, so a gradient wrapper may call it before
every gradient. `model` is as for `release_tape_memory`.
"""
function reuse_tape_memory(model)
    Sys.isapple() || return nothing
    lib, close_lib = _tape_library(model)
    lib === nothing && return nothing
    try
        reuse = _tape_function(lib, :stanblocks_tape_reuse)
        reuse === nothing || ccall(reuse, Cint, ())
    finally
        close_lib && Libc.Libdl.dlclose(lib)
    end
    nothing
end

# The calling thread's tape bytes for `model`'s library: reserved, or (with
# `resident=true`) how many of its arena's bytes are in memory now (Linux).
# `nothing` when the library has no such function, -1 when the thread has no
# tape there or residency cannot be measured. For tests and diagnostics.
function _tape_bytes(model; resident::Bool=false)
    lib, close_lib = _tape_library(model)
    lib === nothing && return nothing
    try
        bytes = _tape_function(lib, :stanblocks_tape_bytes)
        bytes === nothing ? nothing : Int(ccall(bytes, Clonglong, (Cint,), resident))
    finally
        close_lib && Libc.Libdl.dlclose(lib)
    end
end
