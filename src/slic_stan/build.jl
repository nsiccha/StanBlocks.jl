# File locks cover both Julia Tasks and cooperating processes, using Julia's
# Pidfile protocol so external builders can share it. Do not expire a live
# compiler's lock on a timeout: a slow build must never acquire a competitor.
# A holder killed before release (SIGKILL, OOM) cannot remove its file, so a
# waiter reclaims a lock only when its holder is gone from this host and
# stopped refreshing it (`_holder_alive`). Pidfile's own `stale_age` removal is
# check-then-remove: concurrent waiters delete each other's fresh locks and all
# enter. Reclaim therefore re-checks the file under a separate short guard.
# `dead_age`: seconds without an mtime refresh (holders refresh every sixth of
# it) before a reclaim; `poll`: seconds between a waiter's checks.
function _with_build_lock(f, path; dead_age=60, poll=10)
    mkpath(dirname(path))
    lock = _acquire_build_lock(path, dead_age, poll)
    try
        f()
    finally
        close(lock)
    end
end

function _acquire_build_lock(path, dead_age, poll)
    while true
        lock = Pidfile.trymkpidlock(path; refresh=dead_age / 6)
        lock === false || return lock
        _reclaim_dead_build_lock(path, dead_age) || _wait_build_lock_change(path, poll)
    end
end

function _wait_build_lock_change(path, poll)
    try
        watch_file(path, poll)
    catch err
        err isa Base.IOError || rethrow()
        # A release between the attempt and the watch: retry immediately. Any
        # other watch failure (no inotify watch left) falls back to polling.
        ispath(path) && sleep(poll)
    end
    nothing
end

_process_exists(pid) = ccall(:uv_kill, Cint, (Cint, Cint), pid, 0) != Base.UV_ESRCH

# Whether a lock's holder may still be alive. Every holder, stdlib Pidfile and
# older StanBlocks included, keeps its lock file open for writing until it
# releases. On Linux the kernel's open count therefore decides, whatever pid
# namespace the holder or this waiter runs in: a sandboxed holder records its
# namespace-local pid, which names an unrelated live process here (pid 2 is
# `kthreadd` on a host), and a host holder's pid is invisible from a sandbox.
# Elsewhere, or where the filesystem grants no leases, the recorded pid decides.
function _holder_alive(path, pid)
    writer = _open_for_write_elsewhere(path)
    writer === nothing ? _process_exists(pid) : writer
end

# Linux fcntl constants, identical on every architecture Julia supports.
const _F_SETSIG = Cint(10)
const _F_SETLEASE = Cint(1024)
const _F_RDLCK = Cint(0)
const _F_UNLCK = Cint(2)
const _SIGURG = Cint(23)

# Linux: whether any file description has `path` open for writing, or `nothing`
# when that cannot be measured. The kernel grants a read lease only on a file
# nobody has open for writing; it is released at once. Readers do not count, so
# a waiter reading the record never looks like a holder. A lease break signals
# the lease holder, by default with SIGIO, which terminates Julia: SIGURG is
# ignored by default, so a writer racing this probe only waits for its release.
function _open_for_write_elsewhere(path)
    Sys.islinux() || return nothing
    file = try
        Base.Filesystem.open(path, Base.Filesystem.JL_O_RDONLY | Base.Filesystem.JL_O_NONBLOCK)
    catch err
        err isa Base.IOError || rethrow()
        return nothing
    end
    try
        ccall(:fcntl, Cint, (Cint, Cint, Cint...), file.handle, _F_SETSIG, _SIGURG) == 0 || return nothing
        if ccall(:fcntl, Cint, (Cint, Cint, Cint...), file.handle, _F_SETLEASE, _F_RDLCK) != 0
            return Libc.errno() == Libc.EAGAIN ? true : nothing
        end
        ccall(:fcntl, Cint, (Cint, Cint, Cint...), file.handle, _F_SETLEASE, _F_UNLCK)
        false
    finally
        close(file)
    end
end

# The record of a lock whose holder died on this host, or `nothing`. A remote
# host, a live holder, or a foreign format leaves the lock to its holder.
function _dead_build_lock_record(path, dead_age)
    # Read through libuv, as stdlib Pidfile does: on Windows it opens with
    # delete sharing, so this read cannot block a reclaimer's rename. An
    # IOStream `open` there denies it, failing that rename with EBUSY.
    record, age = try
        file = Base.Filesystem.open(path, Base.Filesystem.JL_O_RDONLY)
        try
            (read(file, String), time() - mtime(file))
        finally
            close(file)
        end
    catch err
        err isa Base.IOError && err.code == Base.UV_ENOENT && return nothing
        rethrow()
    end
    age > dead_age || return nothing
    # Pidfile writes "<pid> <host>" right after creating the file; an empty
    # file's holder died in between unless it still has the file open.
    isempty(record) && return _open_for_write_elsewhere(path) === true ? nothing : record
    m = match(r"^([0-9]+) (.+)$"s, record)
    m === nothing && return nothing
    pid = tryparse(Cint, m[1])
    (pid === nothing || pid == 0 || m[2] != gethostname() || _holder_alive(path, pid)) && return nothing
    record
end

function _reclaim_dead_build_lock(path, dead_age)
    _dead_build_lock_record(path, dead_age) === nothing && return false
    # Another waiter may have reclaimed the file meanwhile and a live process
    # re-created it, so decide again while holding the guard. The guard is held
    # only for that check; Pidfile clears one left by a dead reclaimer.
    guard = Pidfile.trymkpidlock(path * ".reclaim"; stale_age=dead_age)
    guard === false && return false
    try
        record = _dead_build_lock_record(path, dead_age)
        record === nothing && return false
        _remove_build_lock(path) || return false
        @warn "StanBlocks reclaimed a build lock whose holder no longer exists" path record
        true
    finally
        close(guard)
    end
end

# Whether the lock file is gone. Only a non-cooperating process can remove it
# under the guard; its absence is then already the outcome a reclaim needs.
function _remove_build_lock(path)
    # Windows reserves a deleted name while any handle is open; move it aside.
    if Sys.iswindows()
        aside = string(path, '.', getpid(), '.', time_ns(), ".deleted")
        try
            _retry_windows_rename(path, aside, Base.UV_EBUSY)
        catch err
            err isa Base.IOError || rethrow()
            err.code == Base.UV_ENOENT && return true
            # A reader without delete sharing (an older StanBlocks, a virus
            # scanner) still has the dead file open: leave it for a later round.
            err.code == Base.UV_EBUSY && return false
            rethrow()
        end
        path = aside
    end
    rm(path; force=true)
    true
end

"""
    StanBlocks.with_toolchain_lock(f, toolchain)

Run `f()` holding the build lock StanBlocks holds while compiling in the
BridgeStan source/build tree `toolchain` (`.stanblocks-build.pid` in its
`realpath`). External builders sharing that tree, such as a replay of emitted
Stan source through `BridgeStan.compile_model`, use it to exclude StanBlocks'
builds and each other. The lock is never expired while its holder runs; a lock
left by a process that died on this host is reclaimed.
"""
with_toolchain_lock(f, toolchain) = _with_build_lock(f, _toolchain_lock_path(realpath(toolchain)))
_toolchain_lock_path(home) = joinpath(home, ".stanblocks-build.pid")

function _bridgestan_build_home()
    # BridgeStan may install its default source distribution on first use. The
    # installer lock has a stable location even before that distribution exists.
    _with_build_lock(joinpath(homedir(), ".bridgestan", "stanblocks-install.pid")) do
        realpath(BridgeStan.get_bridgestan_path())
    end
end

_build_file_digest(path) = isfile(path) ? bytes2hex(open(SHA.sha256, path)) : nothing

# BridgeStan.compile_model (2.9.0) splices `stanc_args` into the single word
# `"STANCFLAGS=--include-paths=. $stanc_args"`, which Julia expands as a cartesian
# product: one `STANCFLAGS=` assignment per element, appended after the caller's
# make arguments. make keeps the LAST assignment, so of several stanc arguments
# only the last reaches stanc, and any of them replaces a `STANCFLAGS=` make
# argument. Collect every stanc word here; `_compile_stanc_args` passes them as one.
function _split_stanc_make_args(make_args, stanc_args)
    make, stanc = String[], String[]
    for arg in make_args
        startswith(arg, "STANCFLAGS=") ?
            append!(stanc, Base.shell_split(chopprefix(arg, "STANCFLAGS="))) : push!(make, arg)
    end
    make, append!(stanc, stanc_args)
end
# One element yields one `STANCFLAGS=` word under either splicing; none leaves
# BridgeStan's default untouched, as for every build before stanc arguments.
_compile_stanc_args(stanc) = isempty(stanc) ? String[] : [join(stanc, ' ')]

function _build_compilers(make_args)
    commands = String["c++", "g++", "clang++", get(ENV, "CXX", "")]
    for arg in make_args
        startswith(arg, "CXX=") && push!(commands, arg[5:end])
    end
    paths = String[]
    for command in commands
        words = Base.shell_split(command)
        isempty(words) && continue
        path = Sys.which(first(words))
        path === nothing || push!(paths, realpath(path))
    end
    [(; path, digest=_build_file_digest(path)) for path in sort!(unique(paths))]
end

# The state of a file or directory tree without reading file contents: every
# entry's relative path, mode, size, modification and status-change times and
# inode, in sorted order. Writing, replacing, rebuilding or re-linking any
# entry changes it (the status-change time cannot be set back); reading does
# not. Symbolic links inside the tree are recorded, not followed.
function _build_tree_stamp(path)
    ctx = SHA.SHA2_256_CTX()
    _stamp_build_entry!(ctx, path, "", stat(path))
    bytes2hex(SHA.digest!(ctx))
end

function _stamp_build_entry!(ctx, path, rel, s)
    SHA.update!(ctx, codeunits(string(rel, '\0', s.mode, '\0', s.size, '\0',
        s.mtime, '\0', s.ctime, '\0', s.inode, '\n')))
    isdir(s) || return
    for name in readdir(path)
        _stamp_build_entry!(ctx, joinpath(path, name), string(rel, '/', name),
            lstat(joinpath(path, name)))
    end
end

# Variables whose paths are not toolchain inputs. PATH and LD_LIBRARY_PATH are
# search lists: a single entry would stamp a whole system directory, and the
# compilers found through PATH are digested instead. BRIDGESTAN selects the
# home make runs and writes objects in; its version and location identify it.
const _BUILD_UNSTAMPED_VARIABLES = ("BRIDGESTAN", "LD_LIBRARY_PATH", "PATH")

# The path a Make variable's value names. BridgeStan runs make in its home, so
# a relative value names a path there.
_named_path(home, value) = isabspath(value) ? value : joinpath(home, value)

# The resolved path a Make variable's value names, or `nothing`.
function _toolchain_path(home, value)
    (value === nothing || isempty(value)) && return nothing
    path = _named_path(home, value)
    ispath(path) ? realpath(path) : nothing
end

const _QUOTED_INCLUDE = r"^[ \t]*#[ \t]*include[ \t]*\"([^\"\n]+)\""m

# The files C++ headers reach through quoted `#include "…"` lines that resolve
# next to the including file, where the preprocessor looks first, transitively
# and resolved. Angle includes and quoted includes found only through `-I` are
# not followed; such files are passed as dependencies. A commented-out include
# is followed too, which only costs a rebuild when the file it names changes.
function _quoted_includes(headers)
    seen, found = Set{String}(), String[]
    pending = String[normpath(header) for header in headers if isfile(header)]
    while !isempty(pending)
        path = pop!(pending)
        path in seen && continue
        push!(seen, path)
        for m in eachmatch(_QUOTED_INCLUDE, read(path, String))
            included = normpath(joinpath(dirname(path), m[1]))
            isfile(included) || continue
            push!(found, realpath(included))
            push!(pending, included)
        end
    end
    found
end

# Make arguments and build environment variables that name a toolchain tree
# or file (`STAN=`, `MATH=`, `STANC=`, `USER_HEADER=`, …) are identified by
# what they name, not by their text: a moved link or an in-place rebuild
# behind the same text must not reuse a library built against the old tree.
# The files a `USER_HEADER` and the explicit `dependencies` (absolute paths of
# existing files) include are inputs too, though no variable names them. Each
# path is recorded resolved, and each outermost one is stamped once.
function _toolchain_paths(home, make_args, environment; dependencies=String[])
    assignments = Pair{String,String}[]
    for arg in make_args
        parts = split(arg, '='; limit=2)
        length(parts) == 2 && push!(assignments, String(parts[1]) => String(parts[2]))
    end
    for (name, value) in environment
        value === nothing || push!(assignments, name => value)
    end
    paths = Pair{String,String}[]
    for (name, value) in assignments
        name in _BUILD_UNSTAMPED_VARIABLES && continue
        path = _toolchain_path(home, value)
        path === nothing || push!(paths, name => path)
    end
    append!(paths, ["dependency" => realpath(path) for path in dependencies])
    headers = [_named_path(home, value) for (name, value) in assignments
               if name == "USER_HEADER" && !isempty(value)]
    named = Set(last.(paths))
    for path in _quoted_includes([headers; dependencies])
        path in named || (push!(named, path); push!(paths, "#include" => path))
    end
    roots = String[]
    for path in sort!(unique(last.(paths)))
        any(root -> isdir(root) && startswith(path, joinpath(root, "")), roots) ||
            push!(roots, path)
    end
    (; paths, stamps=[root => _build_tree_stamp(root) for root in roots])
end

function _native_build_context(home, make_args; stanc_args=String[], dependencies=String[])
    # Source distributions and build configuration are read-only while in use.
    # Include compiler binaries and the documented make/local override so an
    # ordinary compiler/configuration upgrade cannot reuse a loaded old library.
    env_names = String["CXX", "CC", "AR", "LD", "RANLIB", "CXXFLAGS", "CPPFLAGS",
        "LDFLAGS", "LDLIBS", "CXXSTDFLAGS", "O", "MATH", "USER_HEADER",
        "MAKE", "MAKEFLAGS", "PATH", "LD_LIBRARY_PATH"]
    # Include documented and future Stan/BridgeStan/TBB Make switches, notably
    # BRIDGESTAN_AD_HESSIAN and STANC3_VERSION. Build-root configuration is
    # represented separately in _default_build_path.
    append!(env_names, filter(collect(keys(ENV))) do name
        (startswith(name, "STAN") && !startswith(name, "STANBLOCKS_")) ||
            startswith(name, "BRIDGESTAN") || startswith(name, "TBB")
    end)
    sort!(unique!(env_names))
    environment = [name => get(ENV, name, nothing) for name in env_names]
    files = ("Makefile", joinpath("make", "local"), joinpath("bin", "stanc"))
    context = (; format=1, bridgestan=string(pkgversion(BridgeStan)), home,
       platform=string(Sys.MACHINE, '/', Sys.KERNEL), make_args,
       cpu=(; target=Sys.CPU_NAME, models=sort!(unique(info.model for info in Sys.cpu_info()))),
       environment,
       files=[name => _build_file_digest(joinpath(home, name)) for name in files],
       compilers=_build_compilers(make_args))
    # The home's own trees are identified by its BridgeStan version and path.
    # Only builds that name other toolchain paths, or pass stanc arguments,
    # gain an entry, so the keys of builds that do neither are unchanged.
    toolchain = _toolchain_paths(home, make_args, environment; dependencies)
    isempty(toolchain.paths) || (context = (; context..., toolchain))
    isempty(stanc_args) ? context : (; context..., stanc_args)
end

_default_build_dir() = abspath(get(ENV, "STANBLOCKS_BUILD_DIR", joinpath(tempdir(), "stanblocks")))
function _default_build_path(sc, context=(;); filename="model.stan")
    directory = _default_build_dir()
    key = bytes2hex(SHA.sha256(JSON.json((; source=sc, context, filename, directory))))
    # Distinct C++ programs also need distinct model class/namespace names,
    # independently of the dynamic loader's library-path identity.
    stem = splitext(filename)[1]
    joinpath(directory, key, string(stem, '_', key, ".stan"))
end

# rename, rather than mv(force=true), keeps the old file visible until the new
# complete bytes replace it. The temporary file lives on the same filesystem.
function _rename_build_file(src, dst)
    # Julia 1.10's Filesystem.rename falls back to copy/remove on failure.
    # Publication needs a strict native rename on every supported Julia line.
    err = ccall(:jl_fs_rename, Int32, (Cstring, Cstring), src, dst)
    err < 0 && Base.uv_error("rename($(repr(src)), $(repr(dst)))", err)
    nothing
end

# Windows readers opened without delete sharing briefly prevent a rename: of
# the source with EBUSY, of a replaced destination with EACCES. Retry only that
# platform's `code` for up to 1.6 s, then rethrow the last error.
function _retry_windows_rename(src, dst, code)
    delays = Base.ExponentialBackOff(n=10, first_delay=0.01,
        max_delay=0.25, factor=2.0, jitter=0.0)
    Base.retry(_rename_build_file; delays,
        check=(_, err) -> Sys.iswindows() && err isa Base.IOError && err.code == code)(src, dst)
end

function _atomic_build_write(path, value)
    mkpath(dirname(path))
    tmp, io = mktemp(dirname(path))
    try
        write(io, value)
        close(io)
        # Keep the old file visible and propagate a persistent failure unchanged.
        _retry_windows_rename(tmp, path, Base.UV_EACCES)
    finally
        isopen(io) && close(io)
        ispath(tmp) && rm(tmp)
    end
    nothing
end

"Write a source alias atomically; identical bytes preserve its modification time."
function _write_stan_source(path, sc)
    if isfile(path)
        read(path, String) == sc && return false
        @warn "stan_instantiate: overwriting stale Stan source" path
    end
    _atomic_build_write(path, sc)
    true
end

function _write_immutable_stan_source(path, sc)
    if isfile(path)
        read(path, String) == sc || error(
            "StanBlocks artifact source was modified or its identity collided: ", path)
    else
        _atomic_build_write(path, sc)
    end
    nothing
end

function _source_alias(path, sc)
    path === nothing && return "model.stan"
    endswith(path, ".stan") || throw(ArgumentError("path must end in .stan"))
    mkpath(dirname(abspath(path)))
    # Canonicalize the directory so two aliases through symlinks share a lock.
    target = joinpath(realpath(dirname(abspath(path))), basename(path))
    _with_build_lock(target * ".pid") do
        _write_stan_source(target, sc)
    end
    basename(target)
end

# BridgeStan constructors load a library and initialize native model state.
# Serialize that boundary conservatively; each call still owns a fresh model.
const _native_constructor_lock = ReentrantLock()

# The source and library paths of `sc`'s artifact under `context`, through the
# real build directory so that aliases through symbolic links share one lock.
function _artifact_paths(sc, context, filename)
    source = _default_build_path(sc, context; filename)
    mkpath(dirname(source))
    source = joinpath(realpath(dirname(source)), basename(source))
    source, splitext(source)[1] * "_model.so"
end

# The library an earlier build published for `sc`, or `nothing`. A published
# library is complete and never overwritten, so finding one takes no lock and
# never waits for another process's build. This context read can meet a first
# build writing into the toolchain (BridgeStan downloads stanc, TBB builds into
# a named tree): it then fails or names no library, and the caller reads the
# context again under the toolchain lock, where a persistent error surfaces.
function _published_library(sc, context, filename)
    current = try
        context()
    catch err
        err isa Base.IOError || err isa SystemError || rethrow()
        return nothing
    end
    source, library = _artifact_paths(sc, current, filename)
    isfile(library) || return nothing
    _write_immutable_stan_source(source, sc)
    library
end

"The published immutable library for Stan source `sc`, compiled first if absent."
function _artifact_library(sc; filename="model.stan", make_args, stanc_args=String[], dependencies=String[])
    args, stanc = _split_stanc_make_args(make_args, stanc_args)
    home = _bridgestan_build_home()
    context() = _native_build_context(home, args; stanc_args=stanc, dependencies)
    published = _published_library(sc, context, filename)
    published === nothing || return published
    toolchain_lock = _toolchain_lock_path(home)
    source, library = _artifact_paths(sc, _with_build_lock(context, toolchain_lock), filename)
    _with_build_lock(source * ".pid") do
        _write_immutable_stan_source(source, sc)
        if !isfile(library)
            # Even distinct artifacts can build common BridgeStan/TBB files.
            _with_build_lock(toolchain_lock) do
                mktempdir(dirname(source); prefix="build-") do staging
                    staged_source = joinpath(staging, basename(source))
                    write(staged_source, sc)
                    built = BridgeStan.compile_model(staged_source; make_args=args,
                        stanc_args=_compile_stanc_args(stanc))
                    # A published library is complete and is NEVER overwritten.
                    # Failed compilation only leaves private staging files.
                    Base.Filesystem.rename(built, library)
                end
            end
        end
    end
    library
end

function _instantiate_artifact(sc, data; path, make_args, nan_on_error, warn)
    library = _artifact_library(sc; filename=_source_alias(path, sc),
        make_args=String[arg for arg in make_args])
    # A published library is immutable, so constructing from it needs no build lock.
    lock(_native_constructor_lock) do
        StanLogDensityProblems.StanProblem(library, data; nan_on_error, warn)
    end
end
