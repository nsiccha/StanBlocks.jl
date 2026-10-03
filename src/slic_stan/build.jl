# File locks cover both Julia Tasks and cooperating processes. Do not expire a
# live compiler's lock on a timeout: a slow build must never acquire a competitor.
function _with_build_lock(f, path)
    mkpath(dirname(path))
    Pidfile.mkpidlock(f, path; wait=true)
end

function _bridgestan_build_home()
    # BridgeStan may install its default source distribution on first use. The
    # installer lock has a stable location even before that distribution exists.
    _with_build_lock(joinpath(homedir(), ".bridgestan", "stanblocks-install.pid")) do
        realpath(BridgeStan.get_bridgestan_path())
    end
end

_build_file_digest(path) = isfile(path) ? bytes2hex(open(SHA.sha256, path)) : nothing

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

function _native_build_context(home, make_args)
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
    files = ("Makefile", joinpath("make", "local"), joinpath("bin", "stanc"))
    (; format=1, bridgestan=string(pkgversion(BridgeStan)), home,
       platform=string(Sys.MACHINE, '/', Sys.KERNEL), make_args,
       cpu=(; target=Sys.CPU_NAME, models=sort!(unique(info.model for info in Sys.cpu_info()))),
       environment=[name => get(ENV, name, nothing) for name in env_names],
       files=[name => _build_file_digest(joinpath(home, name)) for name in files],
       compilers=_build_compilers(make_args))
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

function _atomic_build_write(path, value)
    mkpath(dirname(path))
    tmp, io = mktemp(dirname(path))
    try
        write(io, value)
        close(io)
        # Windows readers opened without delete sharing temporarily prevent
        # replacement. Retry only that platform's access-denied error; keep
        # the old file visible and propagate a persistent failure unchanged.
        delays = Base.ExponentialBackOff(n=10, first_delay=0.01,
            max_delay=0.25, factor=2.0, jitter=0.0)
        Base.retry(_rename_build_file; delays,
            check=(_, err) -> Sys.iswindows() && err isa Base.IOError &&
                err.code == Base.UV_EACCES)(tmp, path)
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

function _instantiate_artifact(sc, data; path, make_args, nan_on_error, warn)
    args = String[arg for arg in make_args]
    filename = _source_alias(path, sc)
    home = _bridgestan_build_home()
    toolchain_lock = joinpath(home, ".stanblocks-build.pid")
    context = _with_build_lock(toolchain_lock) do
        _native_build_context(home, args)
    end
    source = _default_build_path(sc, context; filename)
    filename = basename(source)
    mkpath(dirname(source))
    source = joinpath(realpath(dirname(source)), basename(source))
    library = splitext(source)[1] * "_model.so"
    _with_build_lock(source * ".pid") do
        _write_immutable_stan_source(source, sc)
        if !isfile(library)
            # Even distinct artifacts can build common BridgeStan/TBB files.
            _with_build_lock(toolchain_lock) do
                mktempdir(dirname(source); prefix="build-") do staging
                    staged_source = joinpath(staging, filename)
                    write(staged_source, sc)
                    built = BridgeStan.compile_model(staged_source; make_args=args)
                    # A published library is complete and is NEVER overwritten.
                    # Failed compilation only leaves private staging files.
                    Base.Filesystem.rename(built, library)
                end
            end
        end
        lock(_native_constructor_lock) do
            StanLogDensityProblems.StanProblem(library, data; nan_on_error, warn)
        end
    end
end
