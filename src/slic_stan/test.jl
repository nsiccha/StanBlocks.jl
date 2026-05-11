infers(p; re=true) = try
    stan_model(p)
    return true
catch e
    re && rethrow()
    return false
end
full_cqual_eq(p; kwargs...) = begin
    rv = true
    for (key, value) in pairs(stan_model(p).vars)
        rvalue = get(kwargs, key, :missing)
        if cqual(value) !== rvalue
            @error "cqual($key) == $(cqual(value)) != $rvalue"
            rv = false
        end
    end 
    rv || @info map(cqual, (;pairs(stan_model(p).vars)...))
    return rv
end

transpiles(p; re=true) = try
    stan_code(p)
    return true
catch e
    re && rethrow()
    return false
end
compiles(p; re=true) = try
    instantiate(p)
    return true
catch e
    re && rethrow()
    return false
end

# Locate the `stanc` binary. Resolution order:
#   1. $STANC_PATH env var (explicit override)
#   2. BridgeStan's shipped binary (bin/stanc relative to its source path)
#   3. Error — callers should set $STANC_PATH or install BridgeStan.
function _stanc_bin()
    haskey(ENV, "STANC_PATH") && return ENV["STANC_PATH"]
    bs_path = BridgeStan.get_bridgestan_path(; download=false)
    if !isempty(bs_path)
        candidate = joinpath(bs_path, "bin", "stanc")
        isfile(candidate) && return candidate
    end
    error("BridgeStan not loaded and STANC_PATH not set")
end

"""
    stanc_check(stan_code::AbstractString; warn_pedantic=true) -> (ok::Bool, output::String)

Run the `stanc` compiler on `stan_code` (written to a temporary file) and
return whether it accepted the code plus any compiler output (warnings,
errors). Binary discovery: `\$STANC_PATH` env var → BridgeStan's
`bin/stanc`. Errors if neither is available.
"""
function stanc_check(stan_code::AbstractString; warn_pedantic::Bool=true)
    stanc = _stanc_bin()
    tmpfile = tempname() * ".stan"
    write(tmpfile, stan_code)
    io = IOBuffer()
    flags = warn_pedantic ? `--warn-pedantic` : ``
    ok = success(pipeline(`$stanc $flags $tmpfile`; stderr=io, stdout=io))
    rm(tmpfile; force=true)
    (ok=ok, output=String(take!(io)))
end