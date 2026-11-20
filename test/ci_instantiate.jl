# Instantiate the test environment the way CI does: develop this checkout into
# it, and take every package `test/Project.toml` sources by url from that url.
# Julia 1.11 and later honor `[sources]` themselves; Julia 1.10 ignores the
# table, so without the explicit add it would test against the registered
# release. BridgeStan comes from its dogfood line this way: its
# `release_memory` is what `test/release_memory.jl` calls.
#
#     julia --project=test test/ci_instantiate.jl
using Pkg, TOML

Pkg.develop(path=dirname(@__DIR__))
if VERSION < v"1.11"
    sources = get(TOML.parsefile(joinpath(@__DIR__, "Project.toml")), "sources", Dict{String,Any}())
    specs = [PackageSpec(; name, url=source["url"], rev=get(source, "rev", nothing),
                 subdir=get(source, "subdir", nothing))
             for (name, source) in sources if haskey(source, "url")]
    isempty(specs) || Pkg.add(specs)
end
Pkg.instantiate()
