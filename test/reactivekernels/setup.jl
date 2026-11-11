using Pkg, TOML

Pkg.activate(@__DIR__)
sources = TOML.parsefile(joinpath(@__DIR__, "Project.toml"))["sources"]
# Explicit adds also support Julia 1.10, which ignores [sources]. Resolve
# these together because ReactiveKernels depends on OutputSignatures.
withenv("JULIA_PKG_PRECOMPILE_AUTO" => "0") do
    Pkg.add([PackageSpec(; name, url=source["url"], rev=source["rev"])
        for (name, source) in sources if haskey(source, "url")])
    Pkg.develop(path=normpath(joinpath(@__DIR__, "../..")))
end
Pkg.precompile()
