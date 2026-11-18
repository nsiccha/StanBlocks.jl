using Pkg

Pkg.activate(@__DIR__)
# Julia 1.10 ignores [sources]: develop this checkout explicitly.
Pkg.develop(path=normpath(joinpath(@__DIR__, "../..")))
Pkg.instantiate()
Pkg.precompile()
