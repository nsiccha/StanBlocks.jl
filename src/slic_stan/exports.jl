"""
    stan_instantiate(model; kwargs...) -> StanProblem

Exported alias for [`StanBlocks.instantiate`](@ref). Compiles `model`
(a [`SlicModel`](@ref StanBlocks.SlicModel) or [`StanModel`](@ref
StanBlocks.StanModel)) via BridgeStan and returns a
`StanLogDensityProblems.StanProblem` implementing the
`LogDensityProblems` interface. See [`instantiate`](@ref
StanBlocks.instantiate) for the full kwarg list.
"""
const stan_instantiate = instantiate