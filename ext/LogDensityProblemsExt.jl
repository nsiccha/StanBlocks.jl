module LogDensityProblemsExt
import StanBlocks, LogDensityProblems
LogDensityProblems.capabilities(::Type{<:StanBlocks.VectorPosterior}) = LogDensityProblems.LogDensityOrder{0}()
LogDensityProblems.dimension(p::StanBlocks.VectorPosterior) = StanBlocks.dimension(p)
LogDensityProblems.logdensity(p::StanBlocks.VectorPosterior, x) = p(x)
end