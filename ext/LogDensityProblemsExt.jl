module LogDensityProblemsExt
import StanBlocks, LogDensityProblems
LogDensityProblems.capabilities(::Type{StanBlocks.VectorPosterior{F,G,GQ,N}}) where{F,G,GQ,N} = if G == Missing
    LogDensityProblems.LogDensityOrder{0}()
else
    LogDensityProblems.LogDensityOrder{1}()
end
LogDensityProblems.dimension(p::StanBlocks.VectorPosterior) = StanBlocks.dimension(p)
LogDensityProblems.logdensity(p::StanBlocks.VectorPosterior, x) = p(x)
LogDensityProblems.logdensity_and_gradient(p::StanBlocks.VectorPosterior, x) = p.g(x)
end