using Statistics, StanBlocks, PosteriorDB
mad(x) = median(abs.(x .- median(x)))
struct UncertainStatistic{F,V}
    f::F
    vals::V
end
se(s::UncertainStatistic{typeof(mean)}) = std(s.vals)/sqrt(length(s.vals))
se(s::UncertainStatistic{typeof(median)}) = mad(s.vals)/sqrt(length(s.vals))
Base.isless(s::UncertainStatistic, x::Float64) = isless(s.f(s.vals) + 2 * se(s), x)
Base.show(io::IO, s::UncertainStatistic) = print(io, round(s.f(s.vals); sigdigits=2), " ± ", round(se(s); sigdigits=2))

jimplemtations_lines = readlines("PosteriorDBExt.jl")
posterior_dimension(posterior_name) = StanBlocks.dimension(StanBlocks.julia_implementation(PosteriorDB.posterior(PosteriorDB.database(), posterior_name)))
implementations_string(posterior_name) = begin
    data, model = split(posterior_name, "-")
    line = findfirst(line->contains(line, model), jimplemtations_lines)
    Markdown.parse("[Stan](https://github.com/stan-dev/posteriordb/tree/master/posterior_database/models/stan/$model.stan), [Julia](https://github.com/nsiccha/StanBlocks.jl/blob/main/ext/PosteriorDBExt.jl#L$line)")
end