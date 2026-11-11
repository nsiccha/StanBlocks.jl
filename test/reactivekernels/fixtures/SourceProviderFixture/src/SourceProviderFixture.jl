module SourceProviderFixture
using StanBlocks
@deffun @juliacompat precompiled_loop(x::real)::real = begin
    result = 0.0
    for i in 1:4
        result += x * i
    end
    result
end
end
