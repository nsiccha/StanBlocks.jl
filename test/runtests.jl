include("StanBlocksTests.jl")
using .StanBlocksTests
using Test
@testset "StanBlocks" begin
    StanBlocksTests.run_all!()
end
