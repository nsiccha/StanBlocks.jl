module StanBlocksTests
using Test, Random, Statistics, TestModules
using StanBlocks
using LogDensityProblems
import StanBlocks.stan: @deffun, full_cqual_eq, transpiles, compiles, stan_model, stan_code, instantiate
using PosteriorDB
include("StanBlocksTests.jl")
end

using TestModules
runtests!(StanBlocksTests)
