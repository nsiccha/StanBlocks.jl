import StanBlocks.stan: @deffun, full_cqual_eq, transpiles, compiles, stan_model

msg(e::ErrorException) = e.msg
msg(e::AssertionError) = e.msg
msg(e::MethodError) = e.msg


@deffun begin 
    simple_lpdf(y, x) = 0.
    simple_lpdfs(y, x) = 0.
    simple_rng(x) = 0.
    vararg_lpdf(y, args...) = 0.
    vararg_lpdfs(y, args...) = 0.
    vararg_rng(args...) = 0.

    my_lpdf(y, fargs...) = reject(1)
    my_lpdfs(args...) = reject(1)
    my_rng(args...) = reject(1)
    my_lpdf(y, ::typeof(simple), args...) = simple_lpdf(y, args...)
    my_lpdfs(y, ::typeof(simple), args...) = simple_lpdfs(y, args...)
    my_rng(::typeof(simple), args...) = simple_rng(args...)
    my_lpdf(y, ::typeof(vararg), args...) = vararg_lpdf(y, args...)

    fof_lpdf(y, f, args...) = my_lpdf(y, f, args...)
    fof_lpdfs(y, f, args...) = my_lpdfs(y, f, args...)
    fof_rng(f, args...) = my_rng(f, args...)
    srs2_lpdf(y, f, args...) = simple_reduce_sum(srs2_helper, rep_array(y, 1), f, args...)
    srs2_helper(y, f, args...) = my_lpdf(y, f, args...)
    srs2_lpdfs(y, f, args...) = 0.
    srs2_rng(f, args...) = 0.
end

@testset "compilation" begin
    @test compiles(@slic (;obs=0.) begin
        loc ~ std_normal()
        scale ~ std_normal(;lower=0.) 
        obs ~ normal(loc, scale)
    end)  
    @test compiles(@slic (;obs=0.) begin
        loc ~ std_normal()
        obs ~ simple(loc)
    end)  
    @test compiles(@slic (;obs=0.) begin
        loc ~ std_normal()
        obs ~ vararg(loc)
    end)  
    @test compiles(@slic (;obs=0.) begin
        loc ~ std_normal()
        obs ~ fof(simple, loc)
    end)  
    @test compiles(@slic (;obs=0.) begin
        loc ~ std_normal()
        obs ~ srs2(vararg, loc)
    end)  
    @test compiles(@slic (;obs=0.) begin
        loc ~ std_normal()
        obs ~ srs2(vararg, loc, (1, 2, 3))
    end)  
    @test compiles(stan_model(@slic (;obs=randn(5)) begin
        loc ~ std_normal()
        scale ~ std_normal(;lower=0.) 
        obs ~ normal(loc, scale)
    end)(;obs=randn(10)))
end

include("issues/9.jl")
include("issues/10.jl")
include("issues/17.jl")
include("issues/18.jl")
include("issues/19.jl")
include("posteriordb.jl")