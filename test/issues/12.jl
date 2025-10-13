sm12a = @slic begin 
    x ~ std_normal(;n)
    return x .* x
end
sm12b = @slic begin 
    x ~ std_normal(;n)
    return x
end

@testset "issue12" begin
    @test stan_code(sm12a(quote 
        return x
    end ; n=10, y=1.)) == stan_code(sm12b(; n=10, y=1.))
    @test compiles(@slic (;n=10, y=1.) begin 
        x ~ sm12a(;n)
        y ~ simple(x)
    end)
    @test compiles(@slic (;n=10, y=1.) begin 
        x ~ sm12b(;n)
        y ~ simple(x)
    end)
end