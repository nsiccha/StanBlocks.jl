sm19a = @slic begin 
    x = rep_vector(0., n)
    xx = x .* x
    return xx
end
sm19b = @slic begin 
    x = rep_vector(0., n)
    xx = append_row(x,x)
    return xx
end

@testset "issue19" begin
    @test compiles(@slic (;n=10, y=1.) begin 
        x ~ sm19a(;n)
        y ~ simple(x)
    end)
    @test compiles(@slic (;n=10, y=1.) begin 
        x ~ sm19b(;n)
        y ~ simple(x)
    end)
end