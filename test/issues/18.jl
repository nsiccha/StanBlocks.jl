@deffun begin 
    issue18_lpdf(y) = normal_cdf(y, 0, 1) + normal_lcdf(y, 0, 1) + normal_lccdf(y, 0, 1)
end

@testset "issue18" begin
    @test compiles(@slic (;n=10, y=1.) begin 
        x ~ issue18(;n)
        y ~ simple(x)
    end)
end