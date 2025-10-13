@deffun begin 
    issue17_lpdf(y::vector[n]) = begin 
        rv = 0.
        for i in 1:n
            rv += y[i]
        end
        for i in 1:n-1
            rv += y[i]
        end
        for i in 1:dims(y)[1]
            rv += y[i]
        end
        rv
    end
end

@testset "issue17" begin
    @test compiles(@slic (;n=10, y=1.) begin 
        x ~ issue17(;n)
        y ~ simple(x)
    end)
end