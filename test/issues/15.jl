sm15a = @slic begin 
    x = rep_vector(0., n)
    return x
end
sm15b = @slic begin 
    x = rep_vector(0., n)
    xx = append_row(x, x)
    return xx
end

@testset "issue15" begin
    @test stan_code(sm15a(quote 
        xx = append_row(x, x)
        return xx
    end ; n=10, y=1.)) == stan_code(sm15b(; n=10, y=1.))
    @test compiles(sm15a(;n=10, y=1.))
    @test compiles(sm15b(;n=10, y=1.))
end