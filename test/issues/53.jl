@testset "issue53" begin
    sm53 = @slic begin 
        x ~ std_normal(;n)
    end
    sm53a = sm53(quote 
        n=3
    end)
    @test compiles(sm53(;n=3))
    @test compiles(sm53a)
end