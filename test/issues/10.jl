@deffun begin 
    issue10a(::vector[n]) = 0.
    issue10b(_::vector[n]) = 0.
end
@testset "issue10" begin
    @test compiles(@slic (;n=10) begin 
        y ~ std_normal(;n)
        x = issue10a(y)
    end)
    @test compiles(@slic (;n=10) begin 
        y ~ std_normal(;n)
        x = issue10b(y)
    end)
end