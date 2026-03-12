module m20 
    using StanBlocks
    @deffun begin 
        f(x) = begin
            return x
        end
    end
    model = @slic begin 
        x ~ std_normal(;n)
        return x
    end
    modela = @slic begin 
        x ~ model(;n)
    end
    modelb = @slic begin 
        x ~ model(;n)
        y = f(x)
    end
end

@testset "issue20" begin
    @test compiles(m20.model(;n=10))
    @test compiles(m20.modela(;n=10))
    @test compiles(m20.modelb(;n=10))
end