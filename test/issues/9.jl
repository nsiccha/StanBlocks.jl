@deffun begin 
    issue9_lpdf(x::vector[n], n) = 0.
    issue9_rng(n) = rep_vector(0., n)
end
@testset "issue9" begin
    @deffun begin 
        preconditioned_normal_lpdf(xi::matrix[m, n], loc::vector[m], scale::vector[m], prescale::matrix[m,m], n) = begin 
            multi_normal_cholesky_lpdf(eachcol(xi), mdivide_left_tri_low(prescale, loc), mdivide_left_tri_low(prescale, diag_matrix(scale)))
        end
    end
    @test compiles(@slic (;n=10) begin 
        x ~ issue9(n)
    end)
    @test compiles(@slic (;n=10, y=1.) begin 
        x ~ issue9(n)
        y ~ vararg(x)
    end)
end