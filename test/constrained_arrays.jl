"""
An array of `ordered` / `positive_ordered` / `simplex` vectors under an
elementwise family, `c::ordered[2, 3] ~ normal(0, 1)`, is one prior per vector.
Stan's elementwise families take no `array[] vector`, so the emitted `c ~
normal(0, 1);` was a stanc error, and a prior-only program failed to trace for
want of a re-draw (todo 15dof8q). It is now the plate of those vectors, which
collects as the same `array[2] ordered[3]`, samples each cell and re-draws it
sorted. A multivariate family (`dirichlet`) still takes the array whole.
"""
@testitem "slic: an array of constrained vectors under an elementwise family" tags=[:slic, :plate, :regression, :stanc] setup=[StanBlocksImports, StanBlocksTestSetup] begin
    using .StanBlocksTestSetup: stan_block, stanc_compiles

    fitted = @slic (; y = [0.5]) begin
        c::ordered[2, 3] ~ normal(0.0, 1.0); y ~ normal(c[1][1], 1.0)
    end
    code = stan_code(fitted)
    @test occursin("array[2] ordered[3] c_c;", stan_block(code, "parameters"))
    @test occursin("array[2] ordered[3] c;", stan_block(code, "transformed parameters"))
    @test occursin(r"c_c\[plate_i__pl_1\] ~ normal\(0\.0, 1\.0\);", stan_block(code, "model"))
    @test stanc_compiles(fitted)
    # The plate spelling it stands for emits the same program, up to the cell name.
    plated = @slic (; y = [0.5]) begin
        c ~ plate(; outer = (2,)) do i
            c::ordered[3] ~ normal(0.0, 1.0); c
        end
        y ~ normal(c[1][1], 1.0)
    end
    @test code == stan_code(plated)

    prior = @slic begin
        c::ordered[2, 3] ~ normal(0.0, 1.0)
    end
    gq = stan_block(stan_code(prior), "generated quantities")
    @test occursin("sort_asc(normal_vector_rng(3, 0.0, 1.0))", gq)
    @test stanc_compiles(prior)

    positive = @slic (; y = [0.5]) begin
        c::positive_ordered[2, 3] ~ exponential(1.0); y ~ normal(c[1][1], 1.0)
    end
    @test occursin("array[2] positive_ordered[3] c_c;", stan_code(positive))
    @test stanc_compiles(positive)
    bounded = @slic (; y = [0.5]) begin
        c::ordered[2, 3] ~ normal(0.0, 1.0; lower = -1.0); y ~ normal(c[1][1], 1.0)
    end
    @test stanc_compiles(bounded)

    # A multivariate family takes the array as is.
    whole = @slic (; y = [0.5], a = [1.0, 1.0, 1.0]) begin
        c::simplex[2, 3] ~ dirichlet(a); y ~ normal(c[1][1], 1.0)
    end
    code = stan_code(whole)
    @test occursin("array[2] simplex[3] c;", stan_block(code, "parameters"))
    @test occursin("c ~ dirichlet(a);", stan_block(code, "model"))
    @test stanc_compiles(whole)
end
