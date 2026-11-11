"""
A ragged `ordered[Ks]` is filled by `ordered_jacobian` and carries no bound. A
prior whose family has a bounded support (`exponential`: lower=0) used to be
accepted, so the sampler proposed negative elements and Stan rejected each one:
the prior was truncated by rejection, silently (todo 04zbyc9). It is refused at
trace time now, naming `positive_ordered[Ks]`.
"""
@testitem "slic: a ragged ordered prior from a bounded family is refused" tags=[:slic, :ragged, :regression] setup=[StanBlocksImports, StanBlocksTestSetup] begin
    data = (; K = [2, 3], y = [0.5])
    bounded = @slic data begin
        c::ordered[K] ~ exponential(1.0)
        y ~ normal(c[1][1], 1.0)
    end
    err = try
        stan_code(bounded)
        nothing
    catch e
        e
    end
    @test err !== nothing
    msg = err === nothing ? "" : sprint(showerror, err)
    @test occursin("lower=0", msg)
    @test occursin("positive_ordered[", msg)

    # Two-sided support has no ragged spelling; the message says so.
    beta_prior = @slic data begin
        c::ordered[K] ~ beta(2.0, 2.0)
        y ~ normal(c[1][1], 1.0)
    end
    err = try
        stan_code(beta_prior)
        nothing
    catch e
        e
    end
    @test err !== nothing && occursin("unbounded support", sprint(showerror, err))

    # The suggested spelling satisfies lower=0, and an unbounded family is fine.
    positive = @slic data begin
        c::positive_ordered[K] ~ exponential(1.0)
        y ~ normal(c[1][1], 1.0)
    end
    @test occursin("positive_ordered_jacobian", stan_code(positive))
    unbounded = @slic data begin
        c::ordered[K] ~ normal(0.0, 1.0)
        y ~ normal(c[1][1], 1.0)
    end
    @test occursin("ordered_jacobian", stan_code(unbounded))
end
