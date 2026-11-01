"""
Binding ragged data flattens its groups into one memory vector. `to_ragged` did
that with `reduce(vcat, x; init=T[])`, and the `init` selects Base's generic left
fold rather than its `vcat` specialisation, so each group copied everything before
it: 8·T·G²/2 bytes for G groups of T reals. Tracing paid this on every
`stan_model`, so `stan_descriptor` and everything else that traces a `SlicModel`
did too, reaching 6.4 GB per trace at 100 groups of 80,000 (snag
`stan-descriptor-4e1c6bcb`). The groups now flatten in one pass.
"""
@testitem "slic: ragged data flattens in one pass, however many groups it has" tags=[:slic, :ragged, :descriptor, :regression] setup=[StanBlocksImports] begin
    to_ragged = StanBlocks.stan.to_ragged

    # The encoding the fold produced, for every shape of group it accepted.
    r = to_ragged([[1.0, 2.0], Float64[], [3.0]])
    @test r.mem == [1.0, 2.0, 3.0] && r.mem isa Vector{Float64}
    @test r.ends == [2, 2, 3]
    @test to_ragged([[1, 2], [3]]).mem isa Vector{Int}
    @test to_ragged([1.0:3.0, 4.0:5.0]).mem == [1.0, 2.0, 3.0, 4.0, 5.0]
    @test to_ragged([1.0:3.0, 4.0:5.0]).mem isa Vector{Float64}
    backing = [1.0, 2.0, 3.0, 4.0]
    @test to_ragged([view(backing, 1:1), view(backing, 2:4)]).mem == backing
    empty_groups = to_ragged(Vector{Float64}[])
    @test empty_groups.mem == Float64[] && empty_groups.mem isa Vector{Float64}
    @test empty_groups.ends == Int[]

    m = to_ragged([[1.0 2.0; 3.0 4.0], [5.0;;]])
    @test m.mem == [1.0, 3.0, 2.0, 4.0, 5.0] && m.mem isa Vector{Float64}
    @test (m.ends, m.rows, m.cols) == ([4, 5], [2, 1], [2, 1])
    # Each element converts to the memory's type: a rational matrix has float memory.
    converted = to_ragged([[1//2 1//4], [3//4;;]])
    @test converted.mem == [0.5, 0.25, 0.75] && converted.mem isa Vector{Float64}

    # The memory is the encoding's own: one group is not aliased into it.
    only_group = [1.0, 2.0]
    owned = to_ragged([only_group])
    owned.mem[1] = -1.0
    @test only_group == [1.0, 2.0]

    # One flat copy per call. The fold allocates 500× that for these 1,000 groups,
    # and the margin leaves room for `ends`.
    G, T = 1_000, 50
    copy_bytes = 8 * G * T
    vectors = [fill(float(g), T) for g in 1:G]
    matrices = [fill(float(g), 5, 10) for g in 1:G]
    cost(x) = (to_ragged(x); @allocated to_ragged(x))
    @test cost(vectors) < 2 * copy_bytes
    @test cost(matrices) < 2 * copy_bytes

    # The reported surface. At a fixed number of elements, reflecting a model costs
    # about the same whatever its number of groups; with the fold, 16× the groups
    # cost 15× the bytes.
    model(G, T) = begin
        t = [collect(range(0.5, 24.0; length=T)) for _ in 1:G]
        dv = [fill(1.0, T) for _ in 1:G]
        dose = fill(100.0, G)
        @slic (; t, dv, dose) begin
            sigma ~ exponential(1.0)
            @plate for i in eachindex(t)
                log_k ~ normal(0.0, 1.0)
                pred[i] = dose[i] .* exp.(-exp(log_k) .* t[i])
                dv[i] ~ normal(pred[i], sigma)
            end
        end
    end
    reflected(m) = (stan_descriptor(m); @allocated stan_descriptor(m))
    few, many = reflected(model(100, 2_000)), reflected(model(1_600, 125))
    @test many < 2 * few
end
