"""
Every `@deffun`/`@defsig` definition registers its `tracetype` / `fundef` /
`inline_body` rules and its provenance marker on head-first tables
(`tracetype_by_head(::typeof(f), x::CanonicalExpr{…})`), never on the public
node functions. With the node first, every rule shared one method-table bucket
(`CanonicalExpr`), so each new method was type-intersected with all of them: a
downstream module with 250 `@deffun` functions spent 15 of its 19 s `using` time
inserting them (snag `deffun-methods-m-806280c3`). With the head first, a
function's rules sit in a bucket of their own.
"""
@testitem "slic: @deffun rules are filed head-first, off the public node tables" tags=[:slic, :regression] setup=[StanBlocksImports, StanBlocksTestSetup] begin
    using .StanBlocksTestSetup: stan_block
    import StanBlocks as SB

    public = (SB.tracetype, SB._tracetype, SB.fundef, SB._fundef, SB.inline_body)
    by_head = (SB.tracetype_by_head, SB._tracetype_by_head, SB.fundef_by_head,
        SB._fundef_by_head, SB.inline_body_by_head, SB._backing_provenance)
    count_methods(fs) = map(f -> length(methods(f)), fs)
    public_before = count_methods(public)
    by_head_before = count_methods(by_head)

    @deffun begin
        hf_scaled(x::vector[n], s::real)::vector[n] = s * x
        hf_scaled(x::real, s::real)::real = s * x
        hf_twice!(x::real)::real = 2 * x
        @lhs @lpxf hf_shifted_lpdf(y::real, mu::real)::real = normal_lpdf(y, mu, 1.0)
        hf_shifted_lpdfs(y::real, mu::real)::real = normal_lpdf(y, mu, 1.0)
        hf_shifted_rng(mu::real)::real = normal_rng(mu, 1.0)
        hf_native(x::real)::real
    end

    # No definition reaches the public tables; each one lands on its own head.
    @test count_methods(public) == public_before
    @test all(count_methods(by_head) .> by_head_before)
    own = [m for f in by_head for m in methods(f) if m.module === @__MODULE__]
    heads = Set(Base.unwrap_unionall(m.sig).parameters[2] for m in own)
    @test all(h -> h isa DataType && h <: Function && Base.issingletontype(h), heads)
    @test issubset(typeof.((hf_scaled, hf_twice!, hf_shifted_lpdf, hf_shifted, hf_native)), heads)

    # A public node function holds only its bridge for a call node: a rule
    # written on it for a `CanonicalExpr` would bypass (and shadow) the tables.
    node_methods(f) = [m for m in methods(f)
        if (p = Base.unwrap_unionall(m.sig).parameters[2]; p isa Type && p !== Any && p <: SB.CanonicalExpr)]
    @test all(f -> length(node_methods(f)) == 1, public)

    @test SB.udf_backed(hf_scaled) && !SB.native_backed(hf_scaled)
    @test SB.native_backed(hf_native) && !SB.udf_backed(hf_native)

    # The rules still resolve through the bridges, the head-first table and the
    # generic rules beside them (`getindex`, arithmetic) alike.
    model = @slic (; y = 0.4, X = [1.0 2.0; 3.0 4.0; 5.0 6.0]) begin
        s ~ std_normal()
        mu ~ normal(0, 1)
        z = hf_scaled(X[:, 1], s) + hf_twice!(mu) * hf_scaled(1.0, s)
        y ~ hf_shifted(mu + z[1])
    end
    functions = stan_block(stan_code(model), "functions")
    @test occursin("hf_scaled(vector x, real s)", functions)
    @test occursin("hf_scaled(real x, real s)", functions)
    @test occursin("hf_shifted_lpdf(real y, real mu)", functions)
    @test !occursin("hf_twice", functions)
end
