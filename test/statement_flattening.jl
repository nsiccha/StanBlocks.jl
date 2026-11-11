"""
Program-sized concatenation folds copied the accumulated prefix repeatedly.
Check statement order and ownership, then bound allocation growth without
depending on wall-clock timings or a particular allocator's exact byte count.
"""
@testitem "slic: statement flattening preserves order with linear allocation growth" tags=[:slic, :regression] setup=[StanBlocksImports] begin
    st = StanBlocks.stan
    first_stmt, last_stmt = :(x = 1), :(y = x + 2)
    nested = Expr(:block, LineNumberNode(1), first_stmt,
        Expr(:block, LineNumberNode(2), Expr(:block), last_stmt))
    flat = st.unblock(nested)
    @test flat == [first_stmt, last_stmt]
    @test flat[1] === first_stmt && flat[2] === last_stmt
    push!(flat, :(z = 3))
    @test length(nested.args) == 3
    @test st.unblock(Expr(:block)) == Any[]
    @test st.unblock(st.canonical(nested)) ==
        [st.canonical(first_stmt), st.canonical(last_stmt)]

    names = st.builtin_module_names(Expr(:vcat, :plain, Expr(:row, :toy_lpdf, :end_name)))
    @test names == [:plain, :toy_lpdf, :toy_lpdfs, :toy, :toy_rng,
        :toy_cdf, :toy_lccdf, :toy_lcdf, :end_name]
    @test isempty(st.builtin_module_names(Expr(:vcat)))

    statements(n) = Expr(:block, [Expr(:block, LineNumberNode(i), :(x = $i))
        for i in 1:n]...)
    declarations(n) = Expr(:vcat, [Symbol(:name_, i) for i in 1:n]...)
    cost(f, x) = (f(x); @allocated f(x))
    # Four times as many items may allocate up to six times as much. A fold
    # needs about sixteen times as much, even after its compilation is warm.
    @test cost(st.unblock, statements(8_000)) < 6 * cost(st.unblock, statements(2_000))
    @test cost(st.builtin_module_names, declarations(8_000)) <
        6 * cost(st.builtin_module_names, declarations(2_000))
end
