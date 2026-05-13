# Custom-type prototype: `RaggedVector` declared via `@usertype`. Field
# access (`rv.mem`, `rv.ends`) and the `Base.length` / `Base.getindex`
# methods dispatched on the `RaggedVector` tag drive standard Julia
# usage; Stan-side a `RaggedVector` value renders as a positional
# `tuple(vector, array[] int)`.
@slic (;mem=randn(8), ends=[3, 5, 8]) begin
    rv  = RaggedVector(mem, ends)
    n   = length(rv)
    g1  = rv[1]            # first group, length 3
    g3  = rv[3]            # third group, length 3
    tot = sum(g1) + sum(g3)
end
