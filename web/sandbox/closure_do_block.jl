# `do x; body end` syntax desugars to `f((x) -> body, args...)` — should
# flow through the same `:->` trace path as a bare lambda.
@slic (;n=5, idxs=collect(1:5)) begin
    shift ~ std_normal()
    s = simple_reduce_sum(idxs) do i
        shift * i
    end
end
