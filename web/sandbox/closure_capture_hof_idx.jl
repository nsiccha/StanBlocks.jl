# Closure + HOF, PKPD-style: pass an `int[]` index array to
# `simple_reduce_sum`. Stan's `reduce_sum` accepts arrays, so the helper's
# `x_slice::anything[n]` resolves to `array[] int` (not `vector`) and
# stanc accepts the call.
@slic (;m=5, idxs=collect(1:5)) begin
    shift ~ std_normal()
    s     = simple_reduce_sum((i) -> shift * i, idxs)
end
