# Closure capture lifted into Stan's `reduce_sum` via `simple_reduce_sum`.
# Uses `real[]` (Stan array) instead of `vector` because Stan's `reduce_sum`
# only accepts arrays as its second positional argument — orthogonal to the
# closure mechanism, but needed to actually exercise stanc end-to-end.
@deffun simple_reduce_sum_arr(f, x::real[m], args...)::real =
    reduce_sum(simple_reduce_sum_helper_arr, x, 1, f, args...)
@deffun simple_reduce_sum_helper_arr(x_slice::real[m], slice_start::int, slice_end::int, f, args...)::real = begin
    rv = 0.0
    for i in 1:m
        rv += f(x_slice[i], args...)
    end
    rv
end

@slic (;n=5, x=randn(5)) begin
    shift ~ std_normal()
    s     = simple_reduce_sum_arr((y) -> y + shift, x)
end
