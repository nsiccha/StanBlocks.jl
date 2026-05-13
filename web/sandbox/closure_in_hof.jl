# Anonymous closure passed to a SLIC-side HOF wrapper that itself calls
# Stan's `reduce_sum` — the closure inlines into the helper's specialised
# body, so Stan-side `reduce_sum` only sees a regular function pointer.
@slic (;n=5) begin
    mu ~ std_normal(;n=n)
    s  = simple_reduce_sum((x) -> x * x, mu)
end
