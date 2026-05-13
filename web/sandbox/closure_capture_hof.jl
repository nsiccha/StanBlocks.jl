# Capture + HOF: closure carries a captured `shift` into a SLIC-side HOF
# wrapper. After specialisation, the helper's Stan body uses `shift`
# directly (it's a model parameter, so `shift` is a Stan parameter ref).
@slic (;n=5) begin
    shift ~ std_normal()
    mu    ~ std_normal(;n=n)
    s     = simple_reduce_sum((x) -> x + shift, mu)
end
