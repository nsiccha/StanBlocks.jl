# Inline UDF called from inside a for-loop body. The hoisted intermediate
# `sq__il_<n>` lands INSIDE the loop body (per-block scoping), so each
# iteration declares its own fresh local.
@deffun @inline polish(x::real)::real = begin
    sq = x * x
    return sq + 1
end

@deffun loopy(x::vector[n])::real = begin
    rv = 0.
    for i in 1:n
        rv += polish(x[i])
    end
    return rv
end

@slic (;) begin
    mu ~ std_normal(;n=3)
    s  = loopy(mu)
end