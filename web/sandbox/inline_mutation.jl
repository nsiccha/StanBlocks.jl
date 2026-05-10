# `f!` mutation. The trailing `!` only triggers inlining; nothing about
# mutation is special-cased — substitution + the existing assignment-form
# rules carry the rest. After inlining, `set_first!(out)` rewrites to the
# inline body acting on `out` directly, no Stan UDF emitted.
@deffun @inline set_first!(buf::vector[n])::vector[n] = begin
    buf[1] = 42.
    return buf
end

# Compound-assign mutation through a vector.
@deffun @inline accumulate!(rv::real, x::vector[n])::real = begin
    for i in 1:n
        rv += x[i]
    end
    return rv
end

@deffun build(seed::vector[n])::vector[n] = begin
    out = seed
    set_first!(out)            # mutates `out` in place
    return out
end

@deffun loopsum(x::vector[n])::real = begin
    s = 0.
    accumulate!(s, x)          # mutates `s` via compound assignment
    return s
end

@slic (;) begin
    mu  ~ std_normal(;n=4)
    arr = build(mu)
    s   = loopsum(mu)
end