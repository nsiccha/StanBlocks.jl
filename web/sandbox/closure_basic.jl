# Closure phase 1 — anonymous lambda with no captures, called via an
# `@inline` UDF that takes a function-typed param. The closure body fully
# inlines into the receiver's specialised Stan function.
@deffun @inline apply(f, x)::real = f(x)

@slic (;) begin
    a ~ std_normal()
    sq = (x) -> x * x
    b = apply(sq, a)
end
