# Closure with one capture: `shift` is a model-scope StanExpr at the moment
# the lambda is constructed. `forward!(:->)` snapshots it; `expand_inline!`
# substitutes it into the body before re-tracing.
@deffun @inline apply(f, x)::real = f(x)

@slic (;) begin
    shift ~ std_normal()
    a     ~ std_normal()
    add_shift = (x) -> x + shift
    b = apply(add_shift, a)
end
