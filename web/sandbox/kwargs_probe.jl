# Kwargs lowering probe: `f(x; sigma=1.0, alpha=2.0) = sigma*x + alpha`
# lowers to a canonical `Core.kwcall` body method + an `@inline` shim
# whose inline_body carries the kwarg names + defaults.
@deffun foo(x::real; sigma::real=1.0, alpha::real=2.0)::real = sigma * x + alpha

@slic (;) begin
    a ~ std_normal()
    b = foo(a)                              # all defaults
    c = foo(a; sigma=3.0)                   # one kwarg, alpha defaulted
    d = foo(a; sigma=3.0, alpha=4.0)        # all explicit
end
