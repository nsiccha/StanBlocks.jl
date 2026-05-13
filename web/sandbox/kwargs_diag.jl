# Diagnostic — inspect the inline_body record for `foo` after registration.
@deffun foo(x::real; sigma::real=1.0)::real = sigma * x

# `StanBlocks` is `Main.StanBlocks` in the sandbox eval context.
ce = Main.StanBlocks.CanonicalExpr(foo, Main.StanBlocks.StanExpr(:x, Main.StanBlocks.StanType(Main.StanBlocks.types.real)))
meta = Main.StanBlocks.inline_body(ce)
@info "inline_body" arg_names=meta.arg_names kwargs=meta.kwargs
println("body:")
dump(meta.body; maxdepth=4)
println("body.args[2] (the call):")
dump(meta.body.args[2]; maxdepth=3)

@slic (;) begin
    a ~ std_normal()
    b = foo(a)
end
