@deffun foo(x::real, y::real = 2.0)::real = x + y
@slic (;) begin
    a ~ std_normal()
    b = foo(a)
    c = foo(a, 5.0)
end