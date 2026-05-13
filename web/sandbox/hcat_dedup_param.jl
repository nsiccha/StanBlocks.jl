@slic (;a=randn(3), b=randn(5)) begin
    ftime ~ std_normal(; n=num_elements(a))
    X1 = hcat(rep_vector(1.0, num_elements(a)), ftime)
    X2 = hcat(rep_vector(1.0, num_elements(b)), ftime)
    y = X1
end