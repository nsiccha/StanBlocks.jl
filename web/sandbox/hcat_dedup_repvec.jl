@slic (;a=randn(3), b=randn(5)) begin
    X1 = hcat(rep_vector(1.0, num_elements(a)), a)
    X2 = hcat(rep_vector(1.0, num_elements(b)), b)
    y = X1
end