@slic (;y=abs.(randn(10)).+0.1) begin
    nu ~ gamma(2, 1)
    y ~ chi_square(nu)
end
