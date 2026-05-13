@deffun begin
   my_complex_function5(f, g, h) = f + g * h + fgh
   my_complex_function4(f, g, h) = my_complex_function5(f, g, h)
   my_complex_function3(f, g, h) = my_complex_function4(f, g, h)
   my_complex_function2(f, g, h) = my_complex_function3(f, g, h)
   my_complex_function1(f, g, h) = my_complex_function2(f, g, h)
end

@slic (;y=randn(10)) begin
    mu ~ normal(0, 1)
    sigma ~ gamma(1, 1)
    y ~ normal(my_complex_function1(mu, sigma, mu), sigma)
end