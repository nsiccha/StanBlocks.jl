# Top-level model docstring. A leading string literal inside the `@slic`
# body is captured as the model docstring and rendered as a `// ...`
# comment header in the generated Stan code.
@slic (;y=randn(20), x=randn(20)) begin
    """
    Simple linear regression: y ~ Normal(alpha + beta * x, sigma).
    """
    alpha ~ std_normal()
    beta  ~ std_normal()
    sigma ~ std_normal(;lower=0.)
    y     ~ normal(alpha + beta * x, sigma)
end