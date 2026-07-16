base = @slic begin
    theta ~ normal(loc, 1)
    return theta
end
@slic (; loc=0.5, y=[1.0, 2.0]) begin
    mu ~ base(; loc)
    y ~ normal(mu, 1)
end
