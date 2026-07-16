@slic linpred(slope) = begin
    intercept ~ normal(0, 1)
    return intercept + slope
end
@slic (; x=0.7, y=[1.0, 2.0, 3.0]) begin
    mu ~ linpred(x)
    y ~ normal(mu, 1)
end
