# Piece 3: statement-splice now goes through `Base.merge`, not a positional call.
# `base` is an anonymous sub-model; `Base.merge(base, quote … end)` overrides the
# `theta` statement. (Positional `base(quote … end)` now errors; `base(; kwargs)`
# still feeds data.)
base = @slic begin
    theta ~ normal(0, 1)
    return theta
end
merged = Base.merge(base, quote
    theta ~ normal(0, 2)
end)
@slic (; y=[1.0, 2.0, 3.0]) begin
    mu ~ merged
    y ~ normal(mu, 1)
end
