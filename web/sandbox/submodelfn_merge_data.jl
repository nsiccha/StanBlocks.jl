# Piece 4: chained statement-splice + data-kwargs — the one NEW combination the
# `base(quote…end; kwargs)` → `Base.merge(base, quote…end)(; kwargs)` migration
# introduces (real sites: `_simplex.qmd:144`, `test/runtests.jl:266/280`).
# `Base.merge(base, quote…end)` overrides the `theta` statement and returns a
# SlicModel; the trailing `(; scale=5.0)` then feeds the data introduced by that
# override. Both must compose — output should show `theta ~ normal(0, scale)`
# with `scale` bound as data = 5.0.
base = @slic begin
    theta ~ normal(0, 1)
    return theta
end
combined = Base.merge(base, quote
    theta ~ normal(0, scale)
end)(; scale=5.0)
@slic (; y=[1.0, 2.0, 3.0]) begin
    mu ~ combined
    y ~ normal(mu, 1)
end
