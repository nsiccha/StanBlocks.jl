# Hardening (crowdsource.qmd pattern): the 916f100 splice→Base.merge migration
# overlooked crowdsource.qmd, whose composition CHAINS `Base.merge` over
# SEPARATELY-BOUND `quote` transforms doing ASSIGNMENT (`=`) overrides —
#   a = Base.merge(full, a_transform); ab = Base.merge(a, b_transform); …
# (the old positional `a = full(a_transform)` now errors, _submodel_positional_error).
# Each transform is a variable-bound quote; merges chain (each returns a NEW
# SlicModel, base untouched); every override must land. Expected transformed_data:
#   lambda = rep_vector(0, I)   (overridden from 1)
#   delta  = rep_vector(1, I)   (overridden from 2)
full = @slic begin
    lambda = rep_vector(1, I)
    delta  = rep_vector(2, I)
    theta ~ std_normal(; n=I)
    return theta .* lambda .* delta
end
a_transform = quote
    lambda = rep_vector(0, I)
end
b_transform = quote
    delta = rep_vector(1, I)
end
a  = Base.merge(full, a_transform)
ab = Base.merge(a, b_transform)
@slic (; I=3, y=[1.0, 2.0, 3.0]) begin
    mu ~ ab(; I)
    y ~ normal(mu, 1)
end
