# Ragged simplex: 3 groups of dims K=[2,3,4] → simplex[2], simplex[3], simplex[4].
# Stan cannot declare `simplex[K]` natively; SB desugars to a flat improper-uniform
# free param + per-group `simplex_jacobian` constrain loop (injected TP for-loop) +
# a `RaggedVector` pairing. RHS informative prior deferred (scope decision 1mfltua).
@slic (; K = [2, 3, 4], y = 0.3) begin
    p::simplex[K] ~ flat()
    y ~ normal(sum(p[1]), 0.1)
end
