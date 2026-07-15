# Ragged positive-ordered vectors: group sizes K=[2,3,4]. Each group has K free
# coordinates, transformed per group and exposed as a RaggedVector.
@slic (; K = [2, 3, 4], y = 0.3) begin
    p::positive_ordered[K] ~ flat()
    y ~ normal(sum(p[1]), 0.1)
end
