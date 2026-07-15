# A vector[K] cell plus outer=(M,N) has logical K×M×N shape. Stan stores the
# extra N axis as array[N] matrix[K,M]; both loop indices remain available.
@slic (; y = randn(2, 3)) begin
    theta ~ plate(y; outer = (2, 3)) do yi
        z::vector[4] ~ std_normal()
        yi ~ normal(z[1], 1.0)
        z
    end
end
