# Three outer axes exercise array-prefix storage and nested routing: scalar cells
# use array[P] matrix[M,N] while retaining logical M×N×P shape.
@slic (; y = randn(2, 3)) begin
    theta ~ plate(; outer = (2, 3, 4)) do i, j, k
        z ~ normal(0.0, 1.0)
        y[i, j] ~ normal(z, 1.0)
        z + 0.0 * k
    end
end
