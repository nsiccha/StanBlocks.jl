using StanBlocks

# Distance and ball/cup clearance are in feet; counts are from Berry (1996).
golf_data = (;
    distance = Float64.(2:20),
    attempts = [1443, 694, 455, 353, 272, 256, 240, 217, 200, 237,
                202, 192, 174, 167, 201, 195, 191, 147, 152],
    sunk = [1346, 577, 337, 208, 149, 136, 111, 69, 67, 75,
            52, 46, 54, 28, 27, 31, 33, 20, 24],
    gap = (4.25 - 1.68) / 24,
)

const angle_error = @slic begin
    sigma ~ normal(0, 0.1; lower=0)
    tolerance = asin(gap ./ distance)
    p = 2 * Phi(tolerance / sigma) - 1
    sunk ~ binomial(attempts, p)
    sigma_degrees = sigma * 180 / pi
end
model = angle_error(; golf_data...)
