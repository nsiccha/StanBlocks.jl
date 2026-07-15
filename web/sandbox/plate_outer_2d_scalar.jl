# Two mapped outer axes: matrix input cells and scalar fresh/results become M×N.
@slic (; y = randn(2, 3)) begin
    theta ~ plate(y; outer = (2, 3)) do yi
        z ~ normal(0.0, 1.0)
        yi ~ normal(z, 1.0)
        z
    end
end
