# An integer outer shape is the 1-D shorthand and preserves vector[N] storage.
@slic (; y = randn(6)) begin
    theta ~ plate(y; outer = 6) do yi
        z ~ normal(0.0, 1.0)
        yi ~ normal(z, 1.0)
        z
    end
end
