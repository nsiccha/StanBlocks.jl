@slic affine(a, b) = begin
    z ~ normal(0, 1)
    return a + b * z
end
@slic (; p=1.0, q=2.0, y=[1.0, 2.0]) begin
    w ~ affine(p, q)
    y ~ normal(w, 1)
end
