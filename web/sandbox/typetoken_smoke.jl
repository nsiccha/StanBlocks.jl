# Smoke test for TypeTokenExpr creation paths:
#  - bare Stan type at call site                       `type_demo(real)`
#  - `T[dims...]` at call site (sized token via getindex) `type_demo(real[n])`
#  - `typeof(x)` at call site (derived token)            `type_demo(typeof(y))`
#
# `type_demo` is a no-op returning 0 of the requested shape. The intent here is
# purely to verify that each type-token form parses, traces, and renders; the
# Stan function body should reference the token's dims via the tuple-packed arg.

@deffun type_demo(real) = 0.0
@deffun type_demo(real[n]) = rep_vector(0.0, n)
@deffun type_demo(vector[n]) = rep_vector(0.0, n)

@slic (; y = randn(5)) begin
    a = type_demo(real)
    v = type_demo(real[5])
    w = type_demo(typeof(y))
    y ~ normal(a + v + w, 1.0)
end
