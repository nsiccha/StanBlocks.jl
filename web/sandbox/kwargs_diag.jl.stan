functions {
real kwcall_foo(tuple(real) kw, real x) {
    real sigma = kw.1;
    return (sigma * x);
}
}
data {
}
transformed data {
}
parameters {
}
transformed parameters {
}
model {
}
generated quantities {
    real a = std_normal_rng();
    real b = kwcall_foo((1.0), a);
}