functions {
real kwcall_foo(
    tuple(real, real) kw,
    real x
) {
    real sigma = kw.1;
    real alpha = kw.2;
    return ((sigma * x) + alpha);
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
    real b = kwcall_foo((1.0, 2.0), a);
    real c = kwcall_foo((3.0, 2.0), a);
    real d = kwcall_foo((3.0, 4.0), a);
}