functions {
real foo(real x, real y) {
    return (x + y);
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
    real b = foo(a, 2.0);
    real c = foo(a, 5.0);
}