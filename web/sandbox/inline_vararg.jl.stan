functions {
real normal_lpdfs(
    real args1,
    real args2,
    real args3
) {
    return normal_lpdf(args1 | args2, args3);
}
real three(real a, real b, real c) {
    return (a + b + c);
}
}
data {
    real y;
}
transformed data {
}
parameters {
    real a;
    real b;
    real c;
}
transformed parameters {
}
model {
    a ~ std_normal();
    b ~ std_normal();
    c ~ std_normal();
    y ~ normal(three(a, b, c), 1.0);
}
generated quantities {
    real y_likelihood = normal_lpdfs(y, three(a, b, c), 1.0);
    real y_gen = normal_rng(three(a, b, c), 1.0);
}