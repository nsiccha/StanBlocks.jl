functions {
real normal_lpdfs(
    real args1,
    real args2,
    real args3
) {
    return normal_lpdf(args1 | args2, args3);
}
}
data {
    real y;
}
transformed data {
}
parameters {
    real mu;
}
transformed parameters {
}
model {
    mu ~ std_normal();
    y ~ normal((-mu), 1.0);
}
generated quantities {
    real y_likelihood = normal_lpdfs(y, (-mu), 1.0);
    real y_gen = normal_rng((-mu), 1.0);
}