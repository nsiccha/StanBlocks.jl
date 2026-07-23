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
    vector[2] free;
}
transformed parameters {
    vector[(2 + 1)] theta = simplex_jacobian(free);
}
model {
    free ~ std_normal();
    y ~ normal(theta[1], 0.1);
}
generated quantities {
    real y_likelihood = normal_lpdfs(y, theta[1], 0.1);
    real y_gen = normal_rng(theta[1], 0.1);
}