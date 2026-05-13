functions {
vector build_eta(
    real alpha,
    vector beta,
    matrix X
) {
    int k = dims(beta)[1];
    int n = dims(X)[1];
    if (dims(X)[2] != k) reject("build_eta: dim mismatch — `X` dim 2 (= ", dims(X)[2], ") does not match `k` (= ", k, ")");
    vector[n] eta = rep_vector(alpha, n);
    vector[n] smooth = rep_vector(0.0, n);
    for(i__il_6 in 1:n) {
        smooth[i__il_6] = softplus(eta[i__il_6]);
    }
    return (eta + smooth);
}
real softplus(real x) {
    return log1p(exp(x));
}
vector normal_lpdfs(
    vector obs,
    vector loc,
    real scale
) {
    int n = dims(obs)[1];
    return jbroadcasted_normal_lpdfs(obs, loc, scale);
}
vector jbroadcasted_normal_lpdfs(
    vector x1,
    vector x2,
    real x3
) {
    int n = dims(x1)[1];
    vector[n] rv;
    for(i in 1:n) {
        rv[i] = normal_lpdfs(broadcasted_getindex(x1, i), broadcasted_getindex(x2, i), broadcasted_getindex(x3, i));
    }
    return rv;
}
real normal_lpdfs(
    real args1,
    real args2,
    real args3
) {
    return normal_lpdf(args1 | args2, args3);
}
real broadcasted_getindex(vector x, int i) {
    int m = dims(x)[1];
    return x[i];
}
real broadcasted_getindex(real x, int i) {
    return x;
}
vector normal_vector_rng(
    int anontok__1,
    vector a,
    real b
) {
    int n = anontok__1;
    return to_vector(normal_rng(a, b));
}
}
data {
    int X_m;
    int X_n;
    matrix[X_m, X_n] X;
    int y_n;
    vector[y_n] y;
}
transformed data {
}
parameters {
    real alpha;
    vector[3] beta;
    real<lower=0.0> sigma;
}
transformed parameters {
    vector[X_m] mu = build_eta(alpha, beta, X);
}
model {
    alpha ~ std_normal();
    beta ~ std_normal();
    sigma ~ std_normal();
    y ~ normal(mu, sigma);
}
generated quantities {
    vector[y_n] y_likelihood = normal_lpdfs(y, mu, sigma);
    vector[y_n] y_gen = normal_vector_rng(y_n, mu, sigma);
}