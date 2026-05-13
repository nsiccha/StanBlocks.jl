functions {
vector skew_normal_lpdfs(
    vector obs,
    real mu,
    real sigma,
    real alpha
) {
    int n = dims(obs)[1];
    return jbroadcasted_skew_normal_lpdfs(obs, mu, sigma, alpha);
}
vector jbroadcasted_skew_normal_lpdfs(
    vector x1,
    real x2,
    real x3,
    real x4
) {
    int n = dims(x1)[1];
    vector[n] rv;
    for(i in 1:n) {
        rv[i] = skew_normal_lpdfs(
            broadcasted_getindex(x1, i),
            broadcasted_getindex(x2, i),
            broadcasted_getindex(x3, i),
            broadcasted_getindex(x4, i)
        );
    }
    return rv;
}
real skew_normal_lpdfs(
    real args1,
    real args2,
    real args3,
    real args4
) {
    return skew_normal_lpdf(args1 | args2, args3, args4);
}
real broadcasted_getindex(vector x, int i) {
    int m = dims(x)[1];
    return x[i];
}
real broadcasted_getindex(real x, int i) {
    return x;
}
vector skew_normal_vector_rng(
    int anontok__1,
    real nu,
    real a,
    real b
) {
    int n = anontok__1;
    return to_vector(skew_normal_rng(nu, rep_vector(a, n), b));
}
}
data {
    int y_n;
    vector[y_n] y;
}
transformed data {
}
parameters {
    real mu;
    real<lower=0.0> sigma;
    real alpha;
}
transformed parameters {
}
model {
    mu ~ normal(0, 10);
    sigma ~ gamma(2, 1);
    alpha ~ normal(0, 5);
    y ~ skew_normal(mu, sigma, alpha);
}
generated quantities {
    vector[y_n] y_likelihood = skew_normal_lpdfs(y, mu, sigma, alpha);
    vector[y_n] y_gen = skew_normal_vector_rng(y_n, mu, sigma, alpha);
}