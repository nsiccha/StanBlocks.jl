functions {
vector gumbel_lpdfs(
    vector obs,
    real mu,
    real beta
) {
    int n = dims(obs)[1];
    return jbroadcasted_gumbel_lpdfs(obs, mu, beta);
}
vector jbroadcasted_gumbel_lpdfs(
    vector x1,
    real x2,
    real x3
) {
    int n = dims(x1)[1];
    vector[n] rv;
    for(i in 1:n) {
        rv[i] = gumbel_lpdfs(broadcasted_getindex(x1, i), broadcasted_getindex(x2, i), broadcasted_getindex(x3, i));
    }
    return rv;
}
real gumbel_lpdfs(
    real args1,
    real args2,
    real args3
) {
    return gumbel_lpdf(args1 | args2, args3);
}
real broadcasted_getindex(vector x, int i) {
    int m = dims(x)[1];
    return x[i];
}
real broadcasted_getindex(real x, int i) {
    return x;
}
vector gumbel_vector_rng(
    int anontok__1,
    real a,
    real b
) {
    int n = anontok__1;
    return to_vector(gumbel_rng(rep_vector(a, n), b));
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
    real<lower=0.0> beta;
}
transformed parameters {
}
model {
    mu ~ normal(0, 10);
    beta ~ gamma(1, 1);
    y ~ gumbel(mu, beta);
}
generated quantities {
    vector[y_n] y_likelihood = gumbel_lpdfs(y, mu, beta);
    vector[y_n] y_gen = gumbel_vector_rng(y_n, mu, beta);
}