functions {
vector double_exponential_lpdfs(
    vector obs,
    real mu,
    real sigma
) {
    int n = dims(obs)[1];
    return jbroadcasted_double_exponential_lpdfs(obs, mu, sigma);
}
vector jbroadcasted_double_exponential_lpdfs(
    vector x1,
    real x2,
    real x3
) {
    int n = dims(x1)[1];
    vector[n] rv;
    for(i in 1:n) {
        rv[i] = double_exponential_lpdfs(
            broadcasted_getindex(x1, i),
            broadcasted_getindex(x2, i),
            broadcasted_getindex(x3, i)
        );
    }
    return rv;
}
real double_exponential_lpdfs(
    real args1,
    real args2,
    real args3
) {
    return double_exponential_lpdf(args1 | args2, args3);
}
real broadcasted_getindex(vector x, int i) {
    int m = dims(x)[1];
    return x[i];
}
real broadcasted_getindex(real x, int i) {
    return x;
}
vector double_exponential_vector_rng(
    int anontok__1,
    real a,
    real b
) {
    int n = anontok__1;
    return to_vector(double_exponential_rng(rep_vector(a, n), b));
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
}
transformed parameters {
}
model {
    mu ~ normal(0, 10);
    sigma ~ gamma(2, 1);
    y ~ double_exponential(mu, sigma);
}
generated quantities {
    vector[y_n] y_likelihood = double_exponential_lpdfs(y, mu, sigma);
    vector[y_n] y_gen = double_exponential_vector_rng(y_n, mu, sigma);
}