functions {
vector weibull_lpdfs(
    vector obs,
    real alpha,
    real sigma
) {
    int n = dims(obs)[1];
    return jbroadcasted_weibull_lpdfs(obs, alpha, sigma);
}
vector jbroadcasted_weibull_lpdfs(
    vector x1,
    real x2,
    real x3
) {
    int n = dims(x1)[1];
    vector[n] rv;
    for(i in 1:n) {
        rv[i] = weibull_lpdfs(broadcasted_getindex(x1, i), broadcasted_getindex(x2, i), broadcasted_getindex(x3, i));
    }
    return rv;
}
real weibull_lpdfs(
    real args1,
    real args2,
    real args3
) {
    return weibull_lpdf(args1 | args2, args3);
}
real broadcasted_getindex(vector x, int i) {
    int m = dims(x)[1];
    return x[i];
}
real broadcasted_getindex(real x, int i) {
    return x;
}
vector weibull_vector_rng(
    int anontok__1,
    real a,
    real b
) {
    int n = anontok__1;
    return to_vector(weibull_rng(rep_vector(a, n), b));
}
}
data {
    int t_n;
    vector[t_n] t;
}
transformed data {
}
parameters {
    real<lower=0.0> alpha;
    real<lower=0.0> sigma;
}
transformed parameters {
}
model {
    alpha ~ gamma(2, 1);
    sigma ~ gamma(2, 1);
    t ~ weibull(alpha, sigma);
}
generated quantities {
    vector[t_n] t_likelihood = weibull_lpdfs(t, alpha, sigma);
    vector[t_n] t_gen = weibull_vector_rng(t_n, alpha, sigma);
}