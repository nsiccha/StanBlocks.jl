functions {
vector exponential_lpdfs(
    vector obs,
    real rate
) {
    int n = dims(obs)[1];
    return jbroadcasted_exponential_lpdfs(obs, rate);
}
vector jbroadcasted_exponential_lpdfs(
    vector x1,
    real x2
) {
    int n = dims(x1)[1];
    vector[n] rv;
    for(i in 1:n) {
        rv[i] = exponential_lpdfs(broadcasted_getindex(x1, i), broadcasted_getindex(x2, i));
    }
    return rv;
}
real exponential_lpdfs(real args1, real args2) {
    return exponential_lpdf(args1 | args2);
}
real broadcasted_getindex(vector x, int i) {
    int m = dims(x)[1];
    return x[i];
}
real broadcasted_getindex(real x, int i) {
    return x;
}
vector exponential_vector_rng(
    int anontok__1,
    real a
) {
    int n = anontok__1;
    return to_vector(exponential_rng(rep_vector(a, n)));
}
}
data {
    int t_n;
    vector[t_n] t;
}
transformed data {
}
parameters {
    real<lower=0.0> lambda;
}
transformed parameters {
}
model {
    lambda ~ gamma(1, 1);
    t ~ exponential(lambda);
}
generated quantities {
    vector[t_n] t_likelihood = exponential_lpdfs(t, lambda);
    vector[t_n] t_gen = exponential_vector_rng(t_n, lambda);
}