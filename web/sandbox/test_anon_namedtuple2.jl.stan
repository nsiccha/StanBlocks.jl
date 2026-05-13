functions {
vector normal_lpdfs(
    vector obs,
    real loc,
    int scale
) {
    int n = dims(obs)[1];
    return jbroadcasted_normal_lpdfs(obs, loc, scale);
}
vector jbroadcasted_normal_lpdfs(
    vector x1,
    real x2,
    int x3
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
    int args3
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
int broadcasted_getindex(int x, int i) {
    return x;
}
vector normal_vector_rng(
    int anontok__1,
    real a,
    int b
) {
    int n = anontok__1;
    return to_vector(normal_rng(rep_vector(a, n), b));
}
}
data {
    tuple(real, real) priors;
    int y_n;
    vector[y_n] y;
}
transformed data {
}
parameters {
    real mu;
}
transformed parameters {
}
model {
    mu ~ normal(priors.1, priors.2);
    y ~ normal(mu, 1);
}
generated quantities {
    vector[y_n] y_likelihood = normal_lpdfs(y, mu, 1);
    vector[y_n] y_gen = normal_vector_rng(y_n, mu, 1);
}