functions {
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
    int y_n;
    vector[y_n] y;
    int x_n;
    vector[x_n] x;
}
transformed data {
}
parameters {
    real beta;
    real<lower=0.0> sigma;
}
transformed parameters {
}
model {
    beta ~ normal(0, 1);
    sigma ~ gamma(1, 1);
    y ~ normal((x * beta), sigma);
}
generated quantities {
    vector[y_n] y_likelihood = normal_lpdfs(y, (x * beta), sigma);
    vector[y_n] y_gen = normal_vector_rng(y_n, (x * beta), sigma);
}