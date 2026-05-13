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
    int earn_n;
    vector[earn_n] earn;
    int height_n;
    vector[height_n] height;
}
transformed data {
}
parameters {
    real beta;
    real beta2;
    real<lower=0.0> sigma;
}
transformed parameters {
}
model {
    beta ~ normal(0, 10);
    beta2 ~ normal(0, 10);
    sigma ~ gamma(2, 1);
    earn ~ normal((beta + (beta2 * height)), sigma);
}
generated quantities {
    vector[earn_n] earn_likelihood = normal_lpdfs(earn, (beta + (beta2 * height)), sigma);
    vector[earn_n] earn_gen = normal_vector_rng(earn_n, (beta + (beta2 * height)), sigma);
}