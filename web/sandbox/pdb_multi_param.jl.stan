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
    int x1_n;
    vector[x1_n] x1;
    int x2_n;
    vector[x2_n] x2;
    int x3_n;
    vector[x3_n] x3;
}
transformed data {
}
parameters {
    real alpha;
    real b1;
    real b2;
    real b3;
    real<lower=0.0> sigma;
}
transformed parameters {
}
model {
    alpha ~ normal(0, 10);
    b1 ~ normal(0, 10);
    b2 ~ normal(0, 10);
    b3 ~ normal(0, 10);
    sigma ~ gamma(2, 1);
    y ~ normal((alpha + (b1 * x1) + (b2 * x2) + (b3 * x3)), sigma);
}
generated quantities {
    vector[y_n] y_likelihood = normal_lpdfs(y, (alpha + (b1 * x1) + (b2 * x2) + (b3 * x3)), sigma);
    vector[y_n] y_gen = normal_vector_rng(y_n, (alpha + (b1 * x1) + (b2 * x2) + (b3 * x3)), sigma);
}