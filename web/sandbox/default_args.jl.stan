functions {
real lin(real x, real slope, real intercept) {
    return ((slope * x) + intercept);
}
vector normal_lpdfs(
    vector obs,
    real loc,
    real scale
) {
    int n = dims(obs)[1];
    return jbroadcasted_normal_lpdfs(obs, loc, scale);
}
vector jbroadcasted_normal_lpdfs(
    vector x1,
    real x2,
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
    real a,
    real b
) {
    int n = anontok__1;
    return to_vector(normal_rng(rep_vector(a, n), b));
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
}
transformed parameters {
    real a = lin(mu, 1.0, 0.0);
    real b = lin(mu, 2.0, 0.0);
    real c = lin(mu, 2.0, 0.5);
}
model {
    mu ~ std_normal();
    y ~ normal((a + b + c), 1.0);
}
generated quantities {
    vector[y_n] y_likelihood = normal_lpdfs(y, (a + b + c), 1.0);
    vector[y_n] y_gen = normal_vector_rng(y_n, (a + b + c), 1.0);
}