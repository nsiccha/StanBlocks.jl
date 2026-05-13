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
    int log_radon_n;
    vector[log_radon_n] log_radon;
    int floor_measure_n;
    vector[floor_measure_n] floor_measure;
}
transformed data {
}
parameters {
    real alpha;
    real beta;
    real sigma_y;
}
transformed parameters {
}
model {
    alpha ~ normal(0, 10);
    beta ~ normal(0, 10);
    sigma_y ~ normal(0, 1);
    log_radon ~ normal((alpha + (beta * floor_measure)), sigma_y);
}
generated quantities {
    vector[log_radon_n] log_radon_likelihood = normal_lpdfs(log_radon, (alpha + (beta * floor_measure)), sigma_y);
    vector[log_radon_n] log_radon_gen = normal_vector_rng(log_radon_n, (alpha + (beta * floor_measure)), sigma_y);
}