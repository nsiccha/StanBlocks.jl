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
    int kid_score_n;
    vector[kid_score_n] kid_score;
    int mom_iq_n;
    vector[mom_iq_n] mom_iq;
}
transformed data {
}
parameters {
    real beta;
    real beta2;
    real sigma;
}
transformed parameters {
}
model {
    beta ~ normal(0, 10);
    beta2 ~ normal(0, 10);
    sigma ~ cauchy(0, 2.5);
    kid_score ~ normal((beta + (beta2 * mom_iq)), sigma);
}
generated quantities {
    vector[kid_score_n] kid_score_likelihood = normal_lpdfs(kid_score, (beta + (beta2 * mom_iq)), sigma);
    vector[kid_score_n] kid_score_gen = normal_vector_rng(kid_score_n, (beta + (beta2 * mom_iq)), sigma);
}