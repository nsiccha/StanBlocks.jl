functions {
vector student_t_lpdfs(
    vector obs,
    real nu,
    vector loc,
    real scale
) {
    int n = dims(obs)[1];
    return jbroadcasted_student_t_lpdfs(obs, nu, loc, scale);
}
vector jbroadcasted_student_t_lpdfs(
    vector x1,
    real x2,
    vector x3,
    real x4
) {
    int n = dims(x1)[1];
    vector[n] rv;
    for(i in 1:n) {
        rv[i] = student_t_lpdfs(
            broadcasted_getindex(x1, i),
            broadcasted_getindex(x2, i),
            broadcasted_getindex(x3, i),
            broadcasted_getindex(x4, i)
        );
    }
    return rv;
}
real student_t_lpdfs(
    real args1,
    real args2,
    real args3,
    real args4
) {
    return student_t_lpdf(args1 | args2, args3, args4);
}
real broadcasted_getindex(vector x, int i) {
    int m = dims(x)[1];
    return x[i];
}
real broadcasted_getindex(real x, int i) {
    return x;
}
vector student_t_vector_rng(
    int anontok__1,
    real nu,
    vector a,
    real b
) {
    int n = anontok__1;
    return to_vector(student_t_rng(nu, a, b));
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
    real alpha;
    real beta;
    real<lower=0.0> sigma;
    real<lower=0.0> nu;
}
transformed parameters {
}
model {
    alpha ~ normal(0, 10);
    beta ~ normal(0, 10);
    sigma ~ gamma(2, 1);
    nu ~ gamma(2, 0.1);
    y ~ student_t(nu, (alpha + (beta * x)), sigma);
}
generated quantities {
    vector[y_n] y_likelihood = student_t_lpdfs(y, nu, (alpha + (beta * x)), sigma);
    vector[y_n] y_gen = student_t_vector_rng(y_n, nu, (alpha + (beta * x)), sigma);
}