functions {
real safe_log(
    real x
) {
    if(!((x > 0))) {
        reject("safe_log: argument must be positive");
    }
    return log(x);
}
real safe_div(
    real a,
    real b
) {
    if(!((b != 0))) {
        reject("assertion failed: b != 0");
    }
    return (a / b);
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
    real<lower=0.0> mu;
    real nu;
}
transformed parameters {
    real foo = safe_log(mu);
    real bar = safe_div(mu, nu);
}
model {
    mu ~ std_normal();
    nu ~ std_normal();
    y ~ normal((foo + bar), 1.0);
}
generated quantities {
    vector[y_n] y_likelihood = normal_lpdfs(y, (foo + bar), 1.0);
    vector[y_n] y_gen = normal_vector_rng(y_n, (foo + bar), 1.0);
}