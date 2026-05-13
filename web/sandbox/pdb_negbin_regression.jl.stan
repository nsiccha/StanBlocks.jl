functions {
vector neg_binomial_2_lpmfs(
    array[] int obs,
    vector mu,
    real phi
) {
    int n = dims(obs)[1];
    return jbroadcasted_neg_binomial_2_lpmfs(obs, mu, phi);
}
vector jbroadcasted_neg_binomial_2_lpmfs(
    array[] int x1,
    vector x2,
    real x3
) {
    int n = dims(x1)[1];
    vector[n] rv;
    for(i in 1:n) {
        rv[i] = neg_binomial_2_lpmfs(
            broadcasted_getindex(x1, i),
            broadcasted_getindex(x2, i),
            broadcasted_getindex(x3, i)
        );
    }
    return rv;
}
real neg_binomial_2_lpmfs(
    int args1,
    real args2,
    real args3
) {
    return neg_binomial_2_lpmf(args1 | args2, args3);
}
int broadcasted_getindex(array[] int x, int i) {
    int m = dims(x)[1];
    return x[i];
}
real broadcasted_getindex(vector x, int i) {
    int m = dims(x)[1];
    return x[i];
}
real broadcasted_getindex(real x, int i) {
    return x;
}
array[] int neg_binomial_2_int_rng(
    int anontok__1,
    vector a,
    real b
) {
    int n = anontok__1;
    return neg_binomial_2_rng(a, b);
}
}
data {
    int y_n;
    array[y_n] int y;
    int x_n;
    vector[x_n] x;
}
transformed data {
}
parameters {
    real alpha;
    real beta;
    real<lower=0.0> phi;
}
transformed parameters {
}
model {
    alpha ~ normal(0, 5);
    beta ~ normal(0, 5);
    phi ~ gamma(2, 1);
    y ~ neg_binomial_2(exp((alpha + (beta * x))), phi);
}
generated quantities {
    vector[y_n] y_likelihood = neg_binomial_2_lpmfs(y, exp((alpha + (beta * x))), phi);
    array[y_n] int y_gen = neg_binomial_2_int_rng(y_n, exp((alpha + (beta * x))), phi);
}