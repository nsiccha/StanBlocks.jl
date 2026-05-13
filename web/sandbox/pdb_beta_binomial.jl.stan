functions {
vector beta_binomial_lpmfs(
    array[] int obs,
    array[] int trials,
    real alpha,
    real beta
) {
    int n = dims(obs)[1];
    return jbroadcasted_beta_binomial_lpmfs(obs, trials, alpha, beta);
}
vector jbroadcasted_beta_binomial_lpmfs(
    array[] int x1,
    array[] int x2,
    real x3,
    real x4
) {
    int n = dims(x1)[1];
    vector[n] rv;
    for(i in 1:n) {
        rv[i] = beta_binomial_lpmfs(
            broadcasted_getindex(x1, i),
            broadcasted_getindex(x2, i),
            broadcasted_getindex(x3, i),
            broadcasted_getindex(x4, i)
        );
    }
    return rv;
}
real beta_binomial_lpmfs(
    int args1,
    int args2,
    real args3,
    real args4
) {
    return beta_binomial_lpmf(args1 | args2, args3, args4);
}
int broadcasted_getindex(array[] int x, int i) {
    int m = dims(x)[1];
    return x[i];
}
real broadcasted_getindex(real x, int i) {
    return x;
}
array[] int beta_binomial_int_rng(
    int anontok__1,
    array[] int N,
    real a,
    real b
) {
    int n = anontok__1;
    if (dims(N)[1] != n) reject("beta_binomial_rng: dim mismatch — `N` dim 1 (= ", dims(N)[1], ") does not match `n` (= ", n, ")");
    return beta_binomial_rng(N, a, b);
}
}
data {
    int k_n;
    array[k_n] int k;
    int n_n;
    array[n_n] int n;
}
transformed data {
}
parameters {
    real<lower=0.0> alpha;
    real<lower=0.0> beta_param;
}
transformed parameters {
}
model {
    alpha ~ gamma(2, 1);
    beta_param ~ gamma(2, 1);
    k ~ beta_binomial(n, alpha, beta_param);
}
generated quantities {
    vector[k_n] k_likelihood = beta_binomial_lpmfs(k, n, alpha, beta_param);
    array[n_n] int k_gen = beta_binomial_int_rng(k_n, n, alpha, beta_param);
}