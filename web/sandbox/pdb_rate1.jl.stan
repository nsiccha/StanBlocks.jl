functions {
real binomial_lpmfs(
    int args1,
    int args2,
    real args3
) {
    return binomial_lpmf(args1 | args2, args3);
}
}
data {
    int k;
    int n;
}
transformed data {
}
parameters {
    real<lower=0, upper=1> theta;
}
transformed parameters {
}
model {
    theta ~ beta(1, 1);
    k ~ binomial(n, theta);
}
generated quantities {
    real k_likelihood = binomial_lpmfs(k, n, theta);
    int k_gen = binomial_rng(n, theta);
}