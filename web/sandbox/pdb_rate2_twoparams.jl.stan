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
    int k1;
    int n1;
    int k2;
    int n2;
}
transformed data {
}
parameters {
    real<lower=0, upper=1> theta1;
    real<lower=0, upper=1> theta2;
}
transformed parameters {
}
model {
    theta1 ~ beta(1, 1);
    theta2 ~ beta(1, 1);
    k1 ~ binomial(n1, theta1);
    k2 ~ binomial(n2, theta2);
}
generated quantities {
    real k1_likelihood = binomial_lpmfs(k1, n1, theta1);
    int k1_gen = binomial_rng(n1, theta1);
    real k2_likelihood = binomial_lpmfs(k2, n2, theta2);
    int k2_gen = binomial_rng(n2, theta2);
}