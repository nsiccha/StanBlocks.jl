functions {
vector bernoulli_logit_lpmfs(
    array[] int obs,
    vector args1
) {
    int n = dims(obs)[1];
    return jbroadcasted_bernoulli_logit_lpmfs(obs, args1);
}
vector jbroadcasted_bernoulli_logit_lpmfs(
    array[] int x1,
    vector x2
) {
    int n = dims(x1)[1];
    vector[n] rv;
    for(i in 1:n) {
        rv[i] = bernoulli_logit_lpmfs(broadcasted_getindex(x1, i), broadcasted_getindex(x2, i));
    }
    return rv;
}
real bernoulli_logit_lpmfs(
    int args1,
    real args2
) {
    return bernoulli_logit_lpmf(args1 | args2);
}
int broadcasted_getindex(array[] int x, int i) {
    int m = dims(x)[1];
    return x[i];
}
real broadcasted_getindex(vector x, int i) {
    int m = dims(x)[1];
    return x[i];
}
array[] int bernoulli_logit_int_rng(
    int anontok__1,
    vector p
) {
    int n = anontok__1;
    return bernoulli_logit_rng(p);
}
}
data {
    int switched_n;
    array[switched_n] int switched;
    int dist_n;
    vector[dist_n] dist;
}
transformed data {
}
parameters {
    real beta;
    real beta2;
}
transformed parameters {
}
model {
    beta ~ normal(0, 10);
    beta2 ~ normal(0, 10);
    switched ~ bernoulli_logit((beta + (beta2 * dist)));
}
generated quantities {
    vector[switched_n] switched_likelihood = bernoulli_logit_lpmfs(switched, (beta + (beta2 * dist)));
    array[switched_n] int switched_gen = bernoulli_logit_int_rng(switched_n, (beta + (beta2 * dist)));
}