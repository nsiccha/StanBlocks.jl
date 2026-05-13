functions {
vector poisson_log_lpmfs(
    array[] int obs,
    vector alpha
) {
    int n = dims(obs)[1];
    return jbroadcasted_poisson_log_lpmfs(obs, alpha);
}
vector jbroadcasted_poisson_log_lpmfs(
    array[] int x1,
    vector x2
) {
    int n = dims(x1)[1];
    vector[n] rv;
    for(i in 1:n) {
        rv[i] = poisson_log_lpmfs(broadcasted_getindex(x1, i), broadcasted_getindex(x2, i));
    }
    return rv;
}
real poisson_log_lpmfs(int args1, real args2) {
    return poisson_log_lpmf(args1 | args2);
}
int broadcasted_getindex(array[] int x, int i) {
    int m = dims(x)[1];
    return x[i];
}
real broadcasted_getindex(vector x, int i) {
    int m = dims(x)[1];
    return x[i];
}
array[] int poisson_log_int_rng(
    int anontok__1,
    vector p
) {
    int n = anontok__1;
    return poisson_log_rng(p);
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
}
transformed parameters {
}
model {
    alpha ~ normal(0, 5);
    beta ~ normal(0, 5);
    y ~ poisson_log((alpha + (beta * x)));
}
generated quantities {
    vector[y_n] y_likelihood = poisson_log_lpmfs(y, (alpha + (beta * x)));
    array[y_n] int y_gen = poisson_log_int_rng(y_n, (alpha + (beta * x)));
}