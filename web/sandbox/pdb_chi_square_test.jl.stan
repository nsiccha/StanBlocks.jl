functions {
vector chi_square_lpdfs(
    vector obs,
    real nu
) {
    int n = dims(obs)[1];
    return jbroadcasted_chi_square_lpdfs(obs, nu);
}
vector jbroadcasted_chi_square_lpdfs(
    vector x1,
    real x2
) {
    int n = dims(x1)[1];
    vector[n] rv;
    for(i in 1:n) {
        rv[i] = chi_square_lpdfs(broadcasted_getindex(x1, i), broadcasted_getindex(x2, i));
    }
    return rv;
}
real chi_square_lpdfs(real args1, real args2) {
    return chi_square_lpdf(args1 | args2);
}
real broadcasted_getindex(vector x, int i) {
    int m = dims(x)[1];
    return x[i];
}
real broadcasted_getindex(real x, int i) {
    return x;
}
vector chi_square_vector_rng(
    int anontok__1,
    real a
) {
    int n = anontok__1;
    return to_vector(chi_square_rng(rep_vector(a, n)));
}
}
data {
    int y_n;
    vector[y_n] y;
}
transformed data {
}
parameters {
    real<lower=0.0> nu;
}
transformed parameters {
}
model {
    nu ~ gamma(2, 1);
    y ~ chi_square(nu);
}
generated quantities {
    vector[y_n] y_likelihood = chi_square_lpdfs(y, nu);
    vector[y_n] y_gen = chi_square_vector_rng(y_n, nu);
}