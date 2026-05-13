functions {
vector std_normal_vector_rng(
    int anontok__1
) {
    int n = anontok__1;
    return to_vector(normal_rng(rep_vector(0, n), 1));
}
real loopy(
    vector x
) {
    int n = dims(x)[1];
    real rv = 0.0;
    for(i in 1:n) {
        real sq__il_8 = (x[i] * x[i]);
        rv += (sq__il_8 + 1);
    }
    return rv;
}
}
data {
}
transformed data {
}
parameters {
}
transformed parameters {
}
model {
}
generated quantities {
    vector[3] mu = std_normal_vector_rng(3);
    real s = loopy(mu);
}