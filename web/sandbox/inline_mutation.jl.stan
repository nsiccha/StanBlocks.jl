functions {
vector std_normal_vector_rng(
    int anontok__1
) {
    int n = anontok__1;
    return to_vector(normal_rng(rep_vector(0, n), 1));
}
vector build(
    vector seed
) {
    int n = dims(seed)[1];
    vector["dims(seed)[1]"] out = seed;
    out[1] = 42.0;
    return out;
}
real loopsum(
    vector x
) {
    int n = dims(x)[1];
    real s = 0.0;
    for(i__il_9 in 1:n) {
        s += x[i__il_9];
    }
    return s;
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
    vector[4] mu = std_normal_vector_rng(4);
    vector[4] arr = build(mu);
    real s = loopsum(mu);
}