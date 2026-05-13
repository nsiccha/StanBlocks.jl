functions {
vector ordered_logistic_lpmfs(
    array[] int y,
    vector eta,
    vector c
) {
    int n = dims(y)[1];
    int m = dims(c)[1];
    if (dims(eta)[1] != n) reject("ordered_logistic_lpmfs: dim mismatch — `eta` dim 1 (= ", dims(eta)[1], ") does not match `n` (= ", n, ")");
    vector[n] rv;
    for(i in 1:n) {
        rv[i] = ordered_logistic_lpmf(y[i] | eta[i], c);
    }
    return rv;
}
array[] int ordered_logistic_int_rng(
    int anontok__1,
    vector eta,
    vector c
) {
    int n = anontok__1;
    int m = dims(c)[1];
    if (dims(eta)[1] != n) reject("ordered_logistic_rng: dim mismatch — `eta` dim 1 (= ", dims(eta)[1], ") does not match `n` (= ", n, ")");
    return ordered_logistic_rng(eta, c);
}
array[] int ordered_logistic_rng(
    vector eta,
    vector c
) {
    int n = dims(eta)[1];
    int m = dims(c)[1];
    array[n] int rv;
    for(i in 1:n) {
        rv[i] = ordered_logistic_rng(eta[i], c);
    }
    return rv;
}
}
data {
    int y_n;
    array[y_n] int y;
    int n_cuts;
}
transformed data {
}
parameters {
    vector[num_elements(y)] eta;
    real c_base;
    vector[(n_cuts - 1)] c_log_incr;
}
transformed parameters {
    vector[((n_cuts - 1) + 1)] cuts = (c_base + cumulative_sum(append_row(0.0, exp(c_log_incr))));
}
model {
    eta ~ std_normal();
    c_base ~ std_normal();
    c_log_incr ~ std_normal();
    y ~ ordered_logistic(eta, cuts);
}
generated quantities {
    vector[num_elements(y)] y_likelihood = ordered_logistic_lpmfs(y, eta, cuts);
    array[num_elements(y)] int y_gen = ordered_logistic_int_rng(y_n, eta, cuts);
}