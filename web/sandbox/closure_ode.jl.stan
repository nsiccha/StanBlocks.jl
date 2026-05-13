functions {
// lifted closure (id 9)
vector closure_9(
    real t,
    vector y_state,
    real lambda
) {
    return ((-lambda) * y_state);
}
}
data {
    int ts_n;
    vector[ts_n] ts;
}
transformed data {
    vector[1] y0 = [1.0]';
}
parameters {
}
transformed parameters {
}
model {
}
generated quantities {
    real lambda = std_normal_rng();
    array[ts_n] vector[1] y = ode_rk45(closure_9, y0, 0.0, to_array_1d(ts), lambda);
}