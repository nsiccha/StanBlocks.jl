functions {
vector std_normal_vector_rng(
    int anontok__1
) {
    int n = anontok__1;
    return to_vector(normal_rng(rep_vector(0, n), 1));
}
real simple_reduce_sum_closure_8(
    vector x
) {
    int n = dims(x)[1];
    return reduce_sum(simple_reduce_sum_helper_closure_8, to_array_1d(x), 1);
}
real simple_reduce_sum_helper_closure_8(
    array[] real x_slice,
    int slice_start,
    int slice_end
) {
    int n = dims(x_slice)[1];
    real rv = 0.0;
    for(i in 1:n) {
        rv += (x_slice[i] * x_slice[i]);
    }
    return rv;
}
}
data {
    int n;
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
    vector[n] mu = std_normal_vector_rng(n);
    real s = simple_reduce_sum_closure_8(mu);
}