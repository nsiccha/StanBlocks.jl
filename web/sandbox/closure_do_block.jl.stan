functions {
real simple_reduce_sum_closure_7(
    real shift,
    array[] int x
) {
    return reduce_sum(simple_reduce_sum_helper_closure_7, x, 1, shift);
}
real simple_reduce_sum_helper_closure_7(
    array[] int x_slice,
    int slice_start,
    int slice_end,
    real shift
) {
    int n = dims(x_slice)[1];
    real rv = 0.0;
    for(i in 1:n) {
        rv += (shift * x_slice[i]);
    }
    return rv;
}
}
data {
    int idxs_n;
    array[idxs_n] int idxs;
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
    real shift = std_normal_rng();
    real s = simple_reduce_sum_closure_7(shift, idxs);
}