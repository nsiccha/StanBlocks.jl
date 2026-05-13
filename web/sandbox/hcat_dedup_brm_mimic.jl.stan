functions {
vector std_normal_vector_rng(
    int anontok__1
) {
    int n = anontok__1;
    return to_vector(normal_rng(rep_vector(0, n), 1));
}
matrix hcat(
    vector x,
    vector y
) {
    int n = dims(x)[1];
    if (dims(y)[1] != n) reject("hcat: dim mismatch — `y` dim 1 (= ", dims(y)[1], ") does not match `n` (= ", n, ")");
    return append_col(x, y);
}
}
data {
    int ztime_n;
    vector[ztime_n] ztime;
    int assay_idx_n;
    array[assay_idx_n] int assay_idx;
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
    vector[num_elements(ztime)] raw = std_normal_vector_rng(num_elements(ztime));
    vector[num_elements(ztime)] ftime = (raw * 2.0);
    matrix[num_elements(ztime), 2] X_loc_loc = hcat(rep_vector(1.0, num_elements(ztime)), ftime);
    matrix[num_elements(ztime), 2] Z_loc_loc = hcat(rep_vector(1.0, num_elements(assay_idx)), ftime);
    matrix[num_elements(ztime), 2] y = X_loc_loc;
}