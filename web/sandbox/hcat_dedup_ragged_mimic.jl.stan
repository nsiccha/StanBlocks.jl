functions {
vector std_normal_vector_rng(
    int anontok__1
) {
    int n = anontok__1;
    return to_vector(normal_rng(rep_vector(0, n), 1));
}
matrix hcat(vector x) {
    int n = dims(x)[1];
    return to_matrix(x, n, 1);
}
}
data {
    int ztime_n;
    vector[ztime_n] ztime;
    int dose_times_mem_n;
    int dose_times_ends_n;
    tuple(vector[dose_times_mem_n], array[dose_times_ends_n] int) dose_times;
}
transformed data {
    matrix[num_elements(dose_times), 1] X_log_err = hcat(rep_vector(1.0, num_elements(dose_times)));
}
parameters {
}
transformed parameters {
}
model {
}
generated quantities {
    vector[num_elements(ztime)] ftime = std_normal_vector_rng(num_elements(ztime));
    matrix[num_elements(ztime), 1] X_loc = hcat(ftime);
    matrix[num_elements(ztime), 1] y = X_loc;
}