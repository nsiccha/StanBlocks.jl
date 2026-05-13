functions {
int ragged_total(tuple(vector, array[] int) x) {
    return num_elements(x.1);
}
vector std_normal_vector_rng(
    int anontok__1
) {
    int n = anontok__1;
    return to_vector(normal_rng(rep_vector(0, n), 1));
}
}
data {
    int dose_times_mem_n;
    int dose_times_ends_n;
    tuple(vector[dose_times_mem_n], array[dose_times_ends_n] int) dose_times;
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
    vector[ragged_total(dose_times)] x = std_normal_vector_rng(ragged_total(dose_times));
}