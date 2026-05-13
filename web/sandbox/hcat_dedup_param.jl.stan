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
    int a_n;
    vector[a_n] a;
    int b_n;
    vector[b_n] b;
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
    vector[num_elements(a)] ftime = std_normal_vector_rng(num_elements(a));
    matrix[num_elements(a), 2] X1 = hcat(rep_vector(1.0, num_elements(a)), ftime);
    matrix[num_elements(a), 2] X2 = hcat(rep_vector(1.0, num_elements(b)), ftime);
    matrix[num_elements(a), 2] y = X1;
}