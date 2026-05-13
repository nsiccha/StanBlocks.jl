functions {
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
    matrix[a_n, 2] X1 = hcat(rep_vector(1.0, num_elements(a)), a);
    matrix[b_n, 2] X2 = hcat(rep_vector(1.0, num_elements(b)), b);
    matrix[a_n, 2] y = X1;
}
parameters {
}
transformed parameters {
}
model {
}
generated quantities {
}