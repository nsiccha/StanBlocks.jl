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
    int c_n;
    int a_n;
    vector[a_n] a;
    vector[c_n] c;
    int d_n;
    int b_n;
    vector[b_n] b;
    vector[d_n] d;
}
transformed data {
    matrix[c_n, 2] X1 = hcat(a, c);
    matrix[d_n, 2] X2 = hcat(b, d);
    matrix[c_n, 2] y = X1;
}
parameters {
}
transformed parameters {
}
model {
}
generated quantities {
}