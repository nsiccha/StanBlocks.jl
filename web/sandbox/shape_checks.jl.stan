functions {
vector std_normal_vector_rng(
    int anontok__1
) {
    int n = anontok__1;
    return to_vector(normal_rng(rep_vector(0, n), 1));
}
real mydot(
    vector x,
    vector y
) {
    int n = dims(x)[1];
    if (dims(y)[1] != n) reject("mydot: dim mismatch — `y` dim 1 (= ", dims(y)[1], ") does not match `n` (= ", n, ")");
    return dot_product(x, y);
}
vector rowdot(
    matrix A,
    matrix B
) {
    int m = dims(A)[1];
    int n = dims(A)[2];
    if (dims(B)[1] != m) reject("rowdot: dim mismatch — `B` dim 1 (= ", dims(B)[1], ") does not match `m` (= ", m, ")");
    if (dims(B)[2] != n) reject("rowdot: dim mismatch — `B` dim 2 (= ", dims(B)[2], ") does not match `n` (= ", n, ")");
    return rows_dot_product(A, B);
}
vector mvmul(
    matrix A,
    vector v
) {
    int m = dims(A)[1];
    int n = dims(A)[2];
    if (dims(v)[1] != n) reject("mvmul: dim mismatch — `v` dim 1 (= ", dims(v)[1], ") does not match `n` (= ", n, ")");
    return (A * v);
}
}
data {
    int y_n;
    vector[y_n] y;
    int Z_m;
    int Y_m;
    int Y_n;
    matrix[Y_m, Y_n] Y;
    int Z_n;
    matrix[Z_m, Z_n] Z;
}
transformed data {
    vector[Z_m] rd = rowdot(Y, Z);
}
parameters {
}
transformed parameters {
}
model {
}
generated quantities {
    vector[5] mu = std_normal_vector_rng(5);
    real s = mydot(mu, y);
    vector[Y_m] p = mvmul(Y, mu);
}