functions {
matrix hcat(
    vector x,
    vector y
) {
    int n = dims(x)[1];
    if (dims(y)[1] != n) reject("hcat: dim mismatch — `y` dim 1 (= ", dims(y)[1], ") does not match `n` (= ", n, ")");
    return append_col(x, y);
}
vector normal_lpdfs(
    vector obs,
    vector loc,
    real scale
) {
    int n = dims(obs)[1];
    return jbroadcasted_normal_lpdfs(obs, loc, scale);
}
vector jbroadcasted_normal_lpdfs(
    vector x1,
    vector x2,
    real x3
) {
    int n = dims(x1)[1];
    vector[n] rv;
    for(i in 1:n) {
        rv[i] = normal_lpdfs(broadcasted_getindex(x1, i), broadcasted_getindex(x2, i), broadcasted_getindex(x3, i));
    }
    return rv;
}
real normal_lpdfs(
    real args1,
    real args2,
    real args3
) {
    return normal_lpdf(args1 | args2, args3);
}
real broadcasted_getindex(vector x, int i) {
    int m = dims(x)[1];
    return x[i];
}
real broadcasted_getindex(real x, int i) {
    return x;
}
vector normal_vector_rng(
    int anontok__1,
    vector a,
    real b
) {
    int n = anontok__1;
    return to_vector(normal_rng(a, b));
}
}
data {
    int a_n;
    vector[a_n] a;
    int b_n;
    vector[b_n] b;
    int y1_n;
    vector[y1_n] y1;
}
transformed data {
}
parameters {
    vector[num_elements(a)] ftime;
}
transformed parameters {
}
model {
    ftime ~ std_normal();
    y1 ~ normal(ftime, 1.0);
}
generated quantities {
    matrix[num_elements(a), 2] X1 = hcat(rep_vector(1.0, num_elements(a)), ftime);
    matrix[num_elements(a), 2] X2 = hcat(rep_vector(1.0, num_elements(b)), ftime);
    vector[y1_n] y1_likelihood = normal_lpdfs(y1, ftime, 1.0);
    vector[y1_n] y1_gen = normal_vector_rng(y1_n, ftime, 1.0);
}