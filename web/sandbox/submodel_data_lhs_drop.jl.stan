functions {
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
    real sd_a;
}
transformed data {
}
parameters {
    vector[num_elements(a)] me_a_x_true;
}
transformed parameters {
}
model {
    me_a_x_true ~ std_normal();
    a ~ normal(me_a_x_true, sd_a);
}
generated quantities {
    vector[a_n] a_likelihood = normal_lpdfs(a, me_a_x_true, sd_a);
    vector[a_n] a_gen = normal_vector_rng(a_n, me_a_x_true, sd_a);
    vector[num_elements(a)] me_a = me_a_x_true;
    array[num_elements(a)] real y = normal_rng(me_a, 1.0);
}