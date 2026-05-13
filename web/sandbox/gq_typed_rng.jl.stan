functions {
array[] real std_normal_real_rng(
    int anontok__1
) {
    int n = anontok__1;
    return normal_rng(rep_vector(0, n), 1);
}
vector normal_vector_rng(
    int anontok__1,
    real a,
    real b
) {
    int n = anontok__1;
    return to_vector(normal_rng(rep_vector(a, n), b));
}
array[] real cauchy_real_rng(
    int anontok__1,
    real a,
    real b
) {
    int n = anontok__1;
    return cauchy_rng(rep_vector(a, n), b);
}
vector lognormal_vector_rng(
    int anontok__1,
    real a,
    real b
) {
    int n = anontok__1;
    return to_vector(lognormal_rng(rep_vector(a, n), b));
}
array[] real gamma_real_rng(
    int anontok__1,
    real a,
    real b
) {
    int n = anontok__1;
    return gamma_rng(rep_vector(a, n), b);
}
vector exponential_vector_rng(
    int anontok__1,
    real a
) {
    int n = anontok__1;
    return to_vector(exponential_rng(rep_vector(a, n)));
}
array[] real student_t_real_rng(
    int anontok__1,
    real nu,
    real a,
    real b
) {
    int n = anontok__1;
    return student_t_rng(nu, rep_vector(a, n), b);
}
vector beta_vector_rng(
    int anontok__1,
    real a,
    real b
) {
    int n = anontok__1;
    return to_vector(beta_rng(rep_vector(a, n), b));
}
array[] real uniform_real_rng(
    int anontok__1,
    real a,
    real b
) {
    int n = anontok__1;
    return uniform_rng(rep_vector(a, n), b);
}
vector chi_square_vector_rng(
    int anontok__1,
    real a
) {
    int n = anontok__1;
    return to_vector(chi_square_rng(rep_vector(a, n)));
}
}
data {
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
    real x = std_normal_rng();
    array[5] real y = std_normal_real_rng(5);
    vector[3] z = normal_vector_rng(3, 0.0, 1.0);
    array[4] real a = cauchy_real_rng(4, 0.0, 1.0);
    vector[4] b = lognormal_vector_rng(4, 0.0, 1.0);
    array[4] real c = gamma_real_rng(4, 2.0, 1.0);
    vector[4] d = exponential_vector_rng(4, 1.0);
    array[4] real e = student_t_real_rng(4, 3.0, 0.0, 1.0);
    vector[4] f = beta_vector_rng(4, 2.0, 2.0);
    array[4] real g = uniform_real_rng(4, 0.0, 1.0);
    vector[4] h = chi_square_vector_rng(4, 3.0);
}