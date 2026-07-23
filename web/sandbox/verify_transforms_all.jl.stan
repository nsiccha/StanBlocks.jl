functions {
real normal_lpdfs(
    real args1,
    real args2,
    real args3
) {
    return normal_lpdf(args1 | args2, args3);
}
}
data {
    real y;
}
transformed data {
}
parameters {
    vector[2] sf;
    vector[3] of;
    vector[3] pf;
}
transformed parameters {
    vector[(2 + 1)] s_j = simplex_jacobian(sf);
    vector[(2 + 1)] s_c = simplex_constrain(sf);
    vector[((2 + 1) - 1)] s_u = simplex_unconstrain(s_j);
    vector[3] o_j = ordered_jacobian(of);
    vector[3] o_c = ordered_constrain(of);
    vector[3] o_u = ordered_unconstrain(o_j);
    vector[3] p_j = positive_ordered_jacobian(pf);
    vector[3] p_c = positive_ordered_constrain(pf);
    vector[3] p_u = positive_ordered_unconstrain(p_j);
}
model {
    sf ~ std_normal();
    of ~ std_normal();
    pf ~ std_normal();
    y ~ normal((s_j[1] + s_c[1] + s_u[1] + o_j[1] + o_c[1] + o_u[1] + p_j[1] + p_c[1] + p_u[1]), 0.1);
}
generated quantities {
    real y_likelihood = normal_lpdfs(
        y,
        (s_j[1] + s_c[1] + s_u[1] + o_j[1] + o_c[1] + o_u[1] + p_j[1] + p_c[1] + p_u[1]),
        0.1
    );
    real y_gen = normal_rng((s_j[1] + s_c[1] + s_u[1] + o_j[1] + o_c[1] + o_u[1] + p_j[1] + p_c[1] + p_u[1]), 0.1);
}