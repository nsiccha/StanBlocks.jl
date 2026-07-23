# Feature 1 increment 1 — exercise ALL 9 registered Stan 2.37 transforms in one
# valid inference program: 3 families (simplex/ordered/positive_ordered) x
# {constrain, unconstrain, jacobian}, covering both size branches
# (simplex n<->n+/-1, ordered/positive_ordered n->n). One stanc check validates
# every signature + that Stan 2.37 actually exposes each name.
@slic (;y=0.3) begin
    sf::vector[2] ~ std_normal()
    of::vector[3] ~ std_normal()
    pf::vector[3] ~ std_normal()
    s_j = simplex_jacobian(sf)                # vector[3], +jac
    s_c = simplex_constrain(sf)               # vector[3]
    s_u = simplex_unconstrain(s_j)            # vector[2]
    o_j = ordered_jacobian(of)                # vector[3], +jac
    o_c = ordered_constrain(of)               # vector[3]
    o_u = ordered_unconstrain(o_j)            # vector[3]
    p_j = positive_ordered_jacobian(pf)       # vector[3], +jac
    p_c = positive_ordered_constrain(pf)      # vector[3]
    p_u = positive_ordered_unconstrain(p_j)   # vector[3]
    y ~ normal(s_j[1]+s_c[1]+s_u[1]+o_j[1]+o_c[1]+o_u[1]+p_j[1]+p_c[1]+p_u[1], 0.1)
end
