# PKPD-inspired composite: hierarchical regression that uses several
# inline helpers (mutation, HOF, vararg). Each call inlines, so no helper
# pollutes Stan's `functions {}` block.

# Compound-assign mutation: accumulate a weighted sum into rv.
@deffun @inline addinto!(rv::real, w::vector[n], x::vector[n])::real = begin
    for i in 1:n
        rv += w[i] * x[i]
    end
    return rv
end

# HOF + indexed-assign mutation: write f(x[i]) into out[i] elementwise.
@deffun @inline mapinto!(out::vector[n], f, x::vector[n])::vector[n] = begin
    for i in 1:n
        out[i] = f(x[i])
    end
    return out
end

# Math helper used as a HOF argument.
@deffun softplus(x::real)::real = log1p(exp(x))

# Variadic helper — combines a base intercept with several per-feature
# contributions through scalar `addinto!`.
@deffun @inline combine(alpha::real, beta::vector[k], xs...)::real = begin
    rv = alpha
    addinto!(rv, beta, xs[1])
    return rv
end

# Non-inline outer UDF that orchestrates the inline helpers.
@deffun build_eta(alpha::real, beta::vector[k], X::matrix[n, k])::vector[n] = begin
    eta = rep_vector(alpha, n)
    smooth = rep_vector(0., n)
    mapinto!(smooth, softplus, eta)
    return eta + smooth
end

@slic (;y=randn(20), X=randn(20, 3)) begin
    alpha ~ std_normal()
    beta  ~ std_normal(;n=3)
    sigma ~ std_normal(;lower=0.)
    mu    = build_eta(alpha, beta, X)
    y     ~ normal(mu, sigma)
end