@inline bsum(x) = sum(x)
@inline flength(x) = Float64(length(x))
@inline @generated bsum(x::Base.Broadcast.Broadcasted{Style,Axes,typeof(+),Args}) where {Style,Axes,Args} = :(
    flength(x) * +($([
        :(bsum(x.args[$i]) / flength(x.args[$i])) for i in 1:fieldcount(Args)
    ]...))
)
@inline bsum(x::Base.Broadcast.Broadcasted{Style,Axes,typeof(-),Tuple{T1,T2}}) where {Style,Axes,T1,T2} = flength(x) * (
    bsum(x.args[1]) / flength(x.args[1]) - bsum(x.args[2]) / flength(x.args[2])
)
@inline log1m(x) = log(1-x)
ternary(c,t,f) = c ? t : f
@inline nothrow_log(x::Real) = x > 0 ? log(x) : -Inf 
# https://mc-stan.org/docs/functions-reference/real-valued_basic_functions.html#betafun
@inline choose(x, y) = binomial(x, y)
@inline lchoose(x, y) = (
    + Distributions.SpecialFunctions.loggamma(x+1)
    - Distributions.SpecialFunctions.loggamma(y+1)
    - Distributions.SpecialFunctions.loggamma(x-y+1)
)
# https://mc-stan.org/docs/functions-reference/unbounded_discrete_distributions.html#nbalt
@inline neg_binomial_2_lpdf(n, mu, phi) = bsum(@broadcasted(
    lchoose(n+phi-1, n)
    + log(mu) * n
    - log(mu+phi) * n
    + log(phi) * phi
    - log(mu+phi) * phi
))

# https://mc-stan.org/docs/functions-reference/unbounded_discrete_distributions.html#poisson
@inline poisson_lpdf(n, lambda) = bsum(@broadcasted(n * log(lambda) - lambda))
# https://mc-stan.org/docs/functions-reference/unbounded_discrete_distributions.html#poisson-distribution-log-parameterization
@inline poisson_log_lpdf(n, alpha) = bsum(@broadcasted(n * alpha - exp(alpha)))
# https://mc-stan.org/docs/functions-reference/bounded_discrete_distributions.html#binomial-distribution-logit-parameterization
@inline binomial_logit_lpmf(args...) = binomial_logit_lpdf(args...)
@inline binomial_logit_lpdf(n, N, alpha) = bsum(@broadcasted(n * loglogistic(alpha) + (N - n) * log1mlogistic(alpha)))
# https://mc-stan.org/docs/2_21/functions-reference/binomial-distribution.html
@inline binomial_lpmf(args...) = binomial_lpdf(args...)
@inline binomial_lpdf(n, N, theta) = bsum(@broadcasted(
    lchoose(N, n)
    + n * log(theta) 
    + (N - n) * log1m(theta)
))
# https://mc-stan.org/docs/functions-reference/bounded_discrete_distributions.html#categorical-distribution
@inline categorical_lpmf(args...) = categorical_lpdf(args...)
@inline categorical_lpdf(y::Integer, theta::AbstractVector) = log(theta[y])
@inline categorical_logit_lpdf(y::Integer, theta::AbstractVector) = log_softmax(theta)[y]
# https://mc-stan.org/docs/2_21/functions-reference/bernoulli-distribution.html
@inline bernoulli_lpmf(args...) = bernoulli_lpdf(args...)
@inline bernoulli_lpdf(y, theta) = bsum(@broadcasted(bernoulli_lpdf(y, theta)))
@inline bernoulli_lpdf(y::Real, theta::Real) = y == 1 ? log(theta) : log1m(theta)
@inline bernoulli_logit_lpmf(args...) = bernoulli_logit_lpdf(args...)
@inline bernoulli_logit_lpdf(y, alpha) = bsum(@broadcasted(bernoulli_logit_lpdf(y, alpha)))
@inline bernoulli_logit_lpdf(y::Real, alpha::Real) = y == 1 ? loglogistic(alpha) : log1mlogistic(alpha)
@inline bernoulli_logit_glm_lpdf(y, X, alpha, beta) = bernoulli_logit_lpdf(y, alpha .+ X * beta)
# @inline bernoulli_logit_glm_lpdf(y, X::AbstractVector, alpha, beta) = bernoulli_logit_lpdf(y, alpha .+ X * beta)
# https://mc-stan.org/docs/2_21/functions-reference/beta-distribution.html
@inline beta_lpmf(args...) = beta_lpdf(args...)
@inline beta_lpdf(theta, alpha, beta) = bsum(@broadcasted((alpha-1)*log(theta) + log1m(theta)*(beta-1)))
# https://mc-stan.org/docs/2_21/functions-reference/dirichlet-distribution.html
@inline dirichlet_lpdf(theta, alpha) = bsum(@broadcasted(log(theta) * (alpha-1)))
# https://mc-stan.org/docs/2_21/functions-reference/gamma-distribution.html
# @ inline gamma_lpdf()
# https://mc-stan.org/docs/functions-reference/matrix_operations.html#exponentiated-quadratic-kernel
@inline gp_exp_quad_cov(x, sigma, length_scale) = @.(sigma^2 * exp(- .5 * square((x - x')/length_scale)))


@inline std_normal_lpdf(x) = -.5 * bsum(@broadcasted(square(x)))
@inline normal_lpdf(x, mu, sigma) = -bsum(@broadcasted(log(sigma)+.5*square((x-mu)/sigma)))
# https://mc-stan.org/docs/2_21/functions-reference/normal-id-glm.html
@inline normal_id_glm_lpdf(y,X,alpha,beta,sigma) = normal_lpdf(y, alpha .+ X * beta, sigma)
@inline lognormal_lpdf(x, mu, sigma) = begin
    bsum(@broadcasted(
        -log(sigma)
        -.5*log(x)
        -.5*square((log(x)-mu)/sigma)
    ))
end
@inline weibull_lpdf(y, alpha, sigma) = bsum(@broadcasted(
    log(alpha) 
    - alpha * log(sigma)
    + (alpha-1) * log(y)
    - (y/sigma)^alpha
))
@inline StudentT(nu, mu, sigma) = mu + sigma * TDist(nu)
@inline student_t_lpdf(x, args...) = bsum(@broadcasted(logpdf(StudentT(args...), x)))
# https://mc-stan.org/docs/functions-reference/unbounded_continuous_distributions.html#cauchy-distribution
@inline cauchy_lpdf(x, location, scale) = begin
    -bsum(@broadcasted(log(scale) + log1p((x-location)/scale)))
end
# https://mc-stan.org/docs/functions-reference/positive_continuous_distributions.html#exponential-distribution
@inline exponential_lpdf(y, beta) = bsum(@broadcasted(log(beta) - beta * y))
@inline double_exponential_lpdf(x, args...) = bsum(@broadcasted(logpdf(DoubleExponential(args...), x)))
# https://mc-stan.org/docs/functions-reference/positive_continuous_distributions.html#gamma-distribution
@inline gamma_lpdf(x, alpha, theta) = bsum(@broadcasted(logpdf(Gamma(alpha, 1/theta), x)))
# https://mc-stan.org/docs/2_21/functions-reference/inverse-gamma-distribution.html
@inline inv_gamma_lpdf(x, alpha, theta) = sum(@broadcasted(logpdf(InverseGamma(alpha, theta), x)))
@inline uniform_lpdf(x, a, b) = sum(@broadcasted(logpdf(Uniform(a, b), x)))
@inline multi_normal_lpdf(x, mu, cov) = logpdf(MultivariateNormal(mu, cov), x)
@inline multi_normal_lpdf(x::AbstractMatrix, mu::AbstractVector, cov::AbstractMatrix) = sum(multi_normal_lpdf.(eachrow(x), Ref(mu), Ref(cov)))
@inline multi_normal_cholesky_lpdf(x::AbstractVector, mu::AbstractVector, L::AbstractMatrix) = begin
    -2*bsum(@broadcasted(log($view(L, $diagind(L))))) -.5 * dot_self(L \ (mu - x))
end
@inline log_sum_exp(args...) = logsumexp(args)
@inline log_sum_exp(x) = logsumexp(x)
@inline inv_logit(x) = logistic(x)
@inline log_inv_logit(x) = loglogistic(x)
@inline log1m_inv_logit(x) = log1mlogistic(x)
@inline rep_vector(x, n) = fill(x, n)
@inline rep_matrix(x, args...) = fill(x, args)
@inline rep_row_vector(x, n) = fill(x, (1, n))
@inline append_row(args...) = reduce(vcat, args)
@inline append_col(args...) = reduce(hcat, args)
@inline sub_col(x, i, j, n_rows) = x[i:(i+n_rows-1), j]
@inline sub_row(x, i, j, n_cols) = x[i, j:(j+n_cols-1)]'
@inline segment(x, i, n) = x[i:(i+n-1)]
@inline stan_tail(x, i) = x[(end-i+1):end]
@inline diag_matrix(x) = Diagonal(x)
@inline diag_pre_multiply(d, x) = Diagonal(d) * x
@inline cholesky_decompose(x) = cholesky(x).L
@inline softmax(x) = @.(exp(x - $logsumexp(x)))
@inline log_softmax(x) = x .- logsumexp(x)
@inline square(x::Real) = x ^ 2
@inline pow(x, p) = x ^ p
@inline sd(x) = std(x)
@inline cumulative_sum(x) = cumsum(x)
@inline dot_product(x, y) = dot(x, y)
@inline dot_self(x) = sum(@broadcasted(square(x)))
# https://mc-stan.org/docs/2_19/stan-users-guide/vectorizing-mixtures.html
@inline log_mix(lambda, lpdf1, lpdf2) = log_sum_exp(
    log(lambda) + lpdf1,
    log1m(lambda) + lpdf2
)

function integrate_ode_rk45 end
function integrate_ode_bdf end

@inline apply_to_second(f, x::Tuple) = x[1], f(x[2])
@inline constrain_reshape(dty::Tuple) = dty
@inline constrain_reshape(y::AbstractVector, n1, n2) = reshape(y, (n2, n1))'
@inline constrain_reshape(dty::Tuple, args...) = dty[1], constrain_reshape(dty[2], args...)
@inline constrain(x, n=missing; lower=-Inf, upper=+Inf) = if isfinite(lower) && isfinite(upper)
    bsum(@broadcasted(log(upper - lower) - x - 2 * log1pexp(-x))), @.(lower + logistic(x) * (upper - lower))
elseif !isfinite(lower) && !isfinite(upper)
    0., x
elseif isfinite(lower) && !isfinite(upper)
    sum(x), @.(lower + exp(x))
else
    sum(x), @.(upper - exp(x))
end
@inline constrain2(x, n=missing, lower=-Inf, upper=+Inf) = if isfinite(lower) && isfinite(upper)
    bsum(@broadcasted(log(upper - lower) - x - 2 * log1pexp(-x))), @.(lower + logistic(x) * (upper - lower))
elseif !isfinite(lower) && !isfinite(upper)
    0., x
elseif isfinite(lower) && !isfinite(upper)
    sum(x), @.(lower + exp(x))
else
    sum(x), @.(upper - exp(x))
end
@inline constrain(x, args...; kwargs...) = constrain_reshape(constrain(x; kwargs...), args...)
const real_constrain = constrain
const vector_constrain = constrain
const matrix_constrain = constrain
@inline row_vector_constrain(x, n; kwargs...) = apply_to_second(adjoint, constrain(x, n; kwargs...))

@inline real_unconstrained_dim(args...) = prod(args)
const vector_unconstrained_dim = real_unconstrained_dim
const matrix_unconstrained_dim = real_unconstrained_dim
const row_vector_unconstrained_dim = real_unconstrained_dim

const ordered_unconstrained_dim = real_unconstrained_dim
@inline ordered_constrain(x, n) = @views sum(x[2:end]), cumsum(vcat(x[1], @.(exp(x[2:end]))))

const positive_ordered_unconstrained_dim = real_unconstrained_dim
@inline positive_ordered_constrain(x, n) = sum(x), cumsum(@.(exp(x)))

simplex_unconstrained_dim(args...) = prod((Base.front(args)..., args[end]-1))
# https://mc-stan.org/docs/2_19/reference-manual/simplex-transform-section.html
@inline simplex_constrain(xi, K) = simplex_constrain!(similar(xi, K), xi, K)
@inline simplex_constrain(xi, n1, K) = begin 
    x = similar(xi, (K, n1))
    XI = reshape(xi, (K-1, n1))
    dtarget = 0.
    for i in 1:n1
        dtarget += simplex_constrain!(view(x, :, i), view(XI, :, i), K)[1]
    end
    dtarget, x'
end
@inline simplex_constrain!(x, xi, K) = begin 
    drv = 0.
    rem = 1.
    for k in 1:length(xi)
        z = logistic(xi[k] - log(K - k))
        x[k] = rem * z
        drv += log(z) + log(1-z) + log(rem)
        rem -= x[k]
    end
    x[end] = rem
    drv, x
end

@inline cholesky_factor_corr_unconstrained_dim(n) = ((n-1) * n) ÷ 2 
@inline cholesky_factor_corr_constrain(x, n) = begin 
    drv = 0.
    rv = zeros((n,n))
    xi = 1
    for i in 1:n
        rem = 1.
        for j in 1:i-1
            xij = x[xi]
            fac = tanh(xij)
            drv -= 2 * logcosh(xij) + .5 * log(rem)
            rv[i,j] = fac * sqrt(rem)
            rem -= fac^2 * rem
            xi += 1
        end
        rv[i,i] = sqrt(rem)
    end
    drv, rv
end 
@inline lkj_corr_cholesky_lpdf(L::AbstractMatrix, eta) = begin
    rv = 0.
    K = size(L, 1)
    for k in 2:K
        rv += (K -k + 2eta - 2) * log(L[k,k])
    end
    rv
end