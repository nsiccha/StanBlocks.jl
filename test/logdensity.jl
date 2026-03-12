import StanBlocks.stan: stan_model, instantiate
using LogDensityProblems

# ──────────────────────────────────────────────────────────────────────────────
# Tests: mathematical correctness of generated Stan log-densities
#
# For each model we:
#   1. Build the StanBlocks model and compile it to a Stan problem.
#   2. Evaluate LogDensityProblems.logdensity at several unconstrained points.
#   3. Compare against the analytically expected value computed in Julia.
#
# Parameterisation conventions (matching Stan defaults):
#   • Unconstrained real / vector → identity map, no Jacobian.
#   • lower=0  (positive reals)  → σ = exp(θ),       log|J| = +θ
#   • [0,1]    (e.g. beta prior) → p = inv_logit(θ),  log|J| = log p + log(1-p)
#
# Note: Stan drops additive constants that don't depend on parameters:
#   • std_normal_lpdf(x)        = -0.5*x^2
#   • normal_lpdf(x|mu,sigma)   = -log(sigma) - 0.5*((x-mu)/sigma)^2
#   • beta_lpdf(x|1,1)          = 0     (B(1,1)=1, so no constant dropped)
#   • binomial_lpmf(k|n,p)      = k*log(p) + (n-k)*log(1-p)   (drops log C(n,k))
# ──────────────────────────────────────────────────────────────────────────────

# Stan kernel formulas (constants dropped)
_stan_std_normal(x) = -0.5*x^2
_stan_normal(x, mu, sigma) = -log(sigma) - 0.5*((x-mu)/sigma)^2
_stan_binomial(k, n, p) = k*log(p) + (n-k)*log(1.0-p)

@testset "logdensity" begin

    # --------------------------------------------------------------------------
    # 1. Single unconstrained scalar parameter
    #
    # Model:  mu  ~ std_normal()
    #         obs ~ normal(mu, 1.0)
    #
    # log p(θ=[mu]) = std_normal_lpdf(mu) + normal_lpdf(obs|mu,1)
    # --------------------------------------------------------------------------
    @testset "unconstrained scalar" begin
        obs_val = 0.5
        problem = instantiate(stan_model(@slic (;obs=obs_val) begin
            mu ~ std_normal()
            obs ~ normal(mu, 1.0)
        end))

        @test LogDensityProblems.dimension(problem) == 1

        for mu_val in [-2.0, 0.0, 1.5]
            expected = _stan_std_normal(mu_val) + _stan_normal(obs_val, mu_val, 1.0)
            @test LogDensityProblems.logdensity(problem, [mu_val]) ≈ expected atol=1e-6
        end
    end

    # --------------------------------------------------------------------------
    # 2. Lower-bounded scalar parameter (log transform)
    #
    # Model:  sigma ~ std_normal(;lower=0.)
    #         obs   ~ normal(0., sigma)
    #
    # σ = exp(θ),  log|J| = θ
    # log p(θ) = std_normal_lpdf(exp(θ)) + θ + normal_lpdf(obs|0,exp(θ))
    # --------------------------------------------------------------------------
    @testset "lower-bounded scalar" begin
        obs_val = 1.0
        problem = instantiate(stan_model(@slic (;obs=obs_val) begin
            sigma ~ std_normal(;lower=0.)
            obs ~ normal(0., sigma)
        end))

        @test LogDensityProblems.dimension(problem) == 1

        for θ_val in [-1.0, 0.0, 1.0]
            sigma_val = exp(θ_val)
            expected = (
                _stan_std_normal(sigma_val) + θ_val   # prior + log-Jacobian
                + _stan_normal(obs_val, 0.0, sigma_val)  # likelihood
            )
            @test LogDensityProblems.logdensity(problem, [θ_val]) ≈ expected atol=1e-6
        end
    end

    # --------------------------------------------------------------------------
    # 3. Interval-bounded scalar parameter (logit transform)
    #
    # Model:  theta ~ beta(1., 1.)
    #         k     ~ binomial(n, theta)
    #
    # beta automatically gets lower=0, upper=1.
    # p = inv_logit(θ),  log|J| = log(p) + log(1-p)
    # log p(θ) = beta_lpdf(p|1,1) + log(p) + log(1-p) + binomial_lpmf(k|n,p)
    #          = 0                + log(p) + log(1-p) + k*log(p) + (n-k)*log(1-p)
    # --------------------------------------------------------------------------
    @testset "interval-bounded scalar (beta prior)" begin
        k_val, n_val = 3, 10
        problem = instantiate(stan_model(@slic (;k=k_val, n=n_val) begin
            theta ~ beta(1., 1.)
            k ~ binomial(n, theta)
        end))

        @test LogDensityProblems.dimension(problem) == 1

        for θ_val in [-2.0, 0.0, 1.5]
            p = 1.0 / (1.0 + exp(-θ_val))          # inv_logit
            log_jac = log(p) + log(1.0 - p)         # log Jacobian of logit parameterisation
            expected = (
                0.0 + log_jac                        # beta(1,1) contributes 0
                + _stan_binomial(k_val, n_val, p)
            )
            @test LogDensityProblems.logdensity(problem, [θ_val]) ≈ expected atol=1e-6
        end
    end

    # --------------------------------------------------------------------------
    # 4. Unconstrained vector parameter
    #
    # Model:  mu  ~ std_normal()   (n-vector, element-wise iid N(0,1))
    #         obs ~ normal(mu, 1.)
    #
    # log p(θ=mu) = Σᵢ std_normal_lpdf(μᵢ) + Σᵢ normal_lpdf(obsᵢ|μᵢ,1)
    # --------------------------------------------------------------------------
    @testset "unconstrained vector" begin
        obs_val = [0.5, -0.3, 1.2]
        n = length(obs_val)
        problem = instantiate(stan_model(@slic (;obs=obs_val) begin
            mu ~ std_normal(;n=length(obs))
            obs ~ normal(mu, 1.0)
        end))

        @test LogDensityProblems.dimension(problem) == n

        mu_val = [0.1, -0.5, 0.8]
        expected = (
            sum(_stan_std_normal.(mu_val))
            + sum(_stan_normal.(obs_val, mu_val, 1.0))
        )
        @test LogDensityProblems.logdensity(problem, mu_val) ≈ expected atol=1e-6
    end

    # --------------------------------------------------------------------------
    # 5. Linear regression with flat (improper) prior
    #
    # Model:  alpha ~ flat()
    #         beta  ~ flat()
    #         y     ~ normal(alpha .+ beta .* x, 1.0)
    #
    # flat() contributes 0 to the target.
    # log p(θ=[alpha,beta]) = Σᵢ normal_lpdf(yᵢ|alpha + beta*xᵢ, 1)
    # --------------------------------------------------------------------------
    @testset "linear regression, flat prior" begin
        x_val = [1.0, 2.0, 3.0]
        y_val = [2.1, 3.9, 6.2]
        problem = instantiate(stan_model(@slic (;y=y_val, x=x_val) begin
            alpha ~ flat()
            beta  ~ flat()
            y ~ normal(alpha + beta * to_vector(x), 1.0)
        end))

        @test LogDensityProblems.dimension(problem) == 2

        for (alpha_val, beta_val) in [(0.0, 2.0), (1.0, 1.5), (-0.5, 2.1)]
            predicted = alpha_val .+ beta_val .* x_val  # Julia broadcast is fine here
            expected  = sum(_stan_normal.(y_val, predicted, 1.0))
            @test LogDensityProblems.logdensity(problem, [alpha_val, beta_val]) ≈ expected atol=1e-6
        end
    end

    # --------------------------------------------------------------------------
    # 6. Hierarchical model: group means with hyperprior
    #
    # Model:  mu    ~ std_normal()
    #         sigma ~ std_normal(;lower=0.)
    #         theta ~ normal(mu, sigma)   (n-vector)
    #         y     ~ normal(theta, 1.)
    #
    # Unconstrained: θ_unc = [mu, log_sigma, theta_1, …, theta_n]
    # log p(θ_unc) = std_normal_lpdf(mu)
    #              + std_normal_lpdf(exp(log_sigma)) + log_sigma   (prior+Jacobian)
    #              + Σ normal_lpdf(theta_i|mu, exp(log_sigma))
    #              + Σ normal_lpdf(y_i|theta_i, 1)
    # --------------------------------------------------------------------------
    @testset "hierarchical (normal-normal)" begin
        n = 3
        y_val = [0.2, -0.1, 0.8]
        problem = instantiate(stan_model(@slic (;y=y_val) begin
            mu    ~ std_normal()
            sigma ~ std_normal(;lower=0.)
            theta ~ normal(mu, sigma; n=length(y))
            y     ~ normal(theta, 1.)
        end))

        @test LogDensityProblems.dimension(problem) == n + 2  # mu, log_sigma, theta[1..n]

        mu_unc    = 0.3
        log_sigma = 0.1
        theta_unc = [0.5, -0.2, 0.6]
        θ = [mu_unc; log_sigma; theta_unc]

        sigma_val = exp(log_sigma)
        expected = (
            _stan_std_normal(mu_unc)                                          # p(mu)
            + _stan_std_normal(sigma_val) + log_sigma                         # p(sigma) + Jacobian
            + sum(_stan_normal.(theta_unc, mu_unc, sigma_val))                # p(theta|mu,sigma)
            + sum(_stan_normal.(y_val, theta_unc, 1.0))                       # p(y|theta)
        )
        @test LogDensityProblems.logdensity(problem, θ) ≈ expected atol=1e-6
    end

    # --------------------------------------------------------------------------
    # 7. Dimension is correctly inferred from data shapes
    # --------------------------------------------------------------------------
    @testset "dimension from data" begin
        for n in [1, 5, 20]
            obs = randn(n)
            problem = instantiate(stan_model(@slic (;obs=obs) begin
                mu ~ std_normal(;n=length(obs))
                obs ~ normal(mu, 1.0)
            end))
            @test LogDensityProblems.dimension(problem) == n
        end
    end

end
