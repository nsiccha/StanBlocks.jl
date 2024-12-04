module PosteriorDBExt

import PosteriorDB
import StanBlocks
import StanBlocks: julia_implementation, @stan, @parameters, @transformed_parameters, @model, @broadcasted
import StanBlocks: bernoulli_lpmf, binomial_lpmf, log_sum_exp, logit, binomial_logit_lpmf, bernoulli_logit_lpmf, inv_logit, log_inv_logit, rep_vector, square, normal_lpdf, sd, multi_normal_lpdf, student_t_lpdf, gp_exp_quad_cov, log_mix, append_row, append_col, pow, diag_matrix, normal_id_glm_lpdf, rep_matrix, rep_row_vector, cholesky_decompose, dot_self, cumulative_sum, softmax, log1m_inv_logit, matrix_constrain, integrate_ode_rk45, integrate_ode_bdf, poisson_lpdf, nothrow_log, categorical_lpmf, dirichlet_lpdf, exponential_lpdf, sub_col, stan_tail, dot_product, segment, inv_gamma_lpdf, diag_pre_multiply, multi_normal_cholesky_lpdf
using Statistics, LinearAlgebra

@inline PosteriorDB.implementation(model::PosteriorDB.Model, ::Val{:stan_blocks}) = julia_implementation(Val(Symbol(PosteriorDB.name(model))))

julia_implementation(posterior::PosteriorDB.Posterior) = julia_implementation(
    Val(Symbol(PosteriorDB.name(PosteriorDB.model(posterior))));
    Dict([Symbol(k)=>v for (k, v) in pairs(PosteriorDB.load(PosteriorDB.dataset(posterior)))])...
)

julia_implementation(::Val{:earn_height}; N, earn, height, kwargs...) = begin 
    @stan begin 
        @parameters begin
            beta::vector[2]
            sigma::real(lower=0.)
        end
        @model begin
            earn ~ normal(@broadcasted(beta[1] + beta[2] * height), sigma);
        end
    end
end

julia_implementation(::Val{:wells_dist}; N, switched, dist, kwargs...) = begin 
    @stan begin 
        @parameters begin
            beta::vector[2]
        end
        @model begin
            switched ~ bernoulli_logit(@broadcasted(beta[1] + beta[2] * dist));
        end
    end
end
julia_implementation(::Val{:sesame_one_pred_a}; N, encouraged, watched, kwargs...) = begin 
    @stan begin 
        @parameters begin
            beta::vector[2]
            sigma::real(lower=0.)
        end
        @model begin
            watched ~ normal(@broadcasted(beta[1] + beta[2] * encouraged), sigma);
        end
    end
end
julia_implementation(::Val{:Rate_1_model}; n, k, kwargs...) = begin 
    @stan begin 
        target = 0.
        @parameters begin
            theta::real(lower=0.,upper=1.)
        end
        @model begin
            theta ~ beta(1, 1)
            k ~ binomial(n, theta)
        end
        return target
    end
end
julia_implementation(::Val{:nes_logit_model}; N, income, vote, kwargs...) = begin 
    X = reshape(income, (N, 1))
    @stan begin 
        target = 0.
        @parameters begin
            alpha::real
            beta::vector[1]
        end
        @model begin
            vote ~ bernoulli_logit_glm(X, alpha, beta);
        end
        return target
    end
end
julia_implementation(::Val{:kidscore_momiq}; N, kid_score, mom_iq, kwargs...) = begin 
    @stan begin 
        target = 0.
        @parameters begin
            beta::vector[2]
            sigma::real(lower=0)
        end
        @model begin
            sigma ~ cauchy(0, 2.5);
            kid_score ~ normal(@broadcasted(beta[1] + beta[2] * mom_iq), sigma);
        end
        return target
    end
end
julia_implementation(::Val{:kidscore_momhs}; N, kid_score, mom_hs, kwargs...) = begin 
    @stan begin 
        target = 0.
        @parameters begin
            beta::vector[2]
            sigma::real(lower=0)
        end
        @model begin
            sigma ~ cauchy(0, 2.5);
            kid_score ~ normal(@broadcasted(beta[1] + beta[2] * mom_hs), sigma);
        end
        return target
    end
end
julia_implementation(::Val{:logearn_height}; N, earn, height, kwargs...) = begin 
    log_earn = @. log(earn)
    @stan begin 
        target = 0.
        @parameters begin
            beta::vector[2]
            sigma::real(lower=0)
        end
        @model begin
            log_earn ~ normal(@broadcasted(beta[1] + beta[2] * height), sigma);
        end
        return target
    end
end
julia_implementation(::Val{:blr}; N, D, X, y, kwargs...) = begin 
    @assert size(X) == (N,D)
    @stan begin 
        target = 0.
        @parameters begin
            beta::vector[D]
            sigma::real(lower=0)
        end
        @model begin
            target += normal_lpdf(beta, 0, 10);
            target += normal_lpdf(sigma, 0, 10);
            target += normal_lpdf(y, X * beta, sigma);
        end
        return target
    end
end
julia_implementation(::Val{:wells_dist100_model}; N, switched, dist, kwargs...) = begin 
    dist100 = @. dist / 100.
    X = reshape(dist100, (N, 1))
    @stan begin 
        target = 0.
        @parameters begin
            alpha::real
            beta::vector[1]
        end
        @model begin
            switched ~ bernoulli_logit_glm(X, alpha, beta);
        end
        return target
    end
end
julia_implementation(::Val{:Rate_3_model}; n1, n2, k1, k2, kwargs...) = begin 
    @stan begin 
        target = 0.
        @parameters begin
            theta::real(lower=0,upper=1)
        end
        @model begin
            theta ~ beta(1, 1)
            k1 ~ binomial(n1, theta)
            k2 ~ binomial(n2, theta)
        end
        return target
    end
end
julia_implementation(::Val{:logearn_height_male}; N, earn, height, male, kwargs...) = begin 
    log_earn = @. log(earn)
    @stan begin 
        target = 0.
        @parameters begin
            beta::vector[3]
            sigma::real(lower=0)
        end
        @model begin
            log_earn ~ normal(@broadcasted(beta[1] + beta[2] * height + beta[3] * male), sigma);
        end
        return target
    end
end
julia_implementation(::Val{:kidscore_momhsiq}; N, kid_score, mom_iq, mom_hs, kwargs...) = begin 
    @stan begin 
        target = 0.
        @parameters begin
            beta::vector[3]
            sigma::real(lower=0)
        end
        @model begin
            sigma ~ cauchy(0, 2.5);
            kid_score ~ normal(@broadcasted(beta[1] + beta[2] * mom_hs + beta[3] * mom_iq), sigma);
        end
        return target
    end
end
julia_implementation(::Val{:log10earn_height}; N, earn, height, kwargs...) = begin 
    log10_earn = @. log10(earn)
    @stan begin 
        target = 0.
        @parameters begin
            beta::vector[2]
            sigma::real(lower=0)
        end
        @model begin
            log10_earn ~ normal(@broadcasted(beta[1] + beta[2] * height), sigma);
        end
        return target
    end
end
julia_implementation(::Val{:wells_dist100ars_model}; N, switched, dist, arsenic, kwargs...) = begin 
    dist100 = @. dist / 100.
    X = hcat(dist100, arsenic)
    @stan begin 
        target = 0.
        @parameters begin
            alpha::real
            beta::vector[2]
        end
        @model begin
            switched ~ bernoulli_logit_glm(X, alpha, beta);
        end
        return target
    end
end
julia_implementation(::Val{:low_dim_gauss_mix_collapse}; N, y, kwargs...) = begin 
    @stan begin 
        target = 0.
        @parameters begin
            mu::vector[2]
            sigma::vector(lower=0)[2]
            theta::real(lower=0,upper=1)
        end
        @model begin
            sigma ~ normal(0, 2);
            mu ~ normal(0, 2);
            theta ~ beta(5, 5);
            for n in 1:N
              target += log_mix(theta, normal_lpdf(y[n], mu[1], sigma[1]),
                                normal_lpdf(y[n], mu[2], sigma[2]));
            end
        end
        return target
    end
end
julia_implementation(::Val{:normal_mixture_k}; K, N, y, kwargs...) = begin 
    @stan begin 
        target = 0.
        @parameters begin
            theta::simplex[K]
            mu::vector[K]
            sigma::vector(lower=0.,upper=10.)[K]
        end
        @model begin
            mu ~ normal(0., 10.);
            for n in 1:N
                ps = @broadcasted(log(theta) + normal_lpdf(y[n], mu, sigma))
                target += log_sum_exp(ps);
            end
        end
        return target
    end
end
julia_implementation(::Val{:low_dim_gauss_mix}; N, y, kwargs...) = begin 
    @stan begin 
        target = 0.
        @parameters begin
            mu::ordered[2]
            sigma::vector(lower=0)[2]
            theta::real(lower=0,upper=1)
        end
        @model begin
            sigma ~ normal(0, 2);
            mu ~ normal(0, 2);
            theta ~ beta(5, 5);
            for n in 1:N
                target += log_mix(theta, normal_lpdf(y[n], mu[1], sigma[1]),
                                normal_lpdf(y[n], mu[2], sigma[2]));
            end
        end
        return target
    end
end
julia_implementation(::Val{:radon_county}; N, J, county, y, kwargs...) = begin 
    @stan begin 
        target = 0.
        @parameters begin
            a::vector[J]
            mu_a::real
            sigma_a::real(lower=0, upper=100)
            sigma_y::real(lower=0, upper=100)
        end
        @model begin
            y_hat = a[county]
            
            mu_a ~ normal(0, 1);
            a ~ normal(mu_a, sigma_a);
            y ~ normal(y_hat, sigma_y);
        end
        return target
    end
end
julia_implementation(::Val{:logearn_logheight_male}; N, earn, height, male, kwargs...) = begin 
    log_earn = @. log(earn)
    log_height = @. log(height)
    @stan begin 
        target = 0.
        @parameters begin
            beta::vector[3]
            sigma::real(lower=0)
        end
        @model begin
            log_earn ~ normal(@broadcasted(beta[1] + beta[2] * log_height + beta[3] * male), sigma);
        end
        return target
    end
end
julia_implementation(::Val{:wells_dae_model}; N, switched, dist, arsenic, educ, kwargs...) = begin 
    dist100 = @. dist / 100.
    educ4 = @. educ / 4.
    X = hcat(dist100, arsenic, educ4)
    @stan begin 
        target = 0.
        @parameters begin
            alpha::real
            beta::vector[3]
        end
        @model begin
            switched ~ bernoulli_logit_glm(X, alpha, beta);
        end
        return target
    end
end
julia_implementation(::Val{:arK}; K, T, y, kwargs...) = begin 
    @stan begin 
        target = 0.
        @parameters begin
            alpha::real
            beta::vector[K]
            sigma::real(lower=0)
        end
        @model begin
            alpha ~ normal(0, 10);
            beta ~ normal(0, 10);
            sigma ~ cauchy(0, 2.5);
            for t in K+1:T
                mu = alpha
                for k in 1:K
                    mu += beta[k] * y[t-k]
                end
                y[t] ~ normal(mu, sigma)
            end
        end
        return target
    end
end
julia_implementation(::Val{:wells_interaction_model}; N, switched, dist, arsenic, kwargs...) = begin 
    dist100 = @. dist / 100.
    inter = @. dist100 * arsenic
    X = hcat(dist100, arsenic, inter)
    @stan begin 
        target = 0.
        @parameters begin
            alpha::real
            beta::vector[3]
        end
        @model begin
            switched ~ bernoulli_logit_glm(X, alpha, beta);
        end
        return target
    end
end
julia_implementation(::Val{:radon_pooled}; N, floor_measure, log_radon, kwargs...) = begin 
    @stan begin 
        target = 0.
        @parameters begin
            alpha::real
            beta::real
            sigma_y::real(lower=0)
        end
        @model begin
            sigma_y ~ normal(0, 1);
            alpha ~ normal(0, 10);
            beta ~ normal(0, 10);

            log_radon ~ normal(@broadcasted(alpha + beta * floor_measure), sigma_y)
        end
        return target
    end
end
julia_implementation(::Val{:logearn_interaction}; N, earn, height, male, kwargs...) = begin 
        log_earn = @. log(earn)
        inter = @. height * male
@stan begin 
            target = 0.
            @parameters begin
                beta::vector[4]
                sigma::real(lower=0)
            end
            @model begin
                log_earn ~ normal(@broadcasted(beta[1] + beta[2] * height + beta[3] * male + beta[4] * inter), sigma);
            end
            return target
        end
end
julia_implementation(::Val{:logmesquite_logvolume}; N, weight, diam1, diam2, canopy_height, kwargs...) = begin 
        log_weight = @. log(weight);
        log_canopy_volume = @. log(diam1 * diam2 * canopy_height);
@stan begin 
            target = 0.
            @parameters begin
                beta::vector[2]
                sigma::real(lower=0)
            end
            @model begin
                log_weight ~ normal(@broadcasted(beta[1] + beta[2] * log_canopy_volume), sigma);
            end
            return target
    end
end
julia_implementation(::Val{:garch11}; T, y, sigma1, kwargs...) = begin 
    @stan begin 
        target = 0.
        @parameters begin
            mu::real
            alpha0::real(lower=0.)
            alpha1::real(lower=0., upper=1.)
            beta1::real(lower=0., upper=1. - alpha1)
        end
        @model begin
            sigma = sigma1
            y[1] ~ normal(mu, sigma)
            for t in 2:T
                sigma = sqrt(alpha0 + alpha1 * square(y[t - 1] - mu) + beta1 * square(sigma))
                y[t] ~ normal(mu, sigma)
            end
        end
        return target
    end
end
julia_implementation(::Val{:eight_schools_centered}; J, y, sigma, kwargs...) = begin 
    @stan begin 
        target = 0.
        @parameters begin
            theta::vector[J]
            mu::real
            tau::real(lower=0)
        end
        @model begin
            tau ~ cauchy(0, 5);
            theta ~ normal(mu, tau);
            y ~ normal(theta, sigma);
            mu ~ normal(0, 5);
        end
        return target
    end
end
julia_implementation(::Val{:kidscore_interaction}; N, kid_score, mom_iq, mom_hs, kwargs...) = begin 
        inter = @. mom_hs * mom_iq;
@stan begin 
            target = 0.
            @parameters begin
                beta::vector[4]
                sigma::real(lower=0)
            end
            @model begin
                sigma ~ cauchy(0, 2.5);
                kid_score ~ normal(@broadcasted(beta[1] + beta[2] * mom_hs + beta[3] * mom_iq + beta[4] * inter), sigma);
            end
            return target
    end
end
julia_implementation(::Val{:mesquite}; N, weight, diam1, diam2, canopy_height, total_height, density, group, kwargs...) = begin 
    @stan begin 
        target = 0.
        @parameters begin
            beta::vector[7]
            sigma::real(lower=0)
        end
        @model begin
            weight ~ normal(@broadcasted(beta[1] + beta[2] * diam1 + beta[3] * diam2
            + beta[4] * canopy_height + beta[5] * total_height
            + beta[6] * density + beta[7] * group), sigma);
        end
        return target
    end
end
julia_implementation(::Val{:gp_regr}; N, x, y, kwargs...) = begin 
@stan begin 
            target = 0.
            @parameters begin
                rho::real(lower=0)
                alpha::real(lower=0)
                sigma::real(lower=0)
            end
            @model begin
                cov = gp_exp_quad_cov(x, alpha, rho) + diag_matrix(rep_vector(sigma, N));
                # L_cov = cholesky_decompose(cov);
  
                rho ~ gamma(25, 4);
                alpha ~ normal(0, 2);
                sigma ~ normal(0, 1);
                
                y ~ multi_normal(rep_vector(0., N), cov);
                # Think about how to do this
                # y ~ multi_normal_cholesky(rep_vector(0, N), L_cov);
            end
            return target
    end
end
julia_implementation(::Val{:kidscore_mom_work}; N, kid_score, mom_work, kwargs...) = begin 
        work2 = @. Float64(mom_work == 2)
        work3 = @. Float64(mom_work == 3)
        work4 = @. Float64(mom_work == 4)
@stan begin 
            target = 0.
            @parameters begin
                beta::vector[4]
                sigma::real(lower=0)
            end
            @model begin
                kid_score ~ normal(@broadcasted(beta[1] + beta[2] * work2 + beta[3] * work3
                + beta[4] * work4), sigma);
            end
            return target
        end
end
julia_implementation(::Val{:Rate_2_model}; n1, n2, k1, k2, kwargs...) = begin 
    @stan begin 
        target = 0.
        @parameters begin
            theta1::real(lower=0,upper=1)
            theta2::real(lower=0,upper=1)
        end
        delta = theta1 - theta2
        @model begin
            theta1 ~ beta(1, 1)
            theta2 ~ beta(1, 1)
            k1 ~ binomial(n1, theta1)
            k2 ~ binomial(n2, theta2)
        end
        return target
    end
end
julia_implementation(::Val{:wells_interaction_c_model}; N, switched, dist, arsenic, kwargs...) = begin 
        c_dist100 = @. (dist - $mean(dist)) / 100.0;
        c_arsenic = @. arsenic - $mean(arsenic);
        inter = @. c_dist100 * c_arsenic
        X = hcat(c_dist100, c_arsenic, inter)
@stan begin 
            target = 0.
            @parameters begin
                alpha::real
                beta::vector[3]
            end
            @model begin
                switched ~ bernoulli_logit_glm(X, alpha, beta);
            end
            return target
        end
end
julia_implementation(::Val{:radon_county_intercept}; N, J, county_idx, floor_measure, log_radon, kwargs...) = begin 
    @stan begin 
        target = 0.
        @parameters begin
            alpha::vector[J]
            beta::real
            sigma_y::real(lower=0)
        end
        @model begin
            sigma_y ~ normal(0, 1);
            alpha ~ normal(0, 10);
            beta ~ normal(0, 10);
            for n in 1:N
                mu = alpha[county_idx[n]] + beta * floor_measure[n];
                target += normal_lpdf(log_radon[n], mu, sigma_y);
            end
        end
        return target
    end
end
julia_implementation(::Val{:kidscore_interaction_c}; N, kid_score, mom_iq, mom_hs, kwargs...) = begin 
        c_mom_hs = @. mom_hs - $mean(mom_hs);
        c_mom_iq = @. mom_iq - $mean(mom_iq);
        inter = @. c_mom_hs * c_mom_iq;
@stan begin 
            target = 0.
            @parameters begin
                beta::vector[4]
                sigma::real(lower=0)
            end
            @model begin
                kid_score ~ normal(@broadcasted(beta[1] + beta[2] * c_mom_hs + beta[3] * c_mom_iq + beta[4] * inter), sigma);
            end
            return target
    end
end
julia_implementation(::Val{:kidscore_interaction_c2}; N, kid_score, mom_iq, mom_hs, kwargs...) = begin 
        c_mom_hs = @. mom_hs - .5;
        c_mom_iq = @. mom_iq - 100;
        inter = @. c_mom_hs * c_mom_iq;
@stan begin 
            target = 0.
            @parameters begin
                beta::vector[4]
                sigma::real(lower=0)
            end
            @model begin
                kid_score ~ normal(@broadcasted(beta[1] + beta[2] * c_mom_hs + beta[3] * c_mom_iq + beta[4] * inter), sigma);
            end
            return target
        end
end
julia_implementation(::Val{:gp_pois_regr}; N, x, k, kwargs...) = begin 
        nugget = diag_matrix(rep_vector(1e-10, N))
@stan begin 
            target = 0.
            @parameters begin
                rho::real(lower=0)
                alpha::real(lower=0)
                f_tilde::vector[N]
            end
            @transformed_parameters begin 
                cov = gp_exp_quad_cov(x, alpha, rho)
                cov .+= nugget;
                L_cov = cholesky_decompose(cov);
                f = L_cov * f_tilde;
            end
            @model begin
                rho ~ gamma(25, 4);
                alpha ~ normal(0, 2);
                f_tilde ~ normal(0, 1);
                
                k ~ poisson_log(f);
            end
            return target
    end
end
julia_implementation(::Val{:surgical_model}; N, r, n, kwargs...) = begin 
    @stan begin 
        target = 0.
        @parameters begin
            mu::real
            sigmasq::real(lower=0)
            b::vector[N]
        end
        @transformed_parameters begin 
            sigma = sqrt(sigmasq)
            p = @broadcasted inv_logit(b)
        end
        @model begin
            mu ~ normal(0.0, 1000.0);
            sigmasq ~ inv_gamma(0.001, 0.001);
            b ~ normal(mu, sigma);
            r ~ binomial_logit(n, b);
        end
        return target
    end
end
julia_implementation(::Val{:wells_dae_c_model}; N, switched, dist, arsenic, educ, kwargs...) = begin 
    c_dist100 = @. (dist - $mean(dist)) / 100.0;
    c_arsenic = @. arsenic - $mean(arsenic);
    da_inter = @. c_dist100 * c_arsenic;
    educ4 = @. educ / 4.
    X = hcat(c_dist100, c_arsenic, da_inter, educ4)
    @stan begin 
        target = 0.
        @parameters begin
            alpha::real
            beta::vector[4]
        end
        @model begin
            switched ~ bernoulli_logit_glm(X, alpha, beta);
        end
        return target
    end
end
julia_implementation(::Val{:Rate_4_model}; n, k, kwargs...) = begin 
    @stan begin 
        target = 0.
        @parameters begin
            theta::real(lower=0,upper=1)
            thetaprior::real(lower=0,upper=1)
        end
        @model begin
            theta ~ beta(1, 1);
            thetaprior ~ beta(1, 1);
            k ~ binomial(n, theta);
        end
        return target
    end
end
julia_implementation(::Val{:logearn_interaction_z}; N, earn, height, male, kwargs...) = begin 
        log_earn = @. log(earn)
        z_height = @. (height - $mean(height)) / $sd(height);
        inter = @. z_height * male
@stan begin 
            target = 0.
            @parameters begin
                beta::vector[4]
                sigma::real(lower=0)
            end
            @model begin
                log_earn ~ normal(@broadcasted(beta[1] + beta[2] * z_height + beta[3] * male + beta[4] * inter), sigma);
            end
            return target
    end
end
julia_implementation(::Val{:normal_mixture}; N, y, kwargs...) = begin 
    @stan begin 
        target = 0.
        @parameters begin
            theta::real(lower=0,upper=1)
            mu::vector[2]
        end
        @model begin
            theta ~ uniform(0, 1); 
            for k in 1:2
                mu[k] ~ normal(0, 10);
            end
            for n in 1:N
                target += log_mix(theta, normal_lpdf(y[n], mu[1], 1.0),
                                normal_lpdf(y[n], mu[2], 1.0));
            end
        end
        return target
    end
end
julia_implementation(::Val{:radon_partially_pooled_centered}; N, J, county_idx, log_radon, kwargs...) = begin 
    @stan begin 
        target = 0.
        @parameters begin
            alpha::vector[J]
            mu_alpha::real
            sigma_alpha::real(lower=0)
            sigma_y::real(lower=0)
        end
        @model begin
            sigma_y ~ normal(0, 1);
            sigma_alpha ~ normal(0, 1);
            mu_alpha ~ normal(0, 10);

            alpha ~ normal(mu_alpha, sigma_alpha);
            for n in 1:N
                mu = alpha[county_idx[n]];
                target += normal_lpdf(log_radon[n], mu, sigma_y);
            end
        end
        return target
    end
end
julia_implementation(::Val{:kidscore_interaction_z}; N, kid_score, mom_iq, mom_hs, kwargs...) = begin 
        c_mom_hs = @. (mom_hs - $mean(mom_hs)) / (2 * $sd(mom_hs));
        c_mom_iq = @. (mom_iq - $mean(mom_iq)) / (2 * $sd(mom_iq));
        inter = @. c_mom_hs * c_mom_iq;
@stan begin 
            target = 0.
            @parameters begin
                beta::vector[4]
                sigma::real(lower=0)
            end
            @model begin
                kid_score ~ normal(@broadcasted(beta[1] + beta[2] * c_mom_hs + beta[3] * c_mom_iq + beta[4] * inter), sigma);
            end
            return target
        end
end
julia_implementation(::Val{:kilpisjarvi}; N, x, y, xpred, pmualpha, psalpha, pmubeta, psbeta, kwargs...) = begin 
        x_ = x
@stan begin 
            target = 0.
            @parameters begin
                alpha::real
                beta::real
                sigma::real(lower=0)
            end
            @model begin
                alpha ~ normal(pmualpha, psalpha);
                beta ~ normal(pmubeta, psbeta);
                y ~ normal(@broadcasted(alpha + beta * x_), sigma);
            end
            return target
    end
end
julia_implementation(::Val{:wells_daae_c_model}; N, switched, dist, arsenic, assoc, educ, kwargs...) = begin 
        c_dist100 = @. (dist - $mean(dist)) / 100.0;
        c_arsenic = @. arsenic - $mean(arsenic);
        da_inter = @. c_dist100 * c_arsenic;
        educ4 = @. educ / 4.
        X = hcat(c_dist100, c_arsenic, da_inter, assoc, educ4)
@stan begin 
            target = 0.
            @parameters begin
                alpha::real
                beta::vector[5]
            end
            @model begin
                switched ~ bernoulli_logit_glm(X, alpha, beta);
            end
            return target
    end
end
julia_implementation(::Val{:eight_schools_noncentered}; J, y, sigma, kwargs...) = begin 
    @stan begin 
        target = 0.
        @parameters begin
            theta_trans::vector[J]
            mu::real
            tau::real(lower=0)
        end
        theta = @broadcasted(theta_trans * tau + mu);
        @model begin
            theta_trans ~ normal(0, 1);
            y ~ normal(theta, sigma);
            mu ~ normal(0, 5);
            tau ~ cauchy(0, 5);
        end
        return target
    end
end
julia_implementation(::Val{:Rate_5_model}; n1, n2, k1, k2, kwargs...) = begin 
    @stan begin 
        target = 0.
        @parameters begin
            theta::real(lower=0,upper=1)
        end
        @model begin
            theta ~ beta(1, 1);
            k1 ~ binomial(n1, theta);
            k2 ~ binomial(n2, theta);
        end
        return target
    end
end
julia_implementation(::Val{:dugongs_model}; N, x, Y, kwargs...) = begin 
        x_ = x
@stan begin 
            target = 0.
            @parameters begin
                alpha::real
                beta::real
                lambda::real(lower=.5,upper=1.)
                tau::real(lower=0.)
            end
            @transformed_parameters begin
                sigma = 1. / sqrt(tau);
                U3 = logit(lambda);
            end
            @model begin
                for i in 1:N
                    m = alpha - beta * pow(lambda, x_[i]);
                    Y[i] ~ normal(m, sigma);
                end
                
                alpha ~ normal(0.0, 1000.);
                beta ~ normal(0.0, 1000.);
                lambda ~ uniform(.5, 1.);
                tau ~ gamma(.0001, .0001);
            end
            return target
    end
end
julia_implementation(::Val{:irt_2pl}; I, J, y, kwargs...) = begin 
    @stan begin 
        target = 0.
        @parameters begin
            sigma_theta::real(lower=0);
            theta::vector[J];

            sigma_a::real(lower=0);
            a::vector(lower=0)[I];

            mu_b::real;
            sigma_b::real(lower=0);
            b::vector[I];
        end
        @model begin
            sigma_theta ~ cauchy(0, 2);
            theta ~ normal(0, sigma_theta);
            
            sigma_a ~ cauchy(0, 2);
            a ~ lognormal(0, sigma_a);
            
            mu_b ~ normal(0, 5);
            sigma_b ~ cauchy(0, 2);
            b ~ normal(mu_b, sigma_b);
            
            for i in 1:I
                y[i,:] ~ bernoulli_logit(@broadcasted(a[i] * (theta - b[i])));
            end
        end
        return target
    end
end
julia_implementation(::Val{:logmesquite_logva}; N, weight, diam1, diam2, canopy_height, group, kwargs...) = begin 
        log_weight = @. log(weight);
        log_canopy_volume = @. log(diam1 * diam2 * canopy_height);
        log_canopy_area = @. log(diam1 * diam2)
@stan begin 
            target = 0.
            @parameters begin
                beta::vector[4]
                sigma::real(lower=0)
            end
            @model begin
                log_weight ~ normal(@broadcasted(beta[1] + beta[2] * log_canopy_volume
                + beta[3] * log_canopy_area + beta[4] * group), sigma);
            end
            return target
    end
end
julia_implementation(::Val{:radon_variable_slope_centered}; J, N, county_idx, floor_measure, log_radon, kwargs...) = begin 
    @stan begin 
        target = 0.
        @parameters begin
            alpha::real
            beta::vector[J]
            mu_beta::real
            sigma_beta::real(lower=0)
            sigma_y::real(lower=0)
        end
        @model begin
            alpha ~ normal(0, 10);
            sigma_y ~ normal(0, 1);
            sigma_beta ~ normal(0, 1);
            mu_beta ~ normal(0, 10);
            
            beta ~ normal(mu_beta, sigma_beta);
            for n in 1:N
                mu = alpha + floor_measure[n] * beta[county_idx[n]];
                target += normal_lpdf(log_radon[n], mu, sigma_y);
            end
        end
        return target
    end
end
julia_implementation(::Val{:radon_variable_intercept_centered}; J, N, county_idx, floor_measure, log_radon, kwargs...) = begin 
    @stan begin 
        target = 0.
        @parameters begin
            alpha::vector[J]
            beta::real
            mu_alpha::real
            sigma_alpha::real(lower=0)
            sigma_y::real(lower=0)
        end
        @model begin
            sigma_y ~ normal(0, 1);
            sigma_alpha ~ normal(0, 1);
            mu_alpha ~ normal(0, 10);
            beta ~ normal(0, 10);
            
            alpha ~ normal(mu_alpha, sigma_alpha);
            for n in 1:N
                mu = alpha[county_idx[n]] + floor_measure[n] * beta;
                target += normal_lpdf(log_radon[n], mu, sigma_y);
            end
        end
        return target
    end
end
julia_implementation(::Val{:seeds_stanified_model}; I, n, N, x1, x2, kwargs...) = begin 
        x1x2 = @. x1 * x2;
@stan begin 
            target = 0.
            @parameters begin
                alpha0::real;
                alpha1::real;
                alpha12::real;
                alpha2::real;
                b::vector[I];
                sigma::real(lower=0);
            end
            @model begin
                alpha0 ~ normal(0.0, 1.0);
                alpha1 ~ normal(0.0, 1.0);
                alpha2 ~ normal(0.0, 1.0);
                alpha12 ~ normal(0.0, 1.0);
                sigma ~ cauchy(0, 1);
                
                b ~ normal(0.0, sigma);
                n ~ binomial_logit(N,
                                   @broadcasted(alpha0 + alpha1 * x1 + alpha2 * x2 + alpha12 * x1x2 + b));
            end
            return target
    end
end
julia_implementation(::Val{:state_space_stochastic_level_stochastic_seasonal}; n, y, x, w, kwargs...) = begin 
        x_ = x
        mu_lower = mean(y) - 3 * sd(y)
        mu_upper = mean(y) + 3 * sd(y)
@stan begin 
            target = 0.
            @parameters begin
                mu::vector(lower=mu_lower, upper=mu_upper)[n]
                seasonal::vector[n]
                beta::real
                lambda::real
                sigma::positive_ordered[3]
            end
            @model begin
                for t in 12:n
                    seasonal[t] ~ normal(-sum(seasonal[t-11:t-1]), sigma[1]);
                end
                
                for t in 2:n
                    mu[t] ~ normal(mu[t - 1], sigma[2]);
                end
                
                y ~ normal(@broadcasted(mu + beta * x_ + lambda * w + seasonal), sigma[3]);
                
                sigma ~ student_t(4, 0, 1);
            end
            return target
        end
end
julia_implementation(::Val{:radon_partially_pooled_noncentered}; N, J, county_idx, log_radon, kwargs...) = begin 
    @stan begin 
        target = 0.
        @parameters begin
            alpha_raw::vector[J]
            mu_alpha::real
            sigma_alpha::real(lower=0)
            sigma_y::real(lower=0)
        end
        alpha = @.(mu_alpha + sigma_alpha * alpha_raw);
        @model begin
            sigma_y ~ normal(0, 1);
            sigma_alpha ~ normal(0, 1);
            mu_alpha ~ normal(0, 10);
            alpha_raw ~ normal(0, 1);

            for n in 1:N
                mu = alpha[county_idx[n]];
                target += normal_lpdf(log_radon[n], mu, sigma_y);
            end
        end
        return target
    end
end
julia_implementation(::Val{:seeds_model}; I, n, N, x1, x2, kwargs...) = begin 
        x1x2 = @. x1 * x2;
@stan begin 
            target = 0.
            @parameters begin
                alpha0::real;
                alpha1::real;
                alpha12::real;
                alpha2::real;
                tau::real(lower=0);
                b::vector[I];
            end
            sigma = 1.0 / sqrt(tau);
            @model begin
                alpha0 ~ normal(0.0, 1.0E3);
                alpha1 ~ normal(0.0, 1.0E3);
                alpha2 ~ normal(0.0, 1.0E3);
                alpha12 ~ normal(0.0, 1.0E3);
                tau ~ gamma(1.0E-3, 1.0E-3);
                
                b ~ normal(0.0, sigma);
                n ~ binomial_logit(N,
                                   @broadcasted(alpha0 + alpha1 * x1 + alpha2 * x2 + alpha12 * x1x2 + b));
            end
            return target
        end
end
julia_implementation(::Val{:arma11}; T, y, kwargs...) = begin 
    @stan begin 
        target = 0.
        @parameters begin
            mu::real
            phi::real
            theta::real
            sigma::real(lower=0)
        end
        @model begin
            mu ~ normal(0, 10);
            phi ~ normal(0, 2);
            theta ~ normal(0, 2);
            sigma ~ cauchy(0, 2.5);
            nu = mu + phi * mu
            err = y[1] - nu
            err ~ normal(0, sigma);
            for t in 2:T
                nu = mu + phi * y[t-1] + theta * err
                err = y[t] - nu
                err ~ normal(0, sigma);
            end
        end
        return target
    end
end
julia_implementation(::Val{:radon_hierarchical_intercept_centered}; J, N, county_idx, log_uppm, floor_measure, log_radon, kwargs...) = begin 
    @stan begin 
        target = 0.
        @parameters begin
            alpha::vector[J]
            beta::vector[2]
            mu_alpha::real
            sigma_alpha::real(lower=0)
            sigma_y::real(lower=0)
        end
        @model begin
            sigma_alpha ~ normal(0, 1);
            sigma_y ~ normal(0, 1);
            mu_alpha ~ normal(0, 10);
            beta ~ normal(0, 10);
            
            alpha ~ normal(mu_alpha, sigma_alpha);
            for n in 1:N
                muj = alpha[county_idx[n]] + log_uppm[n] * beta[1]
                mu = muj + floor_measure[n] * beta[2];
                target += normal_lpdf(log_radon[n], mu, sigma_y);
            end
        end
        return target
    end
end
julia_implementation(::Val{:seeds_centered_model}; I, n, N, x1, x2, kwargs...) = begin 
        x1x2 = @. x1 * x2;
@stan begin 
            target = 0.
            @parameters begin
                alpha0::real;
                alpha1::real;
                alpha12::real;
                alpha2::real;
                c::vector[I];
                sigma::real(lower=0);
            end
            b = @. c - $mean(c);
            @model begin
                alpha0 ~ normal(0.0, 1.0);
                alpha1 ~ normal(0.0, 1.0);
                alpha2 ~ normal(0.0, 1.0);
                alpha12 ~ normal(0.0, 1.0);
                sigma ~ cauchy(0, 1);
                
                c ~ normal(0.0, sigma);
                n ~ binomial_logit(N,
                                   @broadcasted(alpha0 + alpha1 * x1 + alpha2 * x2 + alpha12 * x1x2 + b));
            end
            return target
    end
end
julia_implementation(::Val{:pilots}; N, n_groups, n_scenarios, group_id, scenario_id, y, kwargs...) = begin 
    @stan begin 
        target = 0.
        @parameters begin
            a::vector[n_groups];
            b::vector[n_scenarios];
            mu_a::real;
            mu_b::real;
            sigma_a::real(lower=0, upper=100);
            sigma_b::real(lower=0, upper=100);
            sigma_y::real(lower=0, upper=100);
        end
        y_hat = @broadcasted(a[group_id] + b[scenario_id]);
        @model begin
            mu_a ~ normal(0, 1);
            a ~ normal(10 * mu_a, sigma_a);
            
            mu_b ~ normal(0, 1);
            b ~ normal(10 * mu_b, sigma_b);
            
            y ~ normal(y_hat, sigma_y);
        end
        return target
    end
end
julia_implementation(::Val{:wells_dae_inter_model}; N, switched, dist, arsenic, educ, kwargs...) = begin 
        c_dist100 = @. (dist - $mean(dist)) / 100.0;
        c_arsenic = @. arsenic - $mean(arsenic);
        c_educ4 = @. (educ - $mean(educ)) / 4.
        da_inter = @. c_dist100 * c_arsenic;
        de_inter = @. c_dist100 * c_educ4;
        ae_inter = @. c_arsenic * c_educ4;
        X = hcat(c_dist100, c_arsenic, c_educ4, da_inter, de_inter, ae_inter, )
@stan begin 
            target = 0.
            @parameters begin
                alpha::real
                beta::vector[6]
            end
            @model begin
                switched ~ bernoulli_logit_glm(X, alpha, beta);
            end
            return target
    end
end
julia_implementation(::Val{:radon_variable_slope_noncentered}; J, N, county_idx, floor_measure, log_radon, kwargs...) = begin 
    @stan begin 
        target = 0.
        @parameters begin
            alpha::real
            beta_raw::vector[J]
            mu_beta::real
            sigma_beta::real(lower=0)
            sigma_y::real(lower=0)
        end
        beta = @. mu_beta + sigma_beta * beta_raw;
        @model begin
            alpha ~ normal(0, 10);
            sigma_y ~ normal(0, 1);
            sigma_beta ~ normal(0, 1);
            mu_beta ~ normal(0, 10);
            beta_raw ~ normal(0, 1);

            for n in 1:N
                mu = alpha + floor_measure[n] * beta[county_idx[n]];
                target += normal_lpdf(log_radon[n], mu, sigma_y);
            end
        end
        return target
    end
end
julia_implementation(::Val{:radon_variable_intercept_slope_centered}; J, N, county_idx, floor_measure, log_radon, kwargs...) = begin 
@stan begin 
            target = 0.
            @parameters begin
                sigma_y::real(lower=0)
                sigma_alpha::real(lower=0)
                sigma_beta::real(lower=0)
                alpha::vector[J]
                beta::vector[J]
                mu_alpha::real
                mu_beta::real
            end
            @model begin
                sigma_y ~ normal(0, 1);
                sigma_beta ~ normal(0, 1);
                sigma_alpha ~ normal(0, 1);
                mu_alpha ~ normal(0, 10);
                mu_beta ~ normal(0, 10);
                
                alpha ~ normal(mu_alpha, sigma_alpha);
                beta ~ normal(mu_beta, sigma_beta);
                for n in 1:N
                    mu = alpha[county_idx[n]] + floor_measure[n] * beta[county_idx[n]];
                    target += normal_lpdf(log_radon[n], mu, sigma_y);
                end
            end
            return target
        end
end
julia_implementation(::Val{:radon_variable_intercept_noncentered}; J, N, county_idx, floor_measure, log_radon, kwargs...) = begin 
    @stan begin 
        target = 0.
        @parameters begin
            alpha_raw::vector[J]
            beta::real
            mu_alpha::real
            sigma_alpha::real(lower=0)
            sigma_y::real(lower=0)
        end
        alpha = @. mu_alpha + sigma_alpha * alpha_raw;
        @model begin
            sigma_y ~ normal(0, 1);
            sigma_alpha ~ normal(0, 1);
            mu_alpha ~ normal(0, 10);
            beta ~ normal(0, 10);
            alpha_raw ~ normal(0, 1);

            for n in 1:N
                mu = alpha[county_idx[n]] + floor_measure[n] * beta;
                target += normal_lpdf(log_radon[n], mu, sigma_y);
            end
        end
        return target
    end
end
julia_implementation(::Val{:GLM_Poisson_model}; n, C, year, kwargs...) = begin 
        year_squared = year .^ 2
        year_cubed = year .^ 3
@stan begin 
            target = 0.
            @parameters begin
                alpha::real(lower=-20, upper=+20)
                beta1::real(lower=-10, upper=+10)
                beta2::real(lower=-10, upper=+10)
                beta3::real(lower=-10, upper=+10)
            end
            log_lambda = @broadcasted(alpha + beta1 * year + beta2 * year_squared + beta3 * year_cubed)
            @model begin
                C ~ poisson_log(log_lambda);
            end
            return target
    end
end
julia_implementation(::Val{:ldaK5}; V, M, N, w, doc, alpha, beta, kwargs...) = begin 
    @stan begin 
        target = 0.
        @parameters begin
            theta::simplex[M,5];
            phi::simplex[5,V];
        end
        @model @views begin
            for m in 1:M
                theta[m] ~ dirichlet(alpha);
            end
            for k in 1:5
                phi[k] ~ dirichlet(beta);
            end
            for n in 1:N
                gamma = @broadcasted(log(theta[doc[n], :]) + log(phi[:, w[n]]))
                target += log_sum_exp(gamma);
            end
        end
        return target
    end
end
julia_implementation(::Val{:dogs}; n_dogs, n_trials, y, kwargs...) = begin 
    @stan begin 
        target = 0.
        @parameters begin
            beta::vector[3];
        end
        @model begin
            beta ~ normal(0, 100);
            for i in 1:n_dogs
                n_avoid = 0
                n_shock = 0
                for j in 1:n_trials
                    p = beta[1] + beta[2] * n_avoid + beta[3] * n_shock
                    y[i, j] ~ bernoulli_logit(p);
                    n_avoid += 1 - y[i,j]
                    n_shock += y[i,j]
                end
            end
        end
        return target
    end
end
julia_implementation(::Val{:nes}; N, partyid7, real_ideo, race_adj, educ1, gender, income, age_discrete, kwargs...) = begin 
        age30_44 = @. Float64(age_discrete == 2);
        age45_64 = @. Float64(age_discrete == 3);
        age65up = @. Float64(age_discrete == 4);
@stan begin 
            target = 0.
            @parameters begin
                beta::vector[9]
                sigma::real(lower=0)
            end
            @model begin
                partyid7 ~ normal(@broadcasted(beta[1] + beta[2] * real_ideo + beta[3] * race_adj
                + beta[4] * age30_44 + beta[5] * age45_64
                + beta[6] * age65up + beta[7] * educ1 + beta[8] * gender
                + beta[9] * income), sigma);
            end
            return target
        end
end
julia_implementation(::Val{:dogs_log}; n_dogs, n_trials, y, kwargs...) = begin 
    @assert size(y) == (n_dogs, n_trials)
@stan begin 
        target = 0.
        @parameters begin
            beta::vector[2];
        end
        @model begin
            beta[1] ~ uniform(-100, 0);
            beta[2] ~ uniform(0, 100);
            for i in 1:n_dogs
                n_avoid = 0
                n_shock = 0
                for j in 1:n_trials
                    p = inv_logit(beta[1] * n_avoid + beta[2] * n_shock)
                    y[i, j] ~ bernoulli(p);
                    n_avoid += 1 - y[i,j]
                    n_shock += y[i,j]
                end
            end
        end
        return target
    end
end
julia_implementation(::Val{:GLM_Binomial_model};nyears, C, N, year, kwargs...) = begin 
        year_squared = year .^ 2
@stan begin 
            target = 0.
            @parameters begin
                alpha::real
                beta1::real
                beta2::real
            end
            logit_p = @broadcasted alpha + beta1 * year + beta2 * year_squared;
            @model begin
                alpha ~ normal(0, 100)
                beta1 ~ normal(0, 100)
                beta2 ~ normal(0, 100)
                C ~ binomial_logit(N, logit_p)
            end
            return target
    end
end
julia_implementation(::Val{:radon_hierarchical_intercept_noncentered}; J, N, county_idx, log_uppm, floor_measure, log_radon, kwargs...) = begin 
    @stan begin 
        target = 0.
        @parameters begin
            alpha_raw::vector[J]
            beta::vector[2]
            mu_alpha::real
            sigma_alpha::real(lower=0)
            sigma_y::real(lower=0)
        end
        alpha = @. mu_alpha + sigma_alpha * alpha_raw;
        @model begin
            sigma_alpha ~ normal(0, 1);
            sigma_y ~ normal(0, 1);
            mu_alpha ~ normal(0, 10);
            beta ~ normal(0, 10);
            alpha_raw ~ normal(0, 1);

            for n in 1:N
                muj = alpha[county_idx[n]] + log_uppm[n] * beta[1]
                mu = muj + floor_measure[n] * beta[2];
                target += normal_lpdf(log_radon[n], mu, sigma_y);
            end
        end
        return target
    end
end
julia_implementation(::Val{:logmesquite_logvash}; N, weight, diam1, diam2, canopy_height, total_height, group, kwargs...) = begin 
        log_weight = @. log(weight);
        log_canopy_volume = @. log(diam1 * diam2 * canopy_height);
        log_canopy_area = @. log(diam1 * diam2)
        log_canopy_shape = @. log(diam1 / diam2);
        log_total_height = @. log(total_height);
@stan begin 
            target = 0.
            @parameters begin
                beta::vector[6]
                sigma::real(lower=0)
            end
            @model begin
                log_weight ~ normal(@broadcasted(beta[1] + beta[2] * log_canopy_volume
                + beta[3] * log_canopy_area
                + beta[4] * log_canopy_shape
                + beta[5] * log_total_height + beta[6] * group), sigma);
            end
            return target
        end
end
julia_implementation(::Val{:ldaK2}; V, M, N, w, doc, kwargs...) = begin 
        K = 2
        alpha = fill(1, K)
        beta = fill(1, V)
@stan begin 
            target = 0.
            @parameters begin
                theta::simplex[M,K];
                phi::simplex[K,V];
            end
            @model begin
                for m in 1:M
                  theta[m] ~ dirichlet(alpha);
                end
                for k in 1:K
                  phi[k] ~ dirichlet(beta);
                end
                for n in 1:N
                    gamma = @broadcasted log(theta[doc[n], :]) + log(phi[:, w[n]])
                  target += log_sum_exp(gamma);
                end
            end
            return target
    end
end
julia_implementation(::Val{:logmesquite}; N, weight, diam1, diam2, canopy_height, total_height, density, group, kwargs...) = begin 
        log_weight = @. log(weight);
        log_diam1 = @. log(diam1);
        log_diam2 = @. log(diam2);
        log_canopy_height = @. log(canopy_height);
        log_total_height = @. log(total_height);
        log_density = @. log(density);
@stan begin 
            target = 0.
            @parameters begin
                beta::vector[7]
                sigma::real(lower=0)
            end
            @model begin
                log_weight ~ normal(@broadcasted(beta[1] + beta[2] * log_diam1 + beta[3] * log_diam2
                + beta[4] * log_canopy_height
                + beta[5] * log_total_height + beta[6] * log_density
                + beta[7] * group), sigma);
            end
            return target
    end
end
julia_implementation(::Val{:rats_model}; N, Npts, rat, x, y, xbar, kwargs...) = begin 
        x_ = x
@stan begin 
            target = 0.
            @parameters begin
                alpha::vector[N];
                beta::vector[N];
                
                mu_alpha::real;
                mu_beta::real;
                sigma_y::real(lower=0);
                sigma_alpha::real(lower=0);
                sigma_beta::real(lower=0);
            end
            @model begin
                mu_alpha ~ normal(0, 100);
                mu_beta ~ normal(0, 100);
                alpha ~ normal(mu_alpha, sigma_alpha);
                beta ~ normal(mu_beta, sigma_beta);
                for n in 1:Npts
                  irat = rat[n];
                  y[n] ~ normal(alpha[irat] + beta[irat] * (x_[n] - xbar), sigma_y);
                end
            end
            return target
    end
end
julia_implementation(::Val{:logmesquite_logvas}; N, weight, diam1, diam2, canopy_height, total_height, density, group, kwargs...) = begin 
        log_weight = @. log(weight);
        log_canopy_volume = @. log(diam1 * diam2 * canopy_height);
        log_canopy_area = @. log(diam1 * diam2)
        log_canopy_shape = @. log(diam1 / diam2);
        log_total_height = @. log(total_height);
        log_density = @. log(density);
@stan begin 
            target = 0.
            @parameters begin
                beta::vector[7]
                sigma::real(lower=0)
            end
            @model begin
                log_weight ~ normal(@broadcasted(beta[1] + beta[2] * log_canopy_volume
                + beta[3] * log_canopy_area
                + beta[4] * log_canopy_shape
                + beta[5] * log_total_height + beta[6] * log_density
                + beta[7] * group), sigma);
            end
            return target
    end
end
julia_implementation(::Val{:lsat_model}; N, R, T, culm, response, kwargs...) = begin 
        r = zeros(Int64, (T, N))
        for j in 1:culm[1], k in 1:T
            r[k, j] = response[1, k];
        end
        for i in 2:R
            for j in (culm[i-1]+1):culm[i], k in 1:T
                r[k, j] = response[i, k];
            end
        end
        ones = fill(1., N)
@stan begin 
            target = 0.
            @parameters begin
                alpha::vector[T];
                theta::vector[N];
                beta::real(lower=0);
            end
            @model begin
                alpha ~ normal(0, 100.);
                theta ~ normal(0, 1);
                beta ~ normal(0.0, 100.);
                for k in 1:T
                    r[k,:] ~ bernoulli_logit(beta * theta - alpha[k] * ones);
                end
            end
            return target
    end
end
julia_implementation(::Val{:radon_variable_intercept_slope_noncentered}; J, N, county_idx, floor_measure, log_radon, kwargs...) = begin 
    @stan begin 
        target = 0.
        @parameters begin
            sigma_y::real(lower=0)
            sigma_alpha::real(lower=0)
            sigma_beta::real(lower=0)
            alpha_raw::vector[J]
            beta_raw::vector[J]
            mu_alpha::real
            mu_beta::real
        end
        alpha = @. mu_alpha + sigma_alpha * alpha_raw;
        beta = @. mu_beta + sigma_beta * beta_raw;
        @model begin
            sigma_y ~ normal(0, 1);
            sigma_beta ~ normal(0, 1);
            sigma_alpha ~ normal(0, 1);
            mu_alpha ~ normal(0, 10);
            mu_beta ~ normal(0, 10);
            alpha_raw ~ normal(0, 1);
            beta_raw ~ normal(0, 1);

            for n in 1:N
                mu = alpha[county_idx[n]] + floor_measure[n] * beta[county_idx[n]];
                target += normal_lpdf(log_radon[n], mu, sigma_y);
            end
        end
        return target
    end
end
julia_implementation(::Val{:GLMM_Poisson_model};n, C, year) = begin 
        year_squared = year .^ 2
        year_cubed = year .^ 3
@stan begin
            target = 0.
            @parameters begin
                alpha::real(lower=-20, upper=+20)
                beta1::real(lower=-10, upper=+10)
                beta2::real(lower=-10, upper=+20)
                beta3::real(lower=-10, upper=+10)
                eps::vector[n]
                sigma::real(lower=0, upper=5)
            end
            log_lambda = @broadcasted alpha + beta1 * year + beta2 * year_squared + beta3 * year_cubed + eps
            @model begin
                C ~ poisson_log(log_lambda)
                eps ~ normal(0, sigma)
            end
            return target
    end
end
julia_implementation(::Val{:GLMM1_model};nsite, nobs, obs, obsyear, obssite, misyear, missite, kwargs...) = begin 
    @stan begin 
        target = 0.
        @parameters begin
            alpha::vector[nsite]
            mu_alpha::real
            sd_alpha::real(lower=0,upper=5)
        end
    #   log_lambda = rep_matrix(alpha', nyear);
        @model begin
            alpha ~ normal(mu_alpha, sd_alpha)
            mu_alpha ~ normal(0, 10)
            for i in 1:nobs
                # obs[i] ~ poisson_log(log_lambda[obsyear[i], obssite[i]])
                obs[i] ~ poisson_log(alpha[obssite[i]])
            end
        end
        return target
    end
end
julia_implementation(::Val{:hier_2pl}; I, J, N, ii, jj, y, kwargs...) = begin 
    @stan begin 
        target = 0.
        @parameters begin
            theta::vector[J];
            xi1::vector[I];
            xi2::vector[I];
            mu::vector[2];
            tau::vector(lower=0)[2];
            L_Omega::cholesky_factor_corr[2]
        end
        xi = hcat(xi1, xi2)
        alpha = @. exp(xi1);
        beta = xi2;
        @model begin
            L_Sigma = diag_pre_multiply(tau, L_Omega);
            for i in 1:I
                target += multi_normal_cholesky_lpdf(xi[i, :], mu, L_Sigma);
            end
            theta ~ normal(0, 1);
            L_Omega ~ lkj_corr_cholesky(4);
            mu[1] ~ normal(0, 1);
            tau[1] ~ exponential(.1);
            mu[2] ~ normal(0, 5);
            tau[2] ~ exponential(.1);
            y ~ bernoulli_logit(alpha[ii] .* (theta[jj] - beta[ii]));
        end
        return target
    end
end
julia_implementation(::Val{:dogs_hierarchical}; n_dogs, n_trials, y, kwargs...) = begin 
        J = n_dogs;
        T = n_trials;
        prev_shock = zeros((J,T));
        prev_avoid = zeros((J,T));
        
        for j in 1:J
            prev_shock[j, 1] = 0;
            prev_avoid[j, 1] = 0;
            for t in 2:T
                prev_shock[j, t] = prev_shock[j, t - 1] + y[j, t - 1];
                prev_avoid[j, t] = prev_avoid[j, t - 1] + 1 - y[j, t - 1];
            end
        end
@stan begin 
            target = 0.
            @parameters begin
                a::real(lower=0, upper=1);
                b::real(lower=0, upper=1);
            end
            @model begin
                y ~ bernoulli(@broadcasted(a ^ prev_shock * b ^ prev_avoid));
            end
            return target
    end
end


julia_implementation(::Val{:dogs_nonhierarchical}; n_dogs, n_trials, y, kwargs...) = begin 
    J = n_dogs;
    T = n_trials;
    prev_shock = zeros((J,T));
    prev_avoid = zeros((J,T));
    
    for j in 1:J
        prev_shock[j, 1] = 0;
        prev_avoid[j, 1] = 0;
        for t in 2:T
            prev_shock[j, t] = prev_shock[j, t - 1] + y[j, t - 1];
            prev_avoid[j, t] = prev_avoid[j, t - 1] + 1 - y[j, t - 1];
        end
    end
    return @stan begin end
    @stan begin 
        @parameters begin
            mu_logit_ab::vector[2]
            sigma_logit_ab::vector(lower=0)[2]
            L_logit_ab::cholesky_factor_corr[2]
            z::matrix[J, 2]
        end
        @model @views begin
            logit_ab = rep_vector(1, J) * mu_logit_ab'
                                    + z * diag_pre_multiply(sigma_logit_ab, L_logit_ab);
            Omega_logit_ab = L_logit_ab * L_logit_ab';
            Sigma_logit_ab = quad_form_diag(Omega_logit_ab,
                                                            sigma_logit_ab);
            a = inv_logit(logit_ab[ : , 1]);
            b = inv_logit(logit_ab[ : , 2]);
            y ~ bernoulli(@broadcasted(a ^ prev_shock * b ^ prev_avoid));
            mu_logit_ab ~ logistic(0, 1);
            sigma_logit_ab ~ normal(0, 1);
            L_logit_ab ~ lkj_corr_cholesky(2);
            (z) ~ normal(0, 1);
        end
    end
end
julia_implementation(::Val{:M0_model}; M, T, y, kwargs...) = begin
        @assert size(y) == (M, T) 
        C = 0
        s = zeros(Int64, M)
        for i in 1:M
            s[i] = sum(y[i, :])
            if s[i] > 0
                C += 1
            end
        end
@stan begin 
            target = 0.
            @parameters begin
                omega::real(lower=0,upper=1)
                p::real(lower=0,upper=1)
            end
            @model begin
                for i in 1:M
                    if s[i] > 0
                        target += bernoulli_lpmf(1, omega) + binomial_lpmf(s[i], T, p)
                    else
                        target += log_sum_exp(bernoulli_lpmf(1, omega)
                            + binomial_lpmf(0, T, p),
                            bernoulli_lpmf(0, omega))
                    end
                end
            end
            return target
    end
end
julia_implementation(::Val{:diamonds}; N, Y, K, X, prior_only, kwargs...) = begin
        Kc = K - 1;
        Xc = zeros((N, Kc))
        means_X = zeros(Kc)
        for i in 2:K
            means_X[i - 1] = mean(X[ : , i]);
            @. Xc[ : , i - 1] = X[ : , i] - means_X[i - 1];
        end
@stan begin 
            target = 0.
            @parameters begin
                b::vector[Kc];
                Intercept::real;
                sigma::real(lower=0.)
            end
            @model begin
                target += normal_lpdf(b, 0., 1.);
                target += student_t_lpdf(Intercept, 3., 8., 10.);
                target += student_t_lpdf(sigma, 3., 0., 10.)# - 1 * student_t_lccdf(0, 3, 0, 10);
                if !(prior_only == 1)
                    target += normal_id_glm_lpdf(Y, Xc, Intercept, b, sigma);
                end
            end
            return target
    end
end
julia_implementation(::Val{:Mt_model}; M, T, y, kwargs...) = begin 
        @assert size(y) == (M, T) 
        C = 0
        s = zeros(Int64, M)
        for i in 1:M
            s[i] = sum(y[i, :])
            if s[i] > 0
                C += 1
            end
        end
@stan begin 
            target = 0.
            @parameters begin
                omega::real(lower=0,upper=1)
                p::vector(lower=0,upper=1)[T]
            end
            @model @views begin
                for i in 1:M
                    if s[i] > 0
                        target += bernoulli_lpmf(1, omega) + bernoulli_lpmf(y[i,:], p)
                    else
                        target += log_sum_exp(bernoulli_lpmf(1, omega)
                            + bernoulli_lpmf(y[i,:], p),
                            bernoulli_lpmf(0, omega))
                    end
                end
            end
            return target
    end
end
julia_implementation(::Val{:election88_full}; 
    N,
    n_age,
    n_age_edu,
    n_edu,
    n_region_full,
    n_state,
    age,
    age_edu,
    black,
    edu,
    female,
    region_full,
    state,
    v_prev_full,
    y,
    kwargs...) = begin 
@stan begin 
            target = 0.
            @parameters begin
                a::vector[n_age];
                b::vector[n_edu];
                c::vector[n_age_edu];
                d::vector[n_state];
                e::vector[n_region_full];
                beta::vector[5];
                sigma_a::real(lower=0, upper=100);
                sigma_b::real(lower=0, upper=100);
                sigma_c::real(lower=0, upper=100);
                sigma_d::real(lower=0, upper=100);
                sigma_e::real(lower=0, upper=100);
            end
            y_hat = @broadcasted (beta[1] + beta[2] * black + beta[3] * female
            + beta[5] * female * black + beta[4] * v_prev_full
            + a[age] + b[edu] + c[age_edu] + d[state]
            + e[region_full])
            @model begin
                a ~ normal(0, sigma_a);
                b ~ normal(0, sigma_b);
                c ~ normal(0, sigma_c);
                d ~ normal(0, sigma_d);
                e ~ normal(0, sigma_e);
                beta ~ normal(0, 100);
                y ~ bernoulli_logit(y_hat);
            end
            return target
        end
end
julia_implementation(::Val{:nn_rbm1bJ10}; N, M, x, K, y, kwargs...) = begin 
            J = 10
            nu_alpha = 0.5;
            s2_0_alpha = (0.05 / M ^ (1 / nu_alpha)) ^ 2;
            nu_beta = 0.5;
            s2_0_beta = (0.05 / J ^ (1 / nu_beta)) ^ 2;
            
            ones = rep_vector(1., N);
            x1 = append_col(ones, x);

    return @stan begin end
    @stan begin 
        target = 0.
        @parameters begin
            sigma2_alpha::real(lower=0);
            sigma2_beta::real(lower=0);
            alpha::matrix[M, J];
            beta::matrix[J, K - 1];
            alpha1::row_vector[J];
            beta1::row_vector[K - 1];
        end
        @model @views begin
            v = append_col(
                ones,
                append_col(
                    ones,
                    tanh.(x1 * append_row(alpha1, alpha))
                ) * append_row(beta1, beta)
            );
            alpha1 ~ normal(0, 1);
            beta1 ~ normal(0, 1);
            sigma2_alpha ~ inv_gamma(nu_alpha / 2, nu_alpha * s2_0_alpha / 2);
            sigma2_beta ~ inv_gamma(nu_beta / 2, nu_beta * s2_0_beta / 2);
            
            (alpha) ~ normal(0, sqrt(sigma2_alpha));
            (beta) ~ normal(0, sqrt(sigma2_beta));
            for n in 1:N
                y[n] ~ categorical_logit(v[n, :]);
            end
        end
        return target
    end
end
julia_implementation(::Val{:nn_rbm1bJ100}; N, M, x, K, y, kwargs...) = begin 
    J = 100
    nu_alpha = 0.5;
    s2_0_alpha = (0.05 / M ^ (1 / nu_alpha)) ^ 2;
    nu_beta = 0.5;
    s2_0_beta = (0.05 / J ^ (1 / nu_beta)) ^ 2;
    
    ones = rep_vector(1., N);
    x1 = append_col(ones, x);
    return @stan begin end
    @stan begin 
        target = 0.
        @parameters begin
            sigma2_alpha::real(lower=0);
            sigma2_beta::real(lower=0);
            alpha::matrix[M, J];
            beta::matrix[J, K - 1];
            alpha1::row_vector[J];
            beta1::row_vector[K - 1];
        end
        @model @views begin
            v = append_col(
                ones,
                append_col(
                    ones,
                    tanh.(x1 * append_row(alpha1, alpha))
                ) * append_row(beta1, beta)
            );
            alpha1 ~ normal(0, 1);
            beta1 ~ normal(0, 1);
            sigma2_alpha ~ inv_gamma(nu_alpha / 2, nu_alpha * s2_0_alpha / 2);
            sigma2_beta ~ inv_gamma(nu_beta / 2, nu_beta * s2_0_beta / 2);
            
            (alpha) ~ normal(0, sqrt(sigma2_alpha));
            (beta) ~ normal(0, sqrt(sigma2_beta));
            for n in 1:N
                y[n] ~ categorical_logit(v[n, :]);
            end
        end
        return target
    end
end
julia_implementation(::Val{:Survey_model}; nmax, m, k) = begin 
    nmin = maximum(k)
    @stan begin 
        @parameters begin 
            theta::real(lower=0, upper=1)
        end
        @model begin 
            target += log_sum_exp([
                n < nmin ? log(1.0 / nmax) - Inf : log(1.0 / nmax) + binomial_lpmf(k, n, theta)
                for n in 1:nmax
            ])
        end
    end
end

julia_implementation(::Val{:sir}; N_t, t, y0, stoi_hat, B_hat) = begin 
    y0 = collect(Float64, y0)
    simple_SIR(t, y, theta, x_r, x_i)  = begin
        dydt = zero(y)
        
        dydt[1] = -theta[1] * y[4] / (y[4] + theta[2]) * y[1];
        dydt[2] = theta[1] * y[4] / (y[4] + theta[2]) * y[1] - theta[3] * y[2];
        dydt[3] = theta[3] * y[2];
        dydt[4] = theta[4] * y[2] - theta[5] * y[4];
        
        return dydt;
    end
    t0 = 0.
    kappa = 1000000.
    @stan begin 
        @parameters begin 
            beta::real(lower=0)
            gamma::real(lower=0)
            xi::real(lower=0)
            delta::real(lower=0)
        end
        @model @views begin 
            y = integrate_ode_rk45(simple_SIR, y0, t0, t, [beta, kappa, gamma, xi, delta]);

            beta ~ cauchy(0, 2.5);
            gamma ~ cauchy(0, 1);
            xi ~ cauchy(0, 25);
            delta ~ cauchy(0, 1);

            stoi_hat[1] ~ poisson(y0[1] - y[1, 1]);
            for n in 2:N_t
                stoi_hat[n] ~ poisson(max(1e-16, y[n - 1, 1] - y[n, 1]));
            end
            
            B_hat ~ lognormal(@broadcasted(nothrow_log(y[:, 4])), 0.15);
        
        end
    end
end

julia_implementation(::Val{:lotka_volterra}; N, ts, y_init, y) = begin 
    dz_dt(t, z, theta, x_r, x_i)  = begin
        u, v = z
        
        alpha, beta, gamma, delta = theta

        du_dt = (alpha - beta * v) * u
        dv_dt = (-gamma +delta * u) * v
        [du_dt, dv_dt]
    end
    @stan begin 
        @parameters begin 
            theta::real(lower=0)[4]
            z_init::real(lower=0)[2]
            sigma::real(lower=0)[2]
        end
        @model @views begin 
            z = integrate_ode_rk45(dz_dt, z_init, 0, ts, theta,
                missing, missing, 1e-5, 1e-3, 5e2);

            theta[[1,3]] ~ normal(1, 0.5);
            theta[[2,4]] ~ normal(0.05, 0.05);
            sigma ~ lognormal(-1, 1);
            z_init ~ lognormal(log(10), 1);
            y_init ~ lognormal(@broadcasted(log(z_init)), sigma);
            y ~ lognormal(@broadcasted(nothrow_log(z)), sigma');
        end
    end
end

julia_implementation(::Val{:soil_incubation}; totalC_t0, t0, N_t, ts, eCO2mean, kwargs...) = begin 
    two_pool_feedback(t, C, theta, x_r, x_i)  = begin
        k1, k2, alpha21, alpha12 = theta
        [
            -k1 * C[1] + alpha12 * k2 * C[2]
            -k2 * C[2] + alpha21 * k1 * C[1]
        ]
    end
    evolved_CO2(N_t, t0, ts, gamma, totalC_t0, k1, k2, alpha21, alpha12) = begin 
        C_t0 = [gamma * totalC_t0, (1 - gamma) * totalC_t0]
        theta = k1, k2, alpha21, alpha12
        C_hat = integrate_ode_rk45(two_pool_feedback, C_t0, t0, ts, theta)
        totalC_t0 .- sum.(eachrow(C_hat))
    end
    @stan begin 
        @parameters begin 
            k1::real(lower=0)
            k2::real(lower=0)
            alpha21::real(lower=0)
            alpha12::real(lower=0)
            gamma::real(lower=0, upper=1)
            sigma::real(lower=0)
        end
        @model @views begin 
            eCO2_hat = evolved_CO2(N_t, t0, ts, gamma, totalC_t0, k1, k2, alpha21, alpha12)
            gamma ~ beta(10, 1); 
            k1 ~ normal(0, 1);
            k2 ~ normal(0, 1);
            alpha21 ~ normal(0, 1);
            alpha12 ~ normal(0, 1);
            sigma ~ cauchy(0, 1);
            eCO2mean ~ normal(eCO2_hat, sigma);
        end
    end
end

julia_implementation(::Val{:one_comp_mm_elim_abs}; t0, D, V, N_t, times, C_hat) = begin 
    one_comp_mm_elim_abs(t, y, theta, x_r, x_i)  = begin
        k_a, K_m, V_m = theta
        D, V = x_r
        dose = 0.
        elim = (V_m / V) * y[1] / (K_m + y[1]);
        if t > 0
            dose = exp(-k_a * t) * D * k_a / V;
        end
        [dose - elim]
    end
    C0 = [0.]
    x_r = [D,V]
    x_i = missing
    @stan begin 
        @parameters begin 
            k_a::real(lower=0)
            K_m::real(lower=0)
            V_m::real(lower=0)
            sigma::real(lower=0)
        end
        @model @views begin 
            theta = [k_a, K_m, V_m]
            C = integrate_ode_bdf(one_comp_mm_elim_abs, C0, t0, times, theta, x_r, x_i)
            k_a ~ cauchy(0, 1);
            K_m ~ cauchy(0, 1);
            V_m ~ cauchy(0, 1);
            sigma ~ cauchy(0, 1);
            
            C_hat ~ lognormal(@broadcasted(nothrow_log(C[:, 1])), sigma);
        end
    end
end


julia_implementation(::Val{:bym2_offset_only}; N, N_edges, node1, node2, y, E, scaling_factor, kwargs...) = begin 
        log_E = @. log(E)
@stan begin 
            target = 0.
            @parameters begin
                beta0::real;
                sigma::real(lower=0);
                rho::real(lower=0, upper=1);
                theta::vector[N];
                phi::vector[N];
            end
            convolved_re = @broadcasted(sqrt(1 - rho) * theta + sqrt(rho / scaling_factor) * phi)
            @model @views begin
                y ~ poisson_log(@broadcasted(log_E + beta0 + convolved_re * sigma));
                
                target += -0.5 * dot_self(phi[node1] - phi[node2]);
                
                beta0 ~ normal(0, 1);
                theta ~ normal(0, 1);
                sigma ~ normal(0, 1);
                rho ~ beta(0.5, 0.5);
                sum(phi) ~ normal(0, 0.001 * N);
            end
            return target
        end
end
julia_implementation(::Val{:bones_model}; nChild, nInd, gamma, delta, ncat, grade, kwargs...) = begin 
        # error(ncat)
@stan begin 
            target = 0.
            @parameters begin
                theta::real[nChild]
            end
            @model @views begin
                theta ~ normal(0.0, 36.);
                for i in 1:nChild
                    for j in 1:nInd
                        Q = vcat(
                            1., 
                            @.(inv_logit(delta[j] * (theta[i] - gamma[j, 1:(ncat[j]-1)]))),
                            0.
                        )
                        p = Q[1:end-1] - Q[2:end]
                        if grade[i, j] != -1
                            target += log(p[grade[i, j]]);
                        end
                    end
                end
            end
            return target
    end
end
julia_implementation(::Val{:logistic_regression_rhs}; n, d, y, x, scale_icept,
    scale_global,
    nu_global,
    nu_local, 
    slab_scale,
    slab_df, kwargs...) = begin 
        x = Matrix{Float64}(x)
@stan begin 
            target = 0.
            @parameters begin
                beta0::real;
                z::vector[d];
                tau::real(lower=0.);
                lambda::vector(lower=0.)[d];
                caux::real(lower=0.);
            end
            c = slab_scale * sqrt(caux);
            lambda_tilde = @broadcasted sqrt(c ^ 2 * square(lambda) / (c ^ 2 + tau ^ 2 * square(lambda)));
            beta = @. z * lambda_tilde * tau;
            @model begin
                z ~ std_normal();
                lambda ~ student_t(nu_local, 0., 1.);
                tau ~ student_t(nu_global, 0, 2. * scale_global);
                caux ~ inv_gamma(0.5 * slab_df, 0.5 * slab_df);
                beta0 ~ normal(0., scale_icept);
                
                y ~ bernoulli_logit_glm(x, beta0, beta);
            end
            return target
        end
end

julia_implementation(::Val{:hmm_example}; N, K, y, kwargs...) = begin 
    @stan begin 
        target = 0.
        @parameters begin
            theta1::simplex[K]
            theta2::simplex[K]
            mu::positive_ordered[K]
        end
        theta = hcat(theta1, theta2)'
        @model begin
            target += normal_lpdf(mu[1], 3, 1);
            target += normal_lpdf(mu[2], 10, 1);
            gamma = @. normal_lpdf(y[1], mu, 1) 
            for t in 2:N
                gamma .= [
                    log_sum_exp(@broadcasted(
                        gamma + log(theta[:, k]) 
                        + normal_lpdf(y[t], mu[k], 1) 
                    )) for k in 1:K
                ]
            end
            target += log_sum_exp(gamma)
        end
        return target
    end
end
julia_implementation(::Val{:Mb_model}; M, T, y, kwargs...) = begin 
        @assert size(y) == (M, T) 
        C = 0
        s = zeros(Int64, M)
        for i in 1:M
            s[i] = sum(y[i, :])
            if s[i] > 0
                C += 1
            end
        end
@stan begin 
            target = 0.
            @parameters begin
                omega::real(lower=0.,upper=1.)
                p::real(lower=0.,upper=1.)
                c::real(lower=0.,upper=1.)
            end
            @model @views begin
                p_eff = hcat(
                    fill(p, M), 
                    @. (1 - y[:, 1:end-1]) * p + y[:, 1:end-1] * c
                )
                for i in 1:M
                    if s[i] > 0
                        target += bernoulli_lpmf(1, omega) + bernoulli_lpmf(y[i, :], p_eff[i, :])
                    else
                        target += log_sum_exp(bernoulli_lpmf(1, omega)
                            + bernoulli_lpmf(y[i, :], p_eff[i, :]),
                            bernoulli_lpmf(0, omega))
                    end
                end
            end
            return target
        end
end
julia_implementation(::Val{:Mh_model}; M, T, y, kwargs...) = begin 
        @assert size(y) == (M, ) 
        C = 0
        for i in 1:M
            if y[i] > 0
                C += 1
            end
        end
@stan begin 
            target = 0.
            @parameters begin
                omega::real(lower=0,upper=1)
                mean_p::real(lower=0,upper=1)
                sigma::real(lower=0,upper=5)
                eps_raw::vector[M]
            end
            eps = @. logit(mean_p) + sigma * eps_raw
            @model begin
                eps_raw ~ normal(0, 1)
                for i in 1:M
                    if y[i] > 0
                        target += bernoulli_lpmf(1, omega) + binomial_logit_lpmf(y[i], T, eps[i])
                    else
                        target += log_sum_exp(bernoulli_lpmf(1, omega)
                            + binomial_logit_lpmf(0, T, eps[i]),
                            bernoulli_lpmf(0, omega))
                    end
                end
            end
            return target
        end
end
julia_implementation(::Val{:Mth_model}; M, T, y, kwargs...) = begin 
        @assert size(y) == (M, T) 
        C = 0
        s = zeros(Int64, M)
        for i in 1:M
            s[i] = sum(y[i, :])
            if s[i] > 0
                C += 1
            end
        end
        @stan begin 
            target = 0.
            @parameters begin
                omega::real(lower=0.,upper=1.)
                mean_p::vector(lower=0.,upper=1.)[T]
                sigma::real(lower=0., upper=5.)
                eps_raw::vector[M]
            end
            @model @views begin
                logit_p = @. logit(mean_p)' .+ sigma * eps_raw
                eps_raw ~ normal(0., 1.)
                for i in 1:M
                    if s[i] > 0
                        target += bernoulli_lpmf(1, omega) + bernoulli_logit_lpmf(y[i, :], logit_p[i, :])
                    else
                        target += log_sum_exp(bernoulli_lpmf(1, omega)
                            + bernoulli_logit_lpmf(y[i, :], logit_p[i, :]),
                            bernoulli_lpmf(0, omega))
                    end
                end
            end
            return target
    end
end
julia_implementation(::Val{Symbol("2pl_latent_reg_irt")}; I, J, N, ii, jj, y, K, W, kwargs...) = begin 
    obtain_adjustments(W) = begin 
        local M, K = size(W)
        adj = zeros((2, K))
        adj[1,1] = 0
        adj[2, 1] = 1
        for k in 2:K
            min_w = minimum(W[:,k])
            max_w = maximum(W[:,k])
            minmax_count = sum(w->w in (min_w, max_w), W[:, k])
            if minmax_count == M
                adj[1, k] = mean(W[:, k]);
                adj[2, k] = max_w - min_w;
            else
                adj[1, k] = mean(W[:, k]);
                adj[2, k] = sd(W[:, k]) * 2;
            end
        end
        return adj
    end
    adj = obtain_adjustments(W);
    W_adj = @. ((W - adj[1:1,:])/adj[2:2,:])
    @stan begin 
        target = 0.
        @parameters begin
            alpha::vector(lower=0)[I];
            beta_free::vector[I - 1] ;
            theta::vector[J];
            lambda_adj::vector[K];
        end
        beta = vcat(beta_free, -sum(beta_free))
        @model @views begin
            alpha ~ lognormal(1, 1);
            target += normal_lpdf(beta, 0, 3);
            lambda_adj ~ student_t(3, 0, 1);
            theta ~ normal(W_adj * lambda_adj, 1);
            y ~ bernoulli_logit(@broadcasted(alpha[ii] * theta[jj] - beta[ii]));
        end
        return target
    end
end
julia_implementation(::Val{:Mtbh_model}; M, T, y, kwargs...) = begin 
        @assert size(y) == (M, T) 
        C = 0
        s = zeros(Int64, M)
        for i in 1:M
            s[i] = sum(y[i, :])
            if s[i] > 0
                C += 1
            end
        end
@stan begin 
            target = 0.
            @parameters begin
                omega::real(lower=0,upper=1)
                mean_p::vector(lower=0,upper=1)[T]
                gamma::real
                sigma::real(lower=0, upper=3)
                eps_raw::vector[M]
            end
            @model @views begin
                eps = @. sigma * eps_raw
                alpha = @. logit(mean_p)
                logit_p = hcat(
                    @.(alpha[1] + eps),
                    @.(alpha[2:end]' + eps + gamma * y[:, 1:end-1])
                )
                gamma ~ normal(0, 10)
                eps_raw ~ normal(0, 1)
                for i in 1:M
                    if s[i] > 0
                        target += bernoulli_lpmf(1, omega) + bernoulli_logit_lpmf(y[i, :], logit_p[i, :])
                    else
                        target += log_sum_exp(bernoulli_lpmf(1, omega)
                            + bernoulli_logit_lpmf(y[i, :], logit_p[i, :]),
                            bernoulli_lpmf(0, omega))
                    end
                end
            end
            return target
    end
end
julia_implementation(::Val{:multi_occupancy}; J, K, n, X, S, kwargs...) = begin 
    cov_matrix_2d(sigma, rho) = begin
        rv12 = sigma[1] * sigma[2] * rho
        [
            square(sigma[1]) rv12
            rv12 square(sigma[2])
        ]
    end
    lp_observed(X, K, logit_psi, logit_theta) = log_inv_logit(logit_psi) + binomial_logit_lpmf(X, K, logit_theta);
    lp_unobserved(K, logit_psi, logit_theta) = log_sum_exp(
        lp_observed(0, K, logit_psi, logit_theta),
        log1m_inv_logit(logit_psi)
    );
    lp_never_observed(J, K, logit_psi, logit_theta, Omega) = begin
        lp_unavailable = bernoulli_lpmf(0, Omega);
        lp_available = bernoulli_lpmf(1, Omega) + J * lp_unobserved(K, logit_psi, logit_theta);
        return log_sum_exp(lp_unavailable, lp_available);
    end
@stan begin 
            target = 0.
            @parameters begin
                alpha::real
                beta::real
                Omega::real(lower=0, upper=+1)
                rho_uv::real(lower=-1, upper=+1)
                sigma_uv::vector(lower=0,upper=+Inf)[2]
                uv1::vector[S]
                uv2::vector[S]
            end
            @transformed_parameters begin 
                uv = hcat(uv1, uv2)
                logit_psi = @. uv1 + alpha
                logit_theta = @. uv2 + beta
            end
            @model begin
                alpha ~ cauchy(0, 2.5);
                beta ~ cauchy(0, 2.5);
                sigma_uv ~ cauchy(0, 2.5);
                (rho_uv + 1) / 2 ~ beta(2, 2);
                target += multi_normal_lpdf(uv, rep_vector(0., 2), cov_matrix_2d(sigma_uv, rho_uv));
                Omega ~ beta(2, 2);
  
                for i in 1:n 
                    1 ~ bernoulli(Omega); 
                    for j in 1:J
                        if X[i, j] > 0
                            target += lp_observed(X[i, j], K, logit_psi[i], logit_theta[i]);
                        else
                            target += lp_unobserved(K, logit_psi[i], logit_theta[i]);
                        end
                    end
                end
                for i in (n + 1):S
                  target += lp_never_observed(J, K, logit_psi[i], logit_theta[i], Omega);
                end
            end
            return target
        end
end
julia_implementation(::Val{:losscurve_sislob};
growthmodel_id, 
n_data,
n_time,
n_cohort,
cohort_id,
t_idx,
cohort_maxtime,
t_value,
premium,
loss,
kwargs...) = begin 
    growth_factor_weibull(t, omega, theta) = begin
        return 1 - exp(-(t / theta) ^ omega);
    end

    growth_factor_loglogistic(t, omega, theta) = begin
        pow_t_omega = t ^ omega;
        return pow_t_omega / (pow_t_omega + theta ^ omega);
    end
    @stan begin 
            target = 0.
            @parameters begin
                omega::real(lower=0);
                theta::real(lower=0);
                
                LR::vector(lower=0)[n_cohort];
                
                mu_LR::real;
                sd_LR::real(lower=0);
                
                loss_sd::real(lower=0);
            end
            gf = if growthmodel_id == 1
                @. growth_factor_weibull(t_value, omega, theta)
            else
                @. growth_factor_loglogistic(t_value, omega, theta)
            end
            @model @views begin
                mu_LR ~ normal(0, 0.5);
                sd_LR ~ lognormal(0, 0.5);
                
                LR ~ lognormal(mu_LR, sd_LR);
                
                loss_sd ~ lognormal(0, 0.7);
                
                omega ~ lognormal(0, 0.5);
                theta ~ lognormal(0, 0.5);
                
                loss ~ normal(@broadcasted(LR[cohort_id] * premium[cohort_id] * gf[t_idx]), (loss_sd * premium)[cohort_id]);
            end
            return target
    end
end

julia_implementation(::Val{:accel_splines}; N,Y,Ks,Xs,knots_1,Zs_1_1, Ks_sigma, Xs_sigma,knots_sigma_1,Zs_sigma_1_1,prior_only, kwargs...) = begin 
@stan begin 
            target = 0.
            @parameters begin
                Intercept::real;
                bs::vector[Ks];
                zs_1_1::vector[knots_1];
                sds_1_1::real(lower=0);
                Intercept_sigma::real;
                bs_sigma::vector[Ks_sigma];
                zs_sigma_1_1::vector[knots_sigma_1];
                sds_sigma_1_1::real(lower=0);
            end
            s_1_1 = @. sds_1_1 * zs_1_1
            s_sigma_1_1 = @. sds_sigma_1_1 * zs_sigma_1_1;
            @model begin
                mu = Intercept .+ Xs * bs + Zs_1_1 * s_1_1;
                sigma = exp.(Intercept_sigma .+ Xs_sigma * bs_sigma
                                  + Zs_sigma_1_1 * s_sigma_1_1);
                target += student_t_lpdf(Intercept, 3, -13, 36);
                target += normal_lpdf(zs_1_1, 0, 1);
                target += student_t_lpdf(sds_1_1, 3, 0, 36)
                        #   - 1 * student_t_lccdf(0, 3, 0, 36);
                target += student_t_lpdf(Intercept_sigma, 3, 0, 10);
                target += normal_lpdf(zs_sigma_1_1, 0, 1);
                target += student_t_lpdf(sds_sigma_1_1, 3, 0, 36)
                        #   - 1 * student_t_lccdf(0, 3, 0, 36);
                if !(prior_only == 1)
                    target += normal_lpdf(Y, mu, sigma);
                end
            end
            return target
        end
end
julia_implementation(::Val{Symbol("grsm_latent_reg_irt")}; I, J, N, ii, jj, y, K, W, kwargs...) = begin 
    rsm(y, theta, beta, kappa) = begin
      unsummed = vcat(0, theta .- beta .- kappa);
      probs = softmax(cumulative_sum(unsummed));
      return categorical_lpmf(y + 1, probs);
    end
    obtain_adjustments(W) = begin 
        local M, K = size(W)
        adj = zeros((2, K))
        adj[1,1] = 0
        adj[2, 1] = 1
        for k in 2:K
            min_w = minimum(W[:,k])
            max_w = maximum(W[:,k])
            minmax_count = sum(w->w in (min_w, max_w), W[:, k])
            if minmax_count == M
                adj[1, k] = mean(W[:, k]);
                adj[2, k] = max_w - min_w;
            else
                adj[1, k] = mean(W[:, k]);
                adj[2, k] = sd(W[:, k]) * 2;
            end
        end
        return adj
    end
        m = maximum(y)
        adj = obtain_adjustments(W);
        W_adj = @. ((W - adj[1:1,:])/adj[2:2,:])
@stan begin 
            target = 0.
            @parameters begin
                alpha::vector(lower=0)[I];
                beta_free::vector[I - 1] ;
                kappa_free::vector[m - 1] ;
                theta::vector[J];
                lambda_adj::vector[K];
            end
            beta = vcat(beta_free, -sum(beta_free))
            kappa = vcat(kappa_free, -sum(kappa_free))
            @model @views begin
                alpha ~ lognormal(1, 1);
                target += normal_lpdf(beta, 0, 3);
                target += normal_lpdf(kappa, 0, 3);
                theta ~ normal(W_adj * lambda_adj, 1);
                lambda_adj ~ student_t(3, 0, 1);
                for n in 1:N
                    target += rsm(y[n], theta[jj[n]] .* alpha[ii[n]], beta[ii[n]], kappa)
                end
            end
            return target
    end
end
julia_implementation(::Val{:prophet};
    T,
    K,
    t,
    cap,
    y,
    S,
    t_change,
    X,
    sigmas,
    tau,
    trend_indicator,
    s_a,
    s_m, 
    kwargs...) = begin 
    get_changepoint_matrix(t, t_change, T, S) = begin
        local A = rep_matrix(0, T, S);
        a_row = rep_row_vector(0, S);
        cp_idx = 1;
        
        for i in 1:T
          while ((cp_idx <= S) && (t[i] >= t_change[cp_idx])) 
            a_row[cp_idx] = 1;
            cp_idx = cp_idx + 1;
          end
          A[i,:] = a_row;
        end
        return A;
      end
      
      
      logistic_gamma(k, m, delta, t_change, S) = begin
        k_s = append_row(k, k + cumulative_sum(delta));
        
        m_pr = m; 
        for i in 1:S
          gamma[i] = (t_change[i] - m_pr) * (1 - k_s[i] / k_s[i + 1]);
          m_pr = m_pr + gamma[i]; 
        end
        return gamma;
      end
      
      logistic_trend(k, m, delta, t, cap, A, t_change, S) = begin
        gamma = logistic_gamma(k, m, delta, t_change, S);
        return cap .* inv_logit.((k .+ A * delta) .* (t .- m .- A * gamma));
      end
      
      linear_trend(k, m, delta, t, A, t_change) = begin
        return (k .+ A * delta) .* t .+ (m .+ A * (-t_change .* delta));
      end
        A = get_changepoint_matrix(t, t_change, T, S)
@stan begin 
            target = 0.
            @parameters begin
                k::real
                m::real
                delta::vector[S]
                sigma_obs::real(lower=0)
                beta::vector[K]
            end
            @model begin
                k ~ normal(0, 5);
                m ~ normal(0, 5);
                delta ~ double_exponential(0, tau);
                sigma_obs ~ normal(0, 0.5);
                beta ~ normal(0, sigmas);
                
                if trend_indicator == 0
                    y ~ normal(linear_trend(k, m, delta, t, A, t_change)
                             .* (1 .+ X * (beta .* s_m)) + X * (beta .* s_a), sigma_obs);
                elseif trend_indicator == 1
                    y ~ normal(logistic_trend(k, m, delta, t, cap, A, t_change, S)
                             .* (1 .+ X * (beta .* s_m)) + X * (beta .* s_a), sigma_obs);
                end
            end
            return target
        end
end

julia_implementation(::Val{:hmm_gaussian}; T, K, y, kwargs...) = begin 
    @stan begin 
        target = 0.
        @parameters begin
            pi1::simplex[K]
            A::simplex[K,K]
            mu::ordered[K]
            sigma::vector(lower=0)[K]
        end
        @model @views begin
            logalpha = log.(pi1) .+ normal_lpdf(y[1], mu, sigma);
            for t in 2 : T
                logalpha .= ([
                    log_sum_exp(@broadcasted(
                        logalpha + log(A[:, j]) + normal_lpdf(y[t], mu[j], sigma[j])
                    ))
                    for j in 1:K
                ]);
            end
            target += log_sum_exp(logalpha);
        end
        return target
    end
end

julia_implementation(::Val{:hmm_drive_0}; K, N, u, v, alpha, kwargs...) = begin 
    @stan begin 
        target = 0.
        @parameters begin
            theta1::simplex[K]
            theta2::simplex[K]
            phi::positive_ordered[K]
            lambda::positive_ordered[K]
        end
        theta = hcat(theta1, theta2)'
        @model @views begin
            for k in 1:K
              target += dirichlet_lpdf(theta[k, :], alpha[k, :]);
            end
            target += normal_lpdf(phi[1], 0, 1);
            target += normal_lpdf(phi[2], 3, 1);
            target += normal_lpdf(lambda[1], 0, 1);
            target += normal_lpdf(lambda[2], 3, 1);
            gamma = @.(exponential_lpdf(u[1], phi) + exponential_lpdf(v[1], lambda))
            for t in 2:N
                gamma .= ([
                    log_sum_exp(@broadcasted(
                        gamma + log(theta[:, k]) + exponential_lpdf(u[t], phi[k]) + exponential_lpdf(v[t], lambda[k])
                    ))
                    for k in 1:K
                ]);
            end
            target += log_sum_exp(gamma);
        end
        return target
    end
end

julia_implementation(::Val{:hmm_drive_1}; K, N, u, v, alpha, tau, rho, kwargs...) = begin 
    @stan begin 
        @parameters begin
            theta1::simplex[K]
            theta2::simplex[K]
            phi::ordered[K]
            lambda::ordered[K]
        end
        theta = hcat(theta1, theta2)'
        @model @views begin
            for k in 1:K
              target += dirichlet_lpdf(theta[k, :], alpha[k, :]);
            end
            target += normal_lpdf(phi[1], 0, 1);
            target += normal_lpdf(phi[2], 3, 1);
            target += normal_lpdf(lambda[1], 0, 1);
            target += normal_lpdf(lambda[2], 3, 1);
            gamma = @.(normal_lpdf(u[1], phi, tau) + normal_lpdf(v[1], lambda, rho))
            for t in 2:N
                gamma .= ([
                    log_sum_exp(@broadcasted(
                        gamma + log(theta[:, k]) + normal_lpdf(u[t], phi[k], tau) + normal_lpdf(v[t], lambda[k], rho)
                    ))
                    for k in 1:K
                ]);
            end
            target += log_sum_exp(gamma);
        end
        return target
    end
end

julia_implementation(::Val{:iohmm_reg}; T, K, M, y, u, kwargs...) = begin 
    @inline normalize(x) = x ./ sum(x)
    @stan begin 
        @parameters begin
            pi1::simplex[K]
            w::vector[K,M]
            b::vector[K,M]
            sigma::vector(lower=0)[K]
        end
        @model @views begin
            unA = hcat(pi1, w * u'[:, 2:end])'
            A = copy(unA)
            for t in 2:T
                A[t, :] .= softmax(unA[t, :])
            end
            logoblik = normal_lpdf.(y, u * b', sigma')
            logalpha = @.(log(pi1) + logoblik[1,:])
            for t in 2:T
                logalpha .= ([
                    log_sum_exp(@broadcasted(
                        logalpha + log(A[t,:]) + logoblik[t, j]
                    ))
                    for j in 1:K
                ]);
            end
            w ~ normal(0, 5);
            b ~ normal(0, 5);
            sigma ~ normal(0, 3);
            target += log_sum_exp(logalpha);
        end
        return target
    end
end

julia_implementation(::Val{:accel_gp}; N, Y, Kgp_1, Dgp_1, NBgp_1, Xgp_1, slambda_1, Kgp_sigma_1, Dgp_sigma_1, NBgp_sigma_1, Xgp_sigma_1, slambda_sigma_1, prior_only, kwargs...) = begin 
    @inline sqrt_spd_cov_exp_quad(slambda, sdgp, lscale) = begin 
        NB, D = size(slambda)
        Dls = length(lscale)
        if Dls == 1
            constant = (sdgp) * (sqrt(2 * pi) * lscale[1]) ^ (D/2)
            neg_half_lscale2 = -0.25 * square(lscale[1])
            @.(constant * exp(neg_half_lscale2 * dot_self($eachrow(slambda))))
        else
            error()
        end
    end
    @inline gpa(X, sdgp, lscale, zgp, slambda) = X * (sqrt_spd_cov_exp_quad(slambda, sdgp, lscale) .* zgp)
    @stan begin 
        @parameters begin
            Intercept::real
            sdgp_1::real(lower=0)
            lscale_1::real(lower=0)
            zgp_1::vector[NBgp_1]
            Intercept_sigma::real
            sdgp_sigma_1::real(lower=0)
            lscale_sigma_1::real(lower=0)
            zgp_sigma_1::vector[NBgp_sigma_1]
        end
        vsdgp_1 = fill(sdgp_1, 1)
        vlscale_1 = fill(lscale_1, (1, 1))
        vsdgp_sigma_1 = fill(sdgp_sigma_1, 1)
        vlscale_sigma_1 = fill(lscale_sigma_1, (1, 1))
        @model @views begin
            mu = Intercept .+ gpa(Xgp_1, vsdgp_1[1], vlscale_1[1,:], zgp_1, slambda_1);
            sigma = exp.(Intercept_sigma .+ gpa(Xgp_sigma_1, vsdgp_sigma_1[1], vlscale_sigma_1[1,:], zgp_sigma_1, slambda_sigma_1));
            target += student_t_lpdf(Intercept, 3, -13, 36);
            target += student_t_lpdf(vsdgp_1, 3, 0, 36)
                    #   - 1 * student_t_lccdf(0, 3, 0, 36);
            target += normal_lpdf(zgp_1, 0, 1);
            target += inv_gamma_lpdf(vlscale_1[1,:], 1.124909, 0.0177);
            target += student_t_lpdf(Intercept_sigma, 3, 0, 10);
            target += student_t_lpdf(vsdgp_sigma_1, 3, 0, 36)
                    #   - 1 * student_t_lccdf(0, 3, 0, 36);
            target += normal_lpdf(zgp_sigma_1, 0, 1);
            target += inv_gamma_lpdf(vlscale_sigma_1[1,:], 1.124909, 0.0177);
            if prior_only == 0
              target += normal_lpdf(Y, mu, sigma);
            end
        end
        return target
    end
end



julia_implementation(::Val{:hierarchical_gp};
        N,
        N_states,
        N_regions,
        N_years_obs,
        N_years,
        state_region_ind,
        state_ind,
        region_ind,
        year_ind,
        y,
        kwargs...) = begin 
    @stan begin 
        target = 0.
        years = 1:N_years
        counts = fill(2, 17)
        @parameters begin
            GP_region_std::matrix[N_years, N_regions]
            GP_state_std::matrix[N_years, N_states]
            year_std::vector[N_years_obs]
            state_std::vector[N_states]
            region_std::vector[N_regions]
            tot_var::real(lower=0)
            prop_var::simplex[17]
            mu::real
            length_GP_region_long::real(lower=0)
            length_GP_state_long::real(lower=0)
            length_GP_region_short::real(lower=0)
            length_GP_state_short::real(lower=0)
        end


        vars = 17 * prop_var * tot_var;
        sigma_year = sqrt(vars[1]);
        sigma_region = sqrt(vars[2]);
        sigma_state = @.(sqrt(vars[3:end]))
        
        sigma_GP_region_long = sqrt(vars[13]);
        sigma_GP_state_long = sqrt(vars[14]);
        sigma_GP_region_short = sqrt(vars[15]);
        sigma_GP_state_short = sqrt(vars[16]);
        sigma_error_state_2 = sqrt(vars[17]);
        
        region_re = sigma_region * region_std;
        year_re = sigma_year * year_std;
        state_re = sigma_state[state_region_ind] .* state_std;
        
        begin
            cov_region = gp_exp_quad_cov(years, sigma_GP_region_long,
                                        length_GP_region_long)
                        + gp_exp_quad_cov(years, sigma_GP_region_short,
                                        length_GP_region_short);
            cov_state = gp_exp_quad_cov(years, sigma_GP_state_long,
                                        length_GP_state_long)
                        + gp_exp_quad_cov(years, sigma_GP_state_short,
                                        length_GP_state_short);
            for year in 1 : N_years
                cov_region[year, year] = cov_region[year, year] + 1e-6;
                cov_state[year, year] = cov_state[year, year] + 1e-6;
            end
            
            L_cov_region = cholesky_decompose(cov_region);
            L_cov_state = cholesky_decompose(cov_state);
            GP_region = L_cov_region * GP_region_std;
            GP_state = L_cov_state * GP_state_std;
        end
        @model begin
            obs_mu = zeros(N)
            for n in 1 : N
                obs_mu[n] = mu + year_re[year_ind[n]] + state_re[state_ind[n]]
                            + region_re[region_ind[n]]
                            + GP_region[year_ind[n], region_ind[n]]
                            + GP_state[year_ind[n], state_ind[n]];
                end
                y ~ normal(obs_mu, sigma_error_state_2); 
                
                (GP_region_std) ~ normal(0, 1);
                (GP_state_std) ~ normal(0, 1);
                year_std ~ normal(0, 1);
                state_std ~ normal(0, 1);
                region_std ~ normal(0, 1);
                mu ~ normal(.5, .5);
                tot_var ~ gamma(3, 3);
                prop_var ~ dirichlet(counts);
                length_GP_region_long ~ weibull(30, 8);
                length_GP_state_long ~ weibull(30, 8);
                length_GP_region_short ~ weibull(30, 3);
                length_GP_state_short ~ weibull(30, 3);
        end
        return target
    end
end
julia_implementation(::Val{:kronecker_gp}; n1, n2, x1, y, kwargs...) = begin 
    kron_mvprod(A, B, V) = adjoint(A * adjoint(B * V))
    calculate_eigenvalues(A, B, sigma2) = A .* B' .+ sigma2
    xd  = -(x1 .- x1') .^ 2
    return @stan begin end
    @stan begin 
        @parameters begin
            var1::real(lower=0)
            bw1::real(lower=0)
            L::cholesky_factor_corr[n2]
            sigma1::real(lower=.00001)
        end
        Lambda = multiply_lower_tri_self_transpose(L)
        Sigma1 = Symmetric(var1 .* exp.(xd .* bw1) + .00001 * I);
        R1, Q1 = eigen(Sigma1);
        R2, Q2 = eigen(Lambda);
        eigenvalues = calculate_eigenvalues(R2, R1, sigma1);
        @model @views begin
            var1 ~ lognormal(0, 1);
            bw1 ~ cauchy(0, 2.5);
            sigma1 ~ lognormal(0, 1);
            L ~ lkj_corr_cholesky(2);
            target += -0.5 * sum(y .* kron_mvprod(Q1, Q2, kron_mvprod(transpose(Q1), transpose(Q2), y) ./ eigenvalues)) - 0.5 * sum(log(eigenvalues))
        end
        return target
    end
end


julia_implementation(::Val{:covid19imperial_v2}; M, P, N0, N, N2, cases, deaths, f, X, EpidemicStart, pop, SI, kwargs...) = begin 
    SI_rev = reverse(SI)
    f_rev = mapreduce(reverse, hcat, eachcol(f))'
    @stan begin 
        @parameters begin
            mu::real(lower=0)[M]
            alpha_hier::real(lower=0)[P]
            kappa::real(lower=0)
            y::real(lower=0)[M]
            phi::real(lower=0)
            tau::real(lower=0)
            ifr_noise::real(lower=0)[M]
        end
        @model @views begin
            prediction = rep_matrix(0., N2, M);
            E_deaths = rep_matrix(0., N2, M);
            Rt = rep_matrix(0., N2, M);
            Rt_adj = copy(Rt);
            cumm_sum = rep_matrix(0., N2, M);
            alpha = alpha_hier .- (log(1.05) / 6.)
            for m in 1:M
                prediction[1:N0, m] .= y[m]
                cumm_sum[2:N0, m] .= cumulative_sum(prediction[2:N0, m]);
                Rt[:, m] .= mu[m] * exp.(-X[m,:,:] * alpha);
                Rt_adj[1:N0, m] .= Rt[1:N0, m];
                for i in (N0+1):N2
                    convolution = dot_product(sub_col(prediction, 1, m, i - 1), stan_tail(SI_rev, i - 1))
                    cumm_sum[i, m] = cumm_sum[i - 1, m] + prediction[i - 1, m];
                    Rt_adj[i, m] = ((pop[m] - cumm_sum[i, m]) / pop[m]) * Rt[i, m];
                    prediction[i, m] = Rt_adj[i, m] * convolution;
                end
                E_deaths[1, m] = 1e-15 * prediction[1, m];
                for i in 2:N2
                    E_deaths[i, m] = ifr_noise[m] * dot_product(sub_col(prediction, 1, m, i - 1), stan_tail(f_rev[m, :], i - 1));
                end
            end
            tau ~ exponential(0.03);
            y ~ exponential(1 / tau)
            phi ~ normal(0, 5);
            kappa ~ normal(0, 0.5);
            mu ~ normal(3.28, kappa);
            alpha_hier ~ gamma(.1667, 1);
            ifr_noise ~ normal(1, 0.1);
            for m in 1:M
                deaths[EpidemicStart[m] : N[m], m] ~ neg_binomial_2(E_deaths[EpidemicStart[m] : N[m], m],
                                                            phi);
            end
        end
        return target
    end
end


julia_implementation(::Val{:covid19imperial_v3}; kwargs...) = julia_implementation(Val{:covid19imperial_v2}(); kwargs...)

julia_implementation(::Val{:gpcm_latent_reg_irt}; I, J, N, ii, jj, y, K, W, kwargs...) = begin 
    pcm(y, theta, beta) = begin 
        unsummed = append_row(rep_vector(0.0, 1), theta .- beta)
        probs = softmax(cumulative_sum(unsummed))
        categorical_lpmf(y + 1, probs)
    end
    obtain_adjustments(W) = begin 
        local M, K = size(W)
        adj = zeros((2, K))
        adj[1,1] = 0
        adj[2, 1] = 1
        for k in 2:K
            min_w = minimum(W[:,k])
            max_w = maximum(W[:,k])
            minmax_count = sum(w->w in (min_w, max_w), W[:, k])
            if minmax_count == M
                adj[1, k] = mean(W[:, k]);
                adj[2, k] = max_w - min_w;
            else
                adj[1, k] = mean(W[:, k]);
                adj[2, k] = sd(W[:, k]) * 2;
            end
        end
        return adj
    end
    m = fill(0, I)
    pos = fill(1, I)
    for n in 1:N
        if y[n] > m[ii[n]]
            m[ii[n]] = y[n]
        end
    end
    for i in 2:I
        pos[i] = m[i - 1] + pos[i - 1]
    end
    adj = obtain_adjustments(W);
    W_adj = @. ((W - adj[1:1,:])/adj[2:2,:])
    @stan begin 
        @parameters begin
            alpha::real(lower=0)[I]
            beta_free::vector[sum(m) - 1]
            theta::vector[J]
            lambda_adj::vector[K]
        end
        beta = vcat(beta_free, -sum(beta_free))
        @model @views begin
            alpha ~ lognormal(1, 1);
            target += normal_lpdf(beta, 0, 3);
            theta ~ normal(W_adj * lambda_adj, 1);
            lambda_adj ~ student_t(3, 0, 1);
            for n in 1:N
                target += pcm(y[n], theta[jj[n]] .* alpha[ii[n]], segment(beta, pos[ii[n]], m[ii[n]]));
            end
        end
    end
end
end

