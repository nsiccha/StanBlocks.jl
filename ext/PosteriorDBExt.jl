module PosteriorDBExt

import PosteriorDB
import StanBlocks
import StanBlocks: slic_implementation, @slic, @deffun

# --- SLIC (Stan transpiler) implementations ---

slic_implementation(posterior::PosteriorDB.Posterior) = slic_implementation(
    Val(Symbol(PosteriorDB.name(PosteriorDB.model(posterior))));
    Dict([Symbol(k)=>v for (k, v) in pairs(PosteriorDB.load(PosteriorDB.dataset(posterior)))])...
)

@deffun begin
    garch11_lpdf(y::anything[T], mu::real, alpha0::real, alpha1::real, beta1::real, sigma1::real)::real = begin
        sigma_vec::real[T]
        sigma_vec[1] = sigma1
        for t in 2:T
            sigma_vec[t] = sqrt(alpha0 + alpha1 * square(y[t-1] - mu) + beta1 * square(sigma_vec[t-1]))
        end
        rv = 0.
        for t in 1:T
            rv += normal_lpdf(y[t], mu, sigma_vec[t])
        end
        rv
    end
    garch11_lpdfs(y::anything[T], mu::real, alpha0::real, alpha1::real, beta1::real, sigma1::real)::real =
        garch11_lpdf(y, mu, alpha0, alpha1, beta1, sigma1)
    garch11_rng(mu::real, alpha0::real, alpha1::real, beta1::real, sigma1::real)::real[1] = begin
        rv::real[1]; rv[1] = normal_rng(mu, sigma1); rv
    end
end

@deffun begin
    arma11_lpdf(y::anything[T], mu::real, phi::real, theta_ma::real, sigma::real)::real = begin
        nu_vec::real[T]
        err_vec::real[T]
        nu_vec[1] = mu + phi * mu
        err_vec[1] = y[1] - nu_vec[1]
        for t in 2:T
            nu_vec[t] = mu + phi * y[t-1] + theta_ma * err_vec[t-1]
            err_vec[t] = y[t] - nu_vec[t]
        end
        rv = 0.
        for t in 1:T
            rv += normal_lpdf(err_vec[t], 0., sigma)
        end
        rv
    end
    arma11_lpdfs(y::anything[T], mu::real, phi::real, theta_ma::real, sigma::real)::real =
        arma11_lpdf(y, mu, phi, theta_ma, sigma)
    arma11_rng(mu::real, phi::real, theta_ma::real, sigma::real)::real[1] = begin
        rv::real[1]; rv[1] = normal_rng(mu + phi * mu, sigma); rv
    end
end

@deffun begin
    gauss_mix2_lpdf(obs::anything[n], theta::real, mu1::real, mu2::real, sig1::real, sig2::real)::real = begin
        rv = 0.
        for i in 1:n
            rv += log_mix(theta, normal_lpdf(obs[i], mu1, sig1), normal_lpdf(obs[i], mu2, sig2))
        end
        rv
    end
    gauss_mix2_lpdfs(obs::anything[n], theta::real, mu1::real, mu2::real, sig1::real, sig2::real)::real =
        gauss_mix2_lpdf(obs, theta, mu1, mu2, sig1, sig2)
    gauss_mix2_rng(theta::real, mu1::real, mu2::real, sig1::real, sig2::real)::real[1] = begin
        rv::real[1]; rv[1] = normal_rng(mu1, sig1); rv
    end
end

@deffun begin
    car_normal_lpdf(phi::vector[N], node1::int[M], node2::int[M])::real =
        -0.5 * dot_self(phi[node1] - phi[node2])
    car_normal_lpdfs(phi::vector[N], node1::int[M], node2::int[M])::real =
        car_normal_lpdf(phi, node1, node2)
    car_normal_rng(node1::int[M], node2::int[M])::real[1] = begin
        rv::real[1]; rv[1] = 0.; rv
    end
end

slic_implementation(::Val{:earn_height}; kwargs...) = @slic kwargs begin
    beta ~ flat(;n=2)
    sigma ~ flat(;lower=0.)
    earn ~ normal(beta[1]+beta[2]*to_vector(height), sigma)
end
slic_implementation(::Val{:logearn_height}; kwargs...) = @slic kwargs begin
    log_earn = log(to_vector(earn))
    beta ~ flat(;n=2)
    sigma ~ flat(;lower=0.)
    log_earn ~ normal(beta[1]+beta[2]*to_vector(height), sigma)
end
slic_implementation(::Val{:logearn_height_male}; kwargs...) = @slic kwargs begin
    log_earn = log(to_vector(earn))
    beta ~ flat(;n=3)
    sigma ~ flat(;lower=0.)
    log_earn ~ normal(beta[1]+beta[2]*to_vector(height)+beta[3]*to_vector(male), sigma)
end
slic_implementation(::Val{:logearn_logheight_male}; kwargs...) = @slic kwargs begin
    log_earn = log(to_vector(earn))
    log_height = log(to_vector(height))
    beta ~ flat(;n=3)
    sigma ~ flat(;lower=0.)
    log_earn ~ normal(beta[1]+beta[2]*log_height+beta[3]*to_vector(male), sigma)
end
slic_implementation(::Val{:log10earn_height_male}; kwargs...) = @slic kwargs begin
    log10_earn = log10(to_vector(earn))
    beta ~ flat(;n=2)
    sigma ~ flat(;lower=0.)
    log10_earn ~ normal(beta[1]+beta[2]*to_vector(height), sigma)
end
slic_implementation(::Val{:logearn_interaction}; kwargs...) = @slic kwargs begin
    log_earn = log(to_vector(earn))
    inter = to_vector(height) .* to_vector(male)
    beta ~ flat(;n=4)
    sigma ~ flat(;lower=0.)
    log_earn ~ normal(beta[1]+beta[2]*to_vector(height)+beta[3]*to_vector(male)+beta[4]*inter, sigma)
end
slic_implementation(::Val{:wells_dist}; kwargs...) = @slic kwargs begin
    beta ~ flat(;n=2)
    switched ~ bernoulli_logit(beta[1]+beta[2]*dist)
end
slic_implementation(::Val{:wells_dist100_model}; kwargs...) = @slic kwargs begin
    X = to_matrix(dist / 100, N, 1)
    alpha ~ flat()
    beta ~ flat(;n=1)
    switched ~ bernoulli_logit_glm(X, alpha, beta)
end
slic_implementation(::Val{:wells_dist100ars_model}; kwargs...) = @slic kwargs begin
    X = append_col(dist / 100, arsenic)
    alpha ~ flat()
    beta ~ flat(;n=2)
    switched ~ bernoulli_logit_glm(X, alpha, beta)
end
slic_implementation(::Val{:wells_dae_model}; kwargs...) = @slic kwargs begin
    X = append_col(append_col(dist / 100, arsenic), to_vector(educ) / 4)
    alpha ~ flat()
    beta ~ flat(;n=3)
    switched ~ bernoulli_logit_glm(X, alpha, beta)
end
slic_implementation(::Val{:wells_interaction_model}; kwargs...) = @slic kwargs begin
    X = append_col(append_col(dist / 100, arsenic), dist / 100 .* arsenic)
    alpha ~ flat()
    beta ~ flat(;n=3)
    switched ~ bernoulli_logit_glm(X, alpha, beta)
end
slic_implementation(::Val{:sesame_one_pred_a}; kwargs...) = @slic kwargs begin
    beta ~ flat(;n=2)
    sigma ~ flat(;lower=0.)
    watched ~ normal(beta[1]+beta[2]*to_vector(encouraged), sigma)
end
slic_implementation(::Val{:Rate_1_model}; kwargs...) = @slic kwargs begin
    theta ~ beta(1, 1)
    k ~ binomial(n, theta)
end
slic_implementation(::Val{:Rate_2_model}; kwargs...) = @slic kwargs begin
    theta1 ~ beta(1, 1)
    theta2 ~ beta(1, 1)
    k1 ~ binomial(n1, theta1)
    k2 ~ binomial(n2, theta2)
end
slic_implementation(::Val{:Rate_3_model}; kwargs...) = @slic kwargs begin
    theta ~ beta(1, 1)
    k1 ~ binomial(n1, theta)
    k2 ~ binomial(n2, theta)
end
slic_implementation(::Val{:nes_logit_model}; kwargs...) = @slic kwargs begin
    X = to_matrix(income, N, 1)
    alpha ~ flat()
    beta ~ flat(;n=1)
    vote ~ bernoulli_logit_glm(X, alpha, beta);
end
slic_implementation(::Val{:kidscore_momiq}; kwargs...) = @slic kwargs begin
    beta ~ flat(;n=2)
    sigma ~ cauchy(0, 2.5; lower=0.)
    kid_score ~ normal(beta[1] + beta[2] * to_vector(mom_iq), sigma)
end
slic_implementation(::Val{:kidscore_momhs}; kwargs...) = @slic kwargs begin
    beta ~ flat(;n=2)
    sigma ~ cauchy(0, 2.5; lower=0.)
    kid_score ~ normal(beta[1] + beta[2] * to_vector(mom_hs), sigma)
end
slic_implementation(::Val{:kidscore_momhsiq}; kwargs...) = @slic kwargs begin
    beta ~ flat(;n=3)
    sigma ~ cauchy(0, 2.5; lower=0.)
    kid_score ~ normal(beta[1] + beta[2] * to_vector(mom_hs) + beta[3] * to_vector(mom_iq), sigma)
end
slic_implementation(::Val{:kidscore_interaction}; kwargs...) = @slic kwargs begin
    inter = to_vector(mom_hs) .* to_vector(mom_iq)
    beta ~ flat(;n=4)
    sigma ~ cauchy(0, 2.5; lower=0.)
    kid_score ~ normal(beta[1] + beta[2] * to_vector(mom_hs) + beta[3] * to_vector(mom_iq) + beta[4] * inter, sigma)
end
slic_implementation(::Val{:kidscore_mom_work}; kwargs...) = @slic kwargs begin
    work2 = jbroadcasted(==, mom_work, 2)
    work3 = jbroadcasted(==, mom_work, 3)
    work4 = jbroadcasted(==, mom_work, 4)
    beta ~ flat(;n=4)
    sigma ~ flat(; lower=0.)
    kid_score ~ normal(beta[1] + beta[2] * work2 + beta[3] * work3 + beta[4] * work4, sigma)
end
slic_implementation(::Val{:blr}; kwargs...) = @slic kwargs begin
    beta ~ normal(0, 10; n=D)
    sigma ~ normal(0, 10; lower=0.)
    y ~ normal(X * beta, sigma)
end
slic_implementation(::Val{:low_dim_gauss_mix_collapse}; kwargs...) = @slic kwargs begin
    mu ~ normal(0, 2; n=2)
    sigma ~ normal(0, 2; lower=0., n=2)
    theta ~ beta(5,5)
end
slic_implementation(::Val{:radon_county}; kwargs...) = @slic kwargs begin
    mu_a ~ normal(0., 1.)
    sigma_a ~ uniform(0, 100)
    a ~ normal(mu_a, sigma_a; n=J)
    sigma_y ~ uniform(0, 100)
    y ~ normal(a[county], sigma_y)
end
slic_implementation(::Val{:radon_pooled}; kwargs...) = @slic kwargs begin
    alpha ~ normal(0, 10)
    beta ~ normal(0, 10)
    sigma_y ~ normal(0, 1; lower=0)
    log_radon ~ normal(alpha + beta * to_vector(floor_measure), sigma_y)
end
slic_implementation(::Val{:logmesquite_logvolume}; kwargs...) = @slic kwargs begin
    log_weight = log(to_vector(weight))
    log_canopy_volume = log(to_vector(diam1) .* to_vector(diam2) .* to_vector(canopy_height))
    beta ~ flat(;n=2)
    sigma ~ flat(;lower=0.)
    log_weight ~ normal(beta[1]+beta[2]*log_canopy_volume, sigma)
end
slic_implementation(::Val{:gp_regr}; kwargs...) = @slic kwargs begin
    rho ~ gamma(25, 4; lower=0.)
    alpha ~ normal(0, 2; lower=0.)
    sigma ~ normal(0, 1; lower=0.)
    cov = gp_exp_quad_cov(x, alpha, rho) + diag_matrix(rep_vector(sigma, N))
    y ~ multi_normal(rep_vector(0., N), cov)
end
slic_implementation(::Val{:mesquite}; kwargs...) = @slic kwargs begin
    beta ~ flat(;n=7)
    sigma ~ flat(;lower=0.)
    weight ~ normal(beta[1] + beta[2] * to_vector(diam1) + beta[3] * to_vector(diam2) + beta[4] * to_vector(canopy_height) + beta[5] * to_vector(total_height) + beta[6] * to_vector(density) + beta[7] * to_vector(group), sigma)
end
slic_implementation(::Val{:eight_schools_centered}; kwargs...) = @slic kwargs begin
    mu ~ normal(0, 5)
    tau ~ cauchy(0, 5)
    theta ~ normal(mu, tau)
    y ~ normal(theta, to_vector(sigma))
end
slic_implementation(::Val{:eight_schools_noncentered}; kwargs...) = @slic kwargs begin
    theta_trans ~ normal(0, 1; n=J)
    mu ~ normal(0, 5)
    tau ~ cauchy(0, 5; lower=0.)
    theta = theta_trans * tau + mu
    y ~ normal(theta, to_vector(sigma))
end
slic_implementation(::Val{:kilpisjarvi}; kwargs...) = @slic kwargs begin
    alpha ~ normal(pmualpha, psalpha)
    beta ~ normal(pmubeta, psbeta)
    sigma ~ flat(; lower=0.)
    y ~ normal(alpha + beta * to_vector(x), sigma)
end
slic_implementation(::Val{:Rate_4_model}; kwargs...) = @slic kwargs begin
    theta ~ beta(1, 1)
    thetaprior ~ beta(1, 1)
    k ~ binomial(n, theta)
end
slic_implementation(::Val{:Rate_5_model}; kwargs...) = @slic kwargs begin
    theta ~ beta(1, 1)
    k1 ~ binomial(n1, theta)
    k2 ~ binomial(n2, theta)
end
slic_implementation(::Val{:logearn_interaction_z}; kwargs...) = @slic kwargs begin
    log_earn = log(to_vector(earn))
    z_height = (to_vector(height) - mean(to_vector(height))) / sd(to_vector(height))
    inter = z_height .* to_vector(male)
    beta ~ flat(; n=4)
    sigma ~ flat(; lower=0.)
    log_earn ~ normal(beta[1] + beta[2]*z_height + beta[3]*to_vector(male) + beta[4]*inter, sigma)
end
slic_implementation(::Val{:kidscore_interaction_c}; kwargs...) = @slic kwargs begin
    c_mom_hs = to_vector(mom_hs) - mean(to_vector(mom_hs))
    c_mom_iq = to_vector(mom_iq) - mean(to_vector(mom_iq))
    inter = c_mom_hs .* c_mom_iq
    beta ~ flat(; n=4)
    sigma ~ cauchy(0, 2.5; lower=0.)
    kid_score ~ normal(beta[1] + beta[2]*c_mom_hs + beta[3]*c_mom_iq + beta[4]*inter, sigma)
end
slic_implementation(::Val{:kidscore_interaction_c2}; kwargs...) = @slic kwargs begin
    c2_mom_hs = to_vector(mom_hs) - 0.5
    c2_mom_iq = to_vector(mom_iq) - 100
    inter = c2_mom_hs .* c2_mom_iq
    beta ~ flat(; n=4)
    sigma ~ flat(; lower=0.)
    kid_score ~ normal(beta[1] + beta[2]*c2_mom_hs + beta[3]*c2_mom_iq + beta[4]*inter, sigma)
end
slic_implementation(::Val{:kidscore_interaction_z}; kwargs...) = @slic kwargs begin
    z_mom_hs = (to_vector(mom_hs) - mean(to_vector(mom_hs))) / (2*sd(to_vector(mom_hs)))
    z_mom_iq = (to_vector(mom_iq) - mean(to_vector(mom_iq))) / (2*sd(to_vector(mom_iq)))
    inter = z_mom_hs .* z_mom_iq
    beta ~ flat(; n=4)
    sigma ~ flat(; lower=0.)
    kid_score ~ normal(beta[1] + beta[2]*z_mom_hs + beta[3]*z_mom_iq + beta[4]*inter, sigma)
end
slic_implementation(::Val{:radon_county_intercept}; kwargs...) = @slic kwargs begin
    alpha ~ normal(0, 10; n=J)
    beta ~ normal(0, 10)
    sigma_y ~ normal(0, 1; lower=0.)
    log_radon ~ normal(alpha[county_idx] + beta * to_vector(floor_measure), sigma_y)
end
slic_implementation(::Val{:radon_hierarchical_intercept_centered}; kwargs...) = @slic kwargs begin
    sigma_alpha ~ normal(0, 1; lower=0.)
    sigma_y ~ normal(0, 1; lower=0.)
    mu_alpha ~ normal(0, 10)
    beta ~ normal(0, 10; n=2)
    alpha ~ normal(mu_alpha, sigma_alpha; n=J)
    log_radon ~ normal(alpha[county_idx] + log_uppm * beta[1] + to_vector(floor_measure) * beta[2], sigma_y)
end
slic_implementation(::Val{:radon_hierarchical_intercept_noncentered}; kwargs...) = @slic kwargs begin
    sigma_alpha ~ normal(0, 1; lower=0.)
    sigma_y ~ normal(0, 1; lower=0.)
    mu_alpha ~ normal(0, 10)
    beta ~ normal(0, 10; n=2)
    alpha_raw ~ normal(0, 1; n=J)
    alpha = mu_alpha + sigma_alpha * alpha_raw
    log_radon ~ normal(alpha[county_idx] + log_uppm * beta[1] + to_vector(floor_measure) * beta[2], sigma_y)
end
slic_implementation(::Val{:radon_partially_pooled_centered}; kwargs...) = @slic kwargs begin
    sigma_y ~ normal(0, 1; lower=0.)
    sigma_alpha ~ normal(0, 1; lower=0.)
    mu_alpha ~ normal(0, 10)
    alpha ~ normal(mu_alpha, sigma_alpha; n=J)
    log_radon ~ normal(alpha[county_idx], sigma_y)
end
slic_implementation(::Val{:radon_partially_pooled_noncentered}; kwargs...) = @slic kwargs begin
    sigma_y ~ normal(0, 1; lower=0.)
    sigma_alpha ~ normal(0, 1; lower=0.)
    mu_alpha ~ normal(0, 10)
    alpha_raw ~ normal(0, 1; n=J)
    alpha = mu_alpha + sigma_alpha * alpha_raw
    log_radon ~ normal(alpha[county_idx], sigma_y)
end
slic_implementation(::Val{:radon_variable_intercept_centered}; kwargs...) = @slic kwargs begin
    sigma_y ~ normal(0, 1; lower=0.)
    sigma_alpha ~ normal(0, 1; lower=0.)
    mu_alpha ~ normal(0, 10)
    beta ~ normal(0, 10)
    alpha ~ normal(mu_alpha, sigma_alpha; n=J)
    log_radon ~ normal(alpha[county_idx] + to_vector(floor_measure) * beta, sigma_y)
end
slic_implementation(::Val{:radon_variable_intercept_noncentered}; kwargs...) = @slic kwargs begin
    sigma_y ~ normal(0, 1; lower=0.)
    sigma_alpha ~ normal(0, 1; lower=0.)
    mu_alpha ~ normal(0, 10)
    beta ~ normal(0, 10)
    alpha_raw ~ normal(0, 1; n=J)
    alpha = mu_alpha + sigma_alpha * alpha_raw
    log_radon ~ normal(alpha[county_idx] + to_vector(floor_measure) * beta, sigma_y)
end
slic_implementation(::Val{:radon_variable_slope_centered}; kwargs...) = @slic kwargs begin
    alpha ~ normal(0, 10)
    sigma_y ~ normal(0, 1; lower=0.)
    sigma_beta ~ normal(0, 1; lower=0.)
    mu_beta ~ normal(0, 10)
    beta ~ normal(mu_beta, sigma_beta; n=J)
    log_radon ~ normal(alpha + to_vector(floor_measure) .* beta[county_idx], sigma_y)
end
slic_implementation(::Val{:radon_variable_slope_noncentered}; kwargs...) = @slic kwargs begin
    alpha ~ normal(0, 10)
    sigma_y ~ normal(0, 1; lower=0.)
    sigma_beta ~ normal(0, 1; lower=0.)
    mu_beta ~ normal(0, 10)
    beta_raw ~ normal(0, 1; n=J)
    beta = mu_beta + sigma_beta * beta_raw
    log_radon ~ normal(alpha + to_vector(floor_measure) .* beta[county_idx], sigma_y)
end
slic_implementation(::Val{:radon_variable_intercept_slope_centered}; kwargs...) = @slic kwargs begin
    sigma_y ~ normal(0, 1; lower=0.)
    sigma_beta ~ normal(0, 1; lower=0.)
    sigma_alpha ~ normal(0, 1; lower=0.)
    mu_alpha ~ normal(0, 10)
    mu_beta ~ normal(0, 10)
    alpha ~ normal(mu_alpha, sigma_alpha; n=J)
    beta ~ normal(mu_beta, sigma_beta; n=J)
    log_radon ~ normal(alpha[county_idx] + to_vector(floor_measure) .* beta[county_idx], sigma_y)
end
slic_implementation(::Val{:radon_variable_intercept_slope_noncentered}; kwargs...) = @slic kwargs begin
    sigma_y ~ normal(0, 1; lower=0.)
    sigma_beta ~ normal(0, 1; lower=0.)
    sigma_alpha ~ normal(0, 1; lower=0.)
    mu_alpha ~ normal(0, 10)
    mu_beta ~ normal(0, 10)
    alpha_raw ~ normal(0, 1; n=J)
    beta_raw ~ normal(0, 1; n=J)
    alpha = mu_alpha + sigma_alpha * alpha_raw
    beta = mu_beta + sigma_beta * beta_raw
    log_radon ~ normal(alpha[county_idx] + to_vector(floor_measure) .* beta[county_idx], sigma_y)
end
slic_implementation(::Val{:logmesquite}; kwargs...) = @slic kwargs begin
    log_weight = log(to_vector(weight))
    log_diam1 = log(to_vector(diam1))
    log_diam2 = log(to_vector(diam2))
    log_canopy_height = log(to_vector(canopy_height))
    log_total_height = log(to_vector(total_height))
    log_density = log(to_vector(density))
    beta ~ flat(; n=7)
    sigma ~ flat(; lower=0.)
    log_weight ~ normal(beta[1] + beta[2]*log_diam1 + beta[3]*log_diam2 + beta[4]*log_canopy_height + beta[5]*log_total_height + beta[6]*log_density + beta[7]*to_vector(group), sigma)
end
slic_implementation(::Val{:logmesquite_logva}; kwargs...) = @slic kwargs begin
    log_weight = log(to_vector(weight))
    log_canopy_volume = log(to_vector(diam1) .* to_vector(diam2) .* to_vector(canopy_height))
    log_canopy_area = log(to_vector(diam1) .* to_vector(diam2))
    beta ~ flat(; n=4)
    sigma ~ flat(; lower=0.)
    log_weight ~ normal(beta[1] + beta[2]*log_canopy_volume + beta[3]*log_canopy_area + beta[4]*to_vector(group), sigma)
end
slic_implementation(::Val{:logmesquite_logvas}; kwargs...) = @slic kwargs begin
    log_weight = log(to_vector(weight))
    log_canopy_volume = log(to_vector(diam1) .* to_vector(diam2) .* to_vector(canopy_height))
    log_canopy_area = log(to_vector(diam1) .* to_vector(diam2))
    log_canopy_shape = log(to_vector(diam1) ./ to_vector(diam2))
    log_total_height = log(to_vector(total_height))
    log_density = log(to_vector(density))
    beta ~ flat(; n=7)
    sigma ~ flat(; lower=0.)
    log_weight ~ normal(beta[1] + beta[2]*log_canopy_volume + beta[3]*log_canopy_area + beta[4]*log_canopy_shape + beta[5]*log_total_height + beta[6]*log_density + beta[7]*to_vector(group), sigma)
end
slic_implementation(::Val{:logmesquite_logvash}; kwargs...) = @slic kwargs begin
    log_weight = log(to_vector(weight))
    log_canopy_volume = log(to_vector(diam1) .* to_vector(diam2) .* to_vector(canopy_height))
    log_canopy_area = log(to_vector(diam1) .* to_vector(diam2))
    log_canopy_shape = log(to_vector(diam1) ./ to_vector(diam2))
    log_total_height = log(to_vector(total_height))
    beta ~ flat(; n=6)
    sigma ~ flat(; lower=0.)
    log_weight ~ normal(beta[1] + beta[2]*log_canopy_volume + beta[3]*log_canopy_area + beta[4]*log_canopy_shape + beta[5]*log_total_height + beta[6]*to_vector(group), sigma)
end
slic_implementation(::Val{:wells_interaction_c_model}; kwargs...) = @slic kwargs begin
    c_dist100 = (dist - mean(dist)) / 100.0
    c_arsenic = arsenic - mean(arsenic)
    inter = c_dist100 .* c_arsenic
    X = append_col(append_col(c_dist100, c_arsenic), inter)
    alpha ~ flat()
    beta ~ flat(; n=3)
    switched ~ bernoulli_logit_glm(X, alpha, beta)
end
slic_implementation(::Val{:wells_dae_c_model}; kwargs...) = @slic kwargs begin
    c_dist100 = (dist - mean(dist)) / 100.0
    c_arsenic = arsenic - mean(arsenic)
    da_inter = c_dist100 .* c_arsenic
    educ4 = to_vector(educ) / 4.0
    X = append_col(append_col(append_col(c_dist100, c_arsenic), da_inter), educ4)
    alpha ~ flat()
    beta ~ flat(; n=4)
    switched ~ bernoulli_logit_glm(X, alpha, beta)
end
slic_implementation(::Val{:wells_daae_c_model}; kwargs...) = @slic kwargs begin
    c_dist100 = (dist - mean(dist)) / 100.0
    c_arsenic = arsenic - mean(arsenic)
    da_inter = c_dist100 .* c_arsenic
    educ4 = to_vector(educ) / 4.0
    X = append_col(append_col(append_col(append_col(c_dist100, c_arsenic), da_inter), to_vector(assoc)), educ4)
    alpha ~ flat()
    beta ~ flat(; n=5)
    switched ~ bernoulli_logit_glm(X, alpha, beta)
end
slic_implementation(::Val{:wells_dae_inter_model}; kwargs...) = @slic kwargs begin
    c_dist100 = (dist - mean(dist)) / 100.0
    c_arsenic = arsenic - mean(arsenic)
    c_educ4 = (to_vector(educ) - mean(to_vector(educ))) / 4.0
    da_inter = c_dist100 .* c_arsenic
    de_inter = c_dist100 .* c_educ4
    ae_inter = c_arsenic .* c_educ4
    X = append_col(append_col(append_col(append_col(append_col(c_dist100, c_arsenic), c_educ4), da_inter), de_inter), ae_inter)
    alpha ~ flat()
    beta ~ flat(; n=6)
    switched ~ bernoulli_logit_glm(X, alpha, beta)
end
slic_implementation(::Val{:log10earn_height}; kwargs...) = @slic kwargs begin
    log10_earn = log10(to_vector(earn))
    beta ~ flat(;n=2)
    sigma ~ flat(;lower=0.)
    log10_earn ~ normal(beta[1]+beta[2]*to_vector(height), sigma)
end
slic_implementation(::Val{:nes}; kwargs...) = @slic kwargs begin
    age30_44 = jbroadcasted(==, age_discrete, 2)
    age45_64 = jbroadcasted(==, age_discrete, 3)
    age65up = jbroadcasted(==, age_discrete, 4)
    beta ~ flat(;n=9)
    sigma ~ flat(;lower=0.)
    to_vector(partyid7) ~ normal(beta[1] + beta[2] * to_vector(real_ideo) + beta[3] * to_vector(race_adj) + beta[4] * age30_44 + beta[5] * age45_64 + beta[6] * age65up + beta[7] * to_vector(educ1) + beta[8] * to_vector(gender) + beta[9] * to_vector(income), sigma)
end
slic_implementation(::Val{:pilots}; kwargs...) = @slic kwargs begin
    mu_a ~ normal(0, 1)
    sigma_a ~ flat(;lower=0., upper=100.)
    a ~ normal(10 * mu_a, sigma_a; n=n_groups)
    mu_b ~ normal(0, 1)
    sigma_b ~ flat(;lower=0., upper=100.)
    b ~ normal(10 * mu_b, sigma_b; n=n_scenarios)
    sigma_y ~ flat(;lower=0., upper=100.)
    y ~ normal(a[group_id] + b[scenario_id], sigma_y)
end
slic_implementation(::Val{:surgical_model}; kwargs...) = @slic kwargs begin
    mu ~ normal(0, 1000.)
    sigmasq ~ inv_gamma(0.001, 0.001; lower=0.)
    sigma = sqrt(sigmasq)
    b ~ normal(mu, sigma; n=N)
    r ~ binomial_logit(n, b)
end
slic_implementation(::Val{:seeds_model}; kwargs...) = @slic kwargs begin
    vx1 = to_vector(x1)
    vx2 = to_vector(x2)
    x1x2 = vx1 .* vx2
    alpha0 ~ normal(0, 1000.)
    alpha1 ~ normal(0, 1000.)
    alpha2 ~ normal(0, 1000.)
    alpha12 ~ normal(0, 1000.)
    tau ~ gamma(0.001, 0.001; lower=0.)
    sigma = 1.0 / sqrt(tau)
    b ~ normal(0, sigma; n=I)
    n ~ binomial_logit(to_vector(N), alpha0 + alpha1 * vx1 + alpha2 * vx2 + alpha12 * x1x2 + b)
end
slic_implementation(::Val{:seeds_stanified_model}; kwargs...) = @slic kwargs begin
    vx1 = to_vector(x1)
    vx2 = to_vector(x2)
    x1x2 = vx1 .* vx2
    alpha0 ~ normal(0, 1.)
    alpha1 ~ normal(0, 1.)
    alpha2 ~ normal(0, 1.)
    alpha12 ~ normal(0, 1.)
    sigma ~ cauchy(0, 1; lower=0.)
    b ~ normal(0, sigma; n=I)
    n ~ binomial_logit(to_vector(N), alpha0 + alpha1 * vx1 + alpha2 * vx2 + alpha12 * x1x2 + b)
end
slic_implementation(::Val{:seeds_centered_model}; kwargs...) = @slic kwargs begin
    vx1 = to_vector(x1)
    vx2 = to_vector(x2)
    x1x2 = vx1 .* vx2
    alpha0 ~ normal(0, 1.)
    alpha1 ~ normal(0, 1.)
    alpha2 ~ normal(0, 1.)
    alpha12 ~ normal(0, 1.)
    sigma ~ cauchy(0, 1; lower=0.)
    c ~ normal(0, sigma; n=I)
    b = c - mean(c)
    n ~ binomial_logit(to_vector(N), alpha0 + alpha1 * vx1 + alpha2 * vx2 + alpha12 * x1x2 + b)
end
slic_implementation(::Val{:GLM_Poisson_model}; kwargs...) = @slic kwargs begin
    year_squared = year .* year
    year_cubed = year_squared .* year
    alpha ~ flat(;lower=-20., upper=20.)
    beta1 ~ flat(;lower=-10., upper=10.)
    beta2 ~ flat(;lower=-10., upper=10.)
    beta3 ~ flat(;lower=-10., upper=10.)
    log_lambda = alpha + beta1 * year + beta2 * year_squared + beta3 * year_cubed
    C ~ poisson_log(log_lambda)
end
slic_implementation(::Val{:GLM_Binomial_model}; kwargs...) = @slic kwargs begin
    year_squared = year .* year
    alpha ~ normal(0, 100)
    beta1 ~ normal(0, 100)
    beta2 ~ normal(0, 100)
    logit_p = alpha + beta1 * year + beta2 * year_squared
    C ~ binomial_logit(N, logit_p)
end
slic_implementation(::Val{:GLMM_Poisson_model}; kwargs...) = @slic kwargs begin
    year_squared = year .* year
    year_cubed = year .* year .* year
    alpha ~ uniform(-20, 20)
    beta1 ~ uniform(-10, 10)
    beta2 ~ uniform(-10, 10; upper=20)
    beta3 ~ uniform(-10, 10)
    sigma ~ uniform(0, 5)
    eps ~ normal(0, sigma; n=n)
    log_lambda = alpha + beta1 * year + beta2 * year_squared + beta3 * year_cubed + eps
    C ~ poisson_log(log_lambda)
end
slic_implementation(::Val{:election88_full}; kwargs...) = @slic kwargs begin
    sigma_a ~ flat(;lower=0., upper=100.)
    sigma_b ~ flat(;lower=0., upper=100.)
    sigma_c ~ flat(;lower=0., upper=100.)
    sigma_d ~ flat(;lower=0., upper=100.)
    sigma_e ~ flat(;lower=0., upper=100.)
    a ~ normal(0, sigma_a; n=n_age)
    b ~ normal(0, sigma_b; n=n_edu)
    c ~ normal(0, sigma_c; n=n_age_edu)
    d ~ normal(0, sigma_d; n=n_state)
    e ~ normal(0, sigma_e; n=n_region_full)
    beta ~ normal(0, 100; n=5)
    vblack = to_vector(black)
    vfemale = to_vector(female)
    female_black = vfemale .* vblack
    y_hat = beta[1] + beta[2] * vblack + beta[3] * vfemale + beta[5] * female_black + beta[4] * v_prev_full + to_vector(a[age]) + to_vector(b[edu]) + to_vector(c[age_edu]) + to_vector(d[state]) + to_vector(e[region_full])
    y ~ bernoulli_logit(y_hat)
end
slic_implementation(::Val{:logistic_regression_rhs}; kwargs...) = @slic kwargs begin
    beta0 ~ normal(0, scale_icept)
    z ~ normal(0, 1; n=d)
    tau ~ student_t(nu_global, 0, scale_global * 2; lower=0.)
    lambda ~ student_t(nu_local, 0, 1; lower=0., n=d)
    caux ~ inv_gamma(0.5 * slab_df, 0.5 * slab_df; lower=0.)
    c = slab_scale * sqrt(caux)
    lambda_tilde = sqrt(c * c * abs2(lambda) ./ (c * c + tau * tau * abs2(lambda)))
    beta = z .* lambda_tilde * tau
    y ~ bernoulli_logit_glm(x, beta0, beta)
end
slic_implementation(::Val{:gp_pois_regr}; kwargs...) = @slic kwargs begin
    rho ~ gamma(25, 4; lower=0.)
    alpha ~ normal(0, 2; lower=0.)
    f_tilde ~ normal(0, 1; n=N)
    cov = gp_exp_quad_cov(x, alpha, rho) + diag_matrix(rep_vector(1e-10, N))
    L_cov = cholesky_decompose(cov)
    f = L_cov * f_tilde
    k ~ poisson_log(f)
end
slic_implementation(::Val{:bym2_offset_only}; N, node1, node2, y, E, scaling_factor, kwargs...) =
    @slic (;N, node1, node2, y, E, scaling_factor) begin
        beta0  ~ normal(0, 1)
        sigma  ~ normal(0, 1; lower=0.)
        rho    ~ beta(0.5, 0.5)
        theta  ~ normal(0, 1; n=N)
        phi    ~ car_normal(node1, node2; n=N)
        log_E  = log(to_vector(E))
        convolved_re = sqrt(1 - rho) * theta + sqrt(rho / scaling_factor) * phi
        y ~ poisson_log(log_E + beta0 + convolved_re * sigma)
    end
slic_implementation(::Val{:garch11}; T, y, sigma1, kwargs...) = @slic (;T, y, sigma1) begin
    mu     ~ normal(0, 10)
    alpha0 ~ normal(0, 1; lower=0.)
    alpha1 ~ uniform(0, 1)
    beta1  ~ uniform(0, 1)
    y ~ garch11(mu, alpha0, alpha1, beta1, sigma1)
end
slic_implementation(::Val{:arma11}; T, y, kwargs...) = @slic (;T, y) begin
    mu       ~ normal(0, 10)
    phi      ~ normal(0, 2)
    theta_ma ~ normal(0, 2)
    sigma    ~ cauchy(0, 2.5; lower=0.)
    y ~ arma11(mu, phi, theta_ma, sigma)
end
slic_implementation(::Val{:low_dim_gauss_mix}; N, y, kwargs...) = @slic (;N, y) begin
    mu    ~ normal(0, 2; n=2)
    sigma ~ normal(0, 2; lower=0., n=2)
    theta ~ beta(5, 5)
    y ~ gauss_mix2(theta, mu[1], mu[2], sigma[1], sigma[2])
end
slic_implementation(::Val{:normal_mixture}; N, y, kwargs...) = @slic (;N, y) begin
    theta ~ uniform(0, 1)
    mu    ~ normal(0, 10; n=2)
    y ~ gauss_mix2(theta, mu[1], mu[2], 1., 1.)
end
slic_implementation(::Val{:dugongs_model}; N, x, Y, kwargs...) = @slic (;N, x, Y) begin
    alpha  ~ normal(0.0, 1000)
    beta   ~ normal(0.0, 1000)
    lambda ~ uniform(.5, 1)
    tau    ~ gamma(.0001, .0001; lower=0.)
    sigma  = 1.0 / sqrt(tau)
    Y ~ normal(alpha - beta * exp(log(lambda) * to_vector(x)), sigma)
end
slic_implementation(::Val{:diamonds}; N, Y, K, X, kwargs...) = begin
    Xmat = reduce(hcat, X)'
    Kc = K - 1
    means_X = vec(sum(Xmat[:, 2:end], dims=1) ./ N)
    Xc = Xmat[:, 2:end] .- means_X'
    @slic (;N, Y, Kc, Xc) begin
        b         ~ normal(0, 1; n=Kc)
        Intercept ~ student_t(3, 8, 10)
        sigma     ~ student_t(3, 0, 10; lower=0.)
        Y ~ normal(Xc * b + Intercept, sigma)
    end
end
slic_implementation(::Val{:accel_splines}; N, Y, Ks, Xs, knots_1, Zs_1_1,
        Ks_sigma, Xs_sigma, knots_sigma_1, Zs_sigma_1_1, kwargs...) = begin
    Xs_m        = reshape(reduce(vcat, Xs), N, Ks)
    Zs_1_1_m    = reshape(reduce(vcat, Zs_1_1), N, knots_1)
    Xs_sigma_m  = reshape(reduce(vcat, Xs_sigma), N, Ks_sigma)
    Zs_s_m      = reshape(reduce(vcat, Zs_sigma_1_1), N, knots_sigma_1)
    @slic (;N, Y, Ks, Xs=Xs_m, knots_1, Zs_1_1=Zs_1_1_m,
            Ks_sigma, Xs_sigma=Xs_sigma_m, knots_sigma_1, Zs_sigma_1_1=Zs_s_m) begin
        Intercept       ~ student_t(3, -13, 36)
        bs              ~ normal(0, 1; n=Ks)
        zs_1_1          ~ normal(0, 1; n=knots_1)
        sds_1_1         ~ student_t(3, 0, 36; lower=0.)
        Intercept_sigma ~ student_t(3, 0, 10)
        bs_sigma        ~ normal(0, 1; n=Ks_sigma)
        zs_sigma_1_1    ~ normal(0, 1; n=knots_sigma_1)
        sds_sigma_1_1   ~ student_t(3, 0, 36; lower=0.)
        s_1_1       = sds_1_1 * zs_1_1
        s_sigma_1_1 = sds_sigma_1_1 * zs_sigma_1_1
        mu          = Intercept + Xs * bs + Zs_1_1 * s_1_1
        sigma_log   = Intercept_sigma + Xs_sigma * bs_sigma + Zs_sigma_1_1 * s_sigma_1_1
        Y ~ normal(mu, exp(sigma_log))
    end
end

end # module PosteriorDBExt
