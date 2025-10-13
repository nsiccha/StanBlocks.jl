import PosteriorDB
const pdb = PosteriorDB.database()
failed_posterior_names = Set(PosteriorDB.posterior_names(pdb))

slic_implementation(key; kwargs...) = nothing
slic_implementation(posterior::PosteriorDB.Posterior) = slic_implementation(
    Val(Symbol(PosteriorDB.name(PosteriorDB.model(posterior))));
    Dict([Symbol(k)=>v for (k, v) in pairs(PosteriorDB.load(PosteriorDB.dataset(posterior)))])...
)

begin
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
# slic_implementation(::Val{:wells_interaction_c_model}; kwargs...) = @slic kwargs begin
#     X = append_col(append_col(dist / 100, arsenic), dist / 100 .* arsenic)
#     alpha ~ flat() 
#     beta ~ flat(;n=3)
#     switched ~ bernoulli_logit_glm(X, alpha, beta)
# end
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
    # implement log mix thing
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
    rho ~ gamma(25, 4; lower=0.);
    alpha ~ normal(0, 2; lower=0.);
    sigma ~ normal(0, 1; lower=0.);
    cov = gp_exp_quad_cov(x, alpha, rho) + diag_matrix(rep_vector(sigma, N));
    y ~ multi_normal(rep_vector(0., N), cov)
end

slic_implementation(::Val{:mesquite}; kwargs...) = @slic kwargs begin 
    beta ~ flat(;n=7)
    sigma ~ flat(;lower=0.)
    weight ~ normal(
        beta[1] + beta[2] * to_vector(diam1) + beta[3] * to_vector(diam2) + beta[4] * to_vector(canopy_height) + beta[5] * to_vector(total_height) + beta[6] * to_vector(density) + beta[7] * to_vector(group), 
        sigma
    )
end

slic_implementation(::Val{:eight_schools_centered}; kwargs...) = @slic kwargs begin 
    mu ~ normal(0, 5)
    tau ~ cauchy(0, 5)
    theta ~ normal(mu, tau)
    y ~ normal(theta, sigma)
end

@testset "posteriordb" begin
    map(collect(failed_posterior_names)) do posterior_name
        post = slic_implementation(PosteriorDB.posterior(pdb, posterior_name))
        isnothing(post) && return 
        @info "Testing $posterior_name"
        @test compiles(post)
    end 
end
end