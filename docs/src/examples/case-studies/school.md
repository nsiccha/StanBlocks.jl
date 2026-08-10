# Disease transmission

Qualitatively reproduces [Léo Grinsztajn et al.'s disease transmissin case study](https://mc-stan.org/learn-stan/case-studies/boarding_school_case_study.html).

::: warning Historical sketch
These examples preserve the original placeholder `sir` ODE input and obsolete
`integrate_ode_rk45` call. Current StanBlocks cannot translate these unfinished
sketches, so no generated program is claimed or hidden here.
:::

```julia
using StanBlocks
data = (;
    sir=1.,
    y0=[1., 1.],
    t0=0.,
    ts=[0., 1.],
    x_r=0, x_i=0,
    cases=[0,1],
    n_days=2
)
@slic data begin 
    beta ~ normal(2, 1; lower=0.)
    gamma ~ normal(0.4, 0.5; lower=0.)
    R0 = beta / gamma
    recovery_time = 1 / gamma
    phi_inv ~ exponential(5)
    phi = 1. / phi_inv
    y = integrate_ode_rk45(sir, to_array_1d(y0), t0, to_array_1d(ts), {beta,gamma}, {}, {})
    cases ~ neg_binomial_2(y[:,2], phi)
end
```

```julia
@slic data begin 
    beta ~ normal(2, 1; lower=0.)
    gamma ~ normal(0.4, 0.5; lower=0.)
    R0 = beta / gamma
    recovery_time = 1 / gamma
    phi_inv ~ exponential(5)
    phi = 1. / phi_inv
    y = integrate_ode_rk45(sir, to_array_1d(y0), t0, to_array_1d(ts), {beta,gamma}, {}, {})
    incidence = (y[1:n_days-1, 1] - y[2:n_days, 1])
    cases[1:(n_days-1)] ~ neg_binomial_2(incidence, phi)
end
```

```julia
@slic data begin 
    beta ~ normal(2, 1; lower=0.)
    gamma ~ normal(0.4, 0.5; lower=0.)
    R0 = beta / gamma
    recovery_time = 1 / gamma
    phi_inv ~ exponential(5)
    phi = 1. / phi_inv
    y = integrate_ode_rk45(sir, to_array_1d(y0), t0, to_array_1d(ts), {beta,gamma}, {}, {})
    "It would again be nice to be able to reuse the previous model."
    p_reported ~ beta(1, 2)
    incidence = (y[1:n_days-1, 1] - y[2:n_days, 1]) .* p_reported
    cases[1:(n_days-1)] ~ neg_binomial_2(incidence, phi)
end
```

More coming at a later point.
