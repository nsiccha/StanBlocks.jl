# Monster model notebook

::: warning Incomplete historical notebook
This source is preserved verbatim from the legacy notebook. It contains
unresolved placeholders and an unfinished model invocation, so there is no
honest build-generated Stan program to show for it. It is retained as design
and data-preparation history, not presented as a runnable example.
:::

```julia
using StanBlocks, JSON, StanLogDensityProblems, WarmupHMC, Term
@deffun begin 
    dydt_exposure(t, concentration_out, FVP, FFPF, CFFPF, VMI, KMI) = FvP .* (
        dot_product(FFPF, concentration_out) + CFFPF - concentration_out
    ) + [
        0, 0, 0, VMI * concentration_out[4] / (KMI + concentration_out[4])
    ]'
    dydt_nonexposure(t, concentration_out, FVP, FFPF, CFFPF, VMI, KMI) = FVP .* (
        dot_product(FFPF, concentration_out) - concentration_out
    ) + [
        0,0,0,VMI * concentration_out[4] / (KMI + concentration_out[4])
    ]'
    lambert_w0_exp(earg) = if earg > 700
      y0 = 693.45830887902549833674969928122
      dydx = 0.9985600287487175852848305439518009385629359235843258625030446976
      y0 + (earg - x0) * dydx
    else
        if earg < -40
            y0 = 4.2483542552915889772807209044045064097730030864961656941893e-18
            y0 * exp(earg - x0)
        else
            lambert_w0(exp(earg))
        end
    end
    exact_michaelis_menten_solution(dt, C, V, K) = if K == 0
        C - dt * V
    else
        K * lambert_w0_exp((dt*V+C)/K+log(C/K))
    end
    reciprocal_lpdf(y) = -sum(log(y))
end
monster_model = @slic begin 
    pop_loc ~ std_normal(; n=no_latent_params)
    pop_squared_scale ~ scaled_inv_chi_square(pop_squared_scale_nu, pop_squared_scale_mu; n=no_latent_params)
    pop_scale = sqrt(pop_squared_scale)
    unit_params ~ std_normal(; n=no_latent_params*no_persons)
    params = rep_matrix(to_vector(pop_loc), no_persons) + diag_pre_multiply(pop_squared_scale, to_matrix(unit_params, no_latent_params, no_persons))
    noise ~ std_normal(;lower=0., n=1)
    obs ~ normal(0 * to_vector(params), noise[1])
end
monster_posterior = monster_model(; 
    no_latent_params=15,
    no_persons=3,
    pop_squared_scale_nu=2.,
    [1.6, 0.48, 0.2, 0.07, 0.25, 0.28, 0.56, 0.033, 12.0, 4.8, 1.6, 125.0, 4.8, 0.042, 16.0]
    [1.3, 1.2, 1.2, 1.2, 1.1, 1.2, 1.2, 1.1, 1.5, 1.5, 1.5, 1.5, 1.5, 10.0, 10.0]
    pop_squared_scale_mu=log.([1.3, 1.2, 1.2, 1.2, 1.1, 1.2, 1.2, 1.1, 1.3, 1.3, 1.3, 1.3, 1.3, 2.0, 1.5]),
    measured_params=stack([[62.0, 0.114, 7.6], [71.0, 0.134, 11.6], [71.0, 0.134, 10.0], [74.0, 0.14, 11.3], [61.0, 0.09, 12.3], [61.0, 0.208, 8.8]])',
    obs=ones(45)
)
WarmupHMC.adaptive_warmup_mcmc(Xoshiro(0), stan_instantiate(monster_posterior; nan_on_error=false); progress=Term.ProgressBar)
```

```julia
using DataFrames
exposures = [0.488, 0.976]
measured_params = stack([[62.0, 0.114, 7.6], [71.0, 0.134, 11.6], [71.0, 0.134, 10.0], [74.0, 0.14, 11.3], [61.0, 0.09, 12.3], [61.0, 0.208, 8.8]])'
tm = [[[[240.0, 2.8, 0.34], [245.0, NaN, 0.099], [270.0, NaN, 0.044], [360.0, 0.92, 0.033], [1320.0, 0.17, 0.0063], [2760.0, 0.082, 0.0034500000000000004], [4260.0, 0.055, 0.0021000000000000003], [8580.0, 0.018, 0.00076], [8580.0, 0.018, 0.00076]], [[240.0, 5.7, 0.632], [245.0, NaN, 0.219], [270.0, NaN, 0.116], [360.0, 1.76, 0.058], [1320.0, 0.36, 0.0129], [2760.0, 0.147, 0.005200000000000001], [4260.0, 0.106, 0.0035], [8580.0, 0.072, 0.0012], [8580.0, 0.072, 0.0012]]], [[[240.0, 3.0, 0.34500000000000003], [245.0, NaN, 0.101], [270.0, NaN, 0.076], [360.0, 1.2, 0.049], [1320.0, 0.15, 0.0063], [2760.0, 0.066, 0.0027], [4260.0, 0.051, 0.0017], [8580.0, 0.02, 0.0007800000000000001], [8580.0, 0.02, 0.0007800000000000001]], [[240.0, 8.8, 0.6990000000000001], [245.0, NaN, 0.241], [270.0, NaN, 0.12], [360.0, 2.9, 0.075], [1320.0, 0.36, 0.0114], [2760.0, 0.19, 0.0067], [4260.0, 0.12, 0.0040999999999999995], [8580.0, 0.036, 0.0013000000000000002], [8580.0, 0.036, 0.0013000000000000002]]], [[[240.0, 3.2, 0.294], [245.0, NaN, 0.11800000000000001], [270.0, NaN, 0.083], [360.0, 1.16, 0.048], [1320.0, 0.115, 0.005200000000000001], [2760.0, 0.048, 0.0025499999999999997], [4260.0, 0.035, 0.0013000000000000002], [8580.0, 0.015, 0.0006500000000000001], [8580.0, 0.015, 0.0006500000000000001]], [[240.0, 6.4, 0.5690000000000001], [245.0, NaN, 0.178], [270.0, NaN, 0.10300000000000001], [360.0, 2.36, 0.064], [1320.0, 0.26, 0.011], [2760.0, 0.177, 0.0054], [4260.0, 0.085, 0.003], [5700.0, 0.085, 0.002], [10020.0, 0.024, 0.00082]]], [[[240.0, 3.1, 0.329], [245.0, NaN, 0.117], [270.0, NaN, 0.065], [360.0, 1.3, 0.03], [1320.0, 0.185, 0.007200000000000001], [2760.0, 0.068, 0.0026000000000000003], [4260.0, 0.04, 0.0016200000000000001], [5700.0, 0.037, 0.00108], [10020.0, 0.0065, 0.00024]], [[240.0, 6.0, 0.646], [245.0, NaN, 0.249], [270.0, NaN, 0.126], [360.0, 2.48, 0.101], [1320.0, 0.36, 0.011], [2760.0, 0.165, 0.0054], [4260.0, 0.071, 0.0027], [5700.0, 0.064, 0.0021000000000000003], [10020.0, 0.018, 0.0006]]], [[[240.0, 2.8, 0.36], [245.0, NaN, 0.093], [270.0, NaN, 0.044], [360.0, 1.12, 0.021], [1320.0, 0.14, 0.0068], [2760.0, 0.068, 0.0024], [4260.0, 0.047, 0.0014], [5700.0, 0.036, 0.00096], [10020.0, 0.014, 0.00038]], [[240.0, 6.4, 0.686], [245.0, NaN, 0.108], [270.0, NaN, 0.098], [360.0, 2.96, 0.0655], [1320.0, 0.35, 0.0112], [2760.0, 0.19, 0.006200000000000001], [4260.0, 0.105, 0.0034], [8580.0, 0.05, 0.0014], [8580.0, 0.05, 0.0014]]], [[[240.0, 2.6, 0.292], [245.0, NaN, 0.064], [270.0, NaN, 0.05], [360.0, 0.96, 0.023], [1320.0, 0.105, 0.00405], [2760.0, 0.07, 0.0025], [4260.0, 0.051, 0.002], [5700.0, 0.05, 0.00145], [10020.0, 0.025, 0.0009000000000000001]], [[240.0, 6.0, 0.628], [245.0, NaN, 0.193], [270.0, NaN, 0.1], [360.0, 1.76, 0.056], [1320.0, 0.245, 0.009300000000000001], [2760.0, 0.16, 0.006], [4260.0, 0.12, 0.0050999999999999995], [5700.0, 0.098, 0.0032], [10020.0, 0.052, 0.0015]]]] |> stack |> stack |> stack

times = tm[[1,1], :, :, :]
obs = tm[2:3, :, :, :]
assay = fill(0, size(obs))
experiment = fill(0, size(obs))
person = fill(0, size(obs))
assay[1, :, :, :] .= 1
assay[2, :, :, :] .= 2
experiment[:, :, 1, :] .= 1
experiment[:, :, 2, :] .= 2
for i in 1:6
    person[:, :, :, i] .= i
end
df = DataFrame(;map(vec, (;person, experiment, time=times, assay, obs))...)
sort!(df, [:person, :experiment, :time, :assay])
filter!(row->isfinite(row.obs), df)
```
