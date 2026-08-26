# Wastewater renewal model

This example ports the core of the CDC
[`ww-inference-model`](https://github.com/CDCgov/ww-inference-model)
(`inst/stan/wwinference.stan`) — a joint model that infers latent infections
from **two** noisy signals at once: hospital admissions and pathogen
concentration in wastewater. It is a *renewal* model: today's infections are
generated from recent infections and a time-varying reproduction number, rather
than from a mechanistic compartmental system.

The published model runs several coupled sub-epidemics and aggregates them to
the state level. This page ports it in two steps. First a **single-population
core**, which preserves every ingredient that makes the model interesting — the
renewal recurrence with infection feedback, a time-varying `Rt`, two delay
convolutions, a day-of-week reporting effect, and censored wastewater
observations below the limit of detection — and keeps the generated program
small enough to read. Then the **full multi-subpopulation model**, which adds
the coupled sub-epidemics and their aggregation.

The small arrays below are build fixtures, not real surveillance data. As on the
other case-study pages, the build evaluates the exact displayed Julia source and
places the complete generated Stan program beside it.

## The renewal recurrence lives in a `@deffun`

The heart of the model is a scan: infections on day `t` depend on infections on
every preceding day, weighted by the generation-interval pmf, and scaled by the
day's reproduction number. A scan is a sequential recurrence — each step reads
the previous steps' results — so it cannot be written in a `@slic` body (which is
straight-line and control-flow-free) or as a `plate` (whose cells are
independent). It belongs in a `@deffun` Stan function, where ordinary `for`
loops and indexed assignment are allowed:

```julia
ww_renewal(...) = begin
    infections::vector[nt]
    for t in 1:uot                     # seeding: exponential growth
        infections[t] = exp(log_i0 + growth * (t - 1))
    end
    for t in (uot + 1):nt              # renewal recurrence
        infectiousness = ww_conv_at(infections, gen_int, t)
        fbk = ww_conv_at(infections, fb_pmf, t)
        rt_eff = exp(log_rt[t] - feedback * fbk)
        infections[t] = rt_eff * infectiousness
    end
    infections
end
```

The `infection_feedback` term reduces the effective `Rt` when recent
infectiousness is high — a nonlinear dependence on the scan's own state, which is
exactly why the recurrence cannot be vectorized away. The CDC source returns a
`tuple(infections, Rt)` from this function; the port returns only the infection
vector and recomputes the effective `Rt` where it is needed, sidestepping
tuple-valued returns.

## `Rt` is a weekly random walk, expanded to daily

`log(Rt)` follows a first-order Gaussian random walk at weekly resolution (also a
`@deffun` scan, `ww_rw`). A data index vector `week_of_day` maps each day to its
week, so the daily series is a single fancy-indexing expression
`log_rt_weekly[week_of_day]` — no hand-written expansion loop.

## Two delay convolutions

Both observation streams are backward convolutions of the latent infection curve
with a delay kernel, computed by the reusable `ww_conv` helper:

- **admissions** convolve infections with an infection-to-hospitalization delay
  pmf, scaled by an infection-hospitalization ratio;
- **wastewater** convolves infections with a shedding kernel and maps the result
  to a log genome-concentration.

## Day-of-week reporting

Admissions are under- or over-reported by day of week. `hosp_wday_effect` is a
`simplex[7]` with a `dirichlet` prior; multiplying by `7` makes the effect
mean-preserving, and `hosp_wday_effect[day_of_week]` applies it by fancy index.
The admissions likelihood is `neg_binomial_2`.

## The spotlight: limit-of-detection censoring, done the robust way

Wastewater concentration below the assay's limit of detection is **left-censored**:
we know only that the true concentration was at or below the LOD. The CDC source
hand-writes this by splitting the observations into two index sets and adding
`target += normal_lcdf(ww_log_lod | mean, sigma)` for the censored subset.

The port expresses the whole observation vector in one line:

```julia
log_conc ~ censored(normal, model_log_conc[ww_day], sigma_ww; lower = lod)
```

`censored(normal, …; lower = lod)` is a distribution higher-order function:
observations at the lower bound contribute the log-CDF (the censored atom),
observations above it contribute the ordinary normal density, and StanBlocks
emits the paired pointwise-likelihood and predictive-draw generated quantities
automatically. Crucially, the emitted log-CDF routes through StanBlocks'
**`erfc`-stable** helper (emitted Stan)

```c
real normal_lcdf_stable(real x, real loc, real scale) {
    return log(erfc((-(x - loc)) / (scale * sqrt(2.0)))) - log(2.0);
}
```

rather than Stan Math's `normal_lcdf`, whose reverse-mode gradient loses accuracy
in the far lower tail. So the transpiled program is not just a mechanical
translation of the original — for the censored likelihood it is numerically
*better* than the hand-written version, from the same one-line author syntax.

## Full Julia source and generated Stan code

The build evaluates the exact displayed source and emits the complete Stan
program. Note the two `@deffun` scans lowered to clean Stan `for` loops, the
`lower_clamping_normal` family in the model block, and the `erfc`-stable
`normal_lcdf_stable` it depends on.

```@raw html
<div class="atlas-comparison" data-atlas-comparison>
```

```@eval
Main.FeatureAtlasDocs.comparison(Main.FeatureAtlasDocs.example_module(:WastewaterCaseStudy), raw"""
using StanBlocks

@deffun begin
    "Backward convolution at day `t`: sum_k x[t-k] * pmf[k], pmf in forward (lag) order."
    ww_conv_at(x::vector[nx], pmf::vector[np], t::int)::real = begin
        acc = 0.0
        kmax = min(np, t - 1)
        for k in 1:kmax
            acc = acc + x[t - k] * pmf[k]
        end
        acc
    end

    "Vector of backward convolutions over `out_n` days, offset into `x` by `off`."
    ww_conv(x::vector[nx], pmf::vector[np], out_n::int, off::int)::vector[out_n] = begin
        out::vector[out_n]
        for i in 1:out_n
            out[i] = ww_conv_at(x, pmf, i + off)
        end
        out
    end

    "First-order Gaussian random walk of length m+1 from x0 and innovations z."
    ww_rw(x0::real, sd::real, z::vector[m])::vector[m+1] = begin
        out::vector[m+1]
        out[1] = x0
        for i in 1:m
            out[i+1] = out[i] + sd * z[i]
        end
        out
    end

    "Renewal recurrence with infection feedback. Returns latent infections only;
     the CDC source returns tuple(infections, Rt) and we recompute Rt separately."
    ww_renewal(log_i0::real, growth::real, log_rt::vector[nt],
               gen_int::vector[gmax], fb_pmf::vector[fl],
               feedback::real, uot::int)::vector[nt] = begin
        infections::vector[nt]
        for t in 1:uot
            infections[t] = exp(log_i0 + growth * (t - 1))
        end
        for t in (uot + 1):nt
            infectiousness = ww_conv_at(infections, gen_int, t)
            fbk = ww_conv_at(infections, fb_pmf, t)
            rt_eff = exp(log_rt[t] - feedback * fbk)
            infections[t] = rt_eff * infectiousness
        end
        infections
    end
end

ww_model = @slic (;
    week_of_day = repeat(1:5, inner = 7),                 # int[35] day -> week
    day_of_week = repeat(1:7, outer = 4),                 # int[28] observed day-of-week
    hosp        = fill(12, 28),                           # int[28] admissions
    gen_int     = [0.15, 0.30, 0.30, 0.15, 0.10],         # generation interval pmf
    fb_pmf      = [0.5, 0.3, 0.2],                         # infection-feedback pmf
    inf_to_hosp = [0.02, 0.1, 0.2, 0.25, 0.2, 0.12, 0.07, 0.04],  # infection->admission delay
    shed        = [0.05, 0.15, 0.25, 0.2, 0.15, 0.1, 0.07, 0.03], # shedding kernel
    log_conc    = [3.4, 2.0, 4.1, 3.8, 2.0, 3.2, 4.4, 3.9, 2.0, 3.1, 4.0, 3.7],
    lod         = fill(2.0, 12),                          # per-sample limit of detection
    ww_day      = collect(1:2:24),                        # int[12] sampled observed days
    state_pop   = 1.0e5,
    mwpd        = 1.0e6,
) begin
    nt      = dims(week_of_day)[1]
    ot      = dims(hosp)[1]
    uot     = nt - ot
    n_weeks = maximum(week_of_day)

    # Rt: weekly Gaussian random walk on log scale, expanded to daily
    log_r0 ~ normal(0.0, 0.2)
    eta_sd ~ normal(0.0, 0.1; lower = 0.0)
    w :: vector[n_weeks - 1] ~ std_normal()
    log_rt_weekly = ww_rw(log_r0, eta_sd, w)
    log_rt_daily  = log_rt_weekly[week_of_day]

    # latent infections via the renewal recurrence
    log_i0   ~ normal(-13.0, 1.0)
    growth   ~ normal(0.0, 0.05)
    feedback ~ normal(0.0, 0.01; lower = 0.0)
    infections = ww_renewal(log_i0, growth, log_rt_daily, gen_int, fb_pmf, feedback, uot)

    # hospital admissions: neg-binomial with day-of-week simplex effect
    ihr ~ beta(2.0, 20.0)
    phi ~ gamma(2.0, 0.1)
    hosp_wday_effect :: simplex[7] ~ dirichlet(rep_vector(1.0, 7))
    wday        = 7.0 * hosp_wday_effect[day_of_week]
    exp_hosp_pc = ww_conv(infections, inf_to_hosp, ot, uot)
    exp_hosp    = (state_pop * ihr) * exp_hosp_pc .* wday
    hosp ~ neg_binomial_2(exp_hosp, phi)

    # wastewater: LOD-censored normal on the log-concentration scale
    log10_g ~ normal(9.0, 1.0)
    sigma_ww ~ normal(0.0, 1.0; lower = 0.0)
    conc_pc        = ww_conv(infections, shed, ot, uot)
    model_log_conc = log(10.0) * log10_g + log(conc_pc + 1.0e-8) - log(mwpd)
    log_conc ~ censored(normal, model_log_conc[ww_day], sigma_ww; lower = lod)
end
""", :ww_model)
```

```@raw html
</div>
```

## The full multi-subpopulation model

The published model is not single-population: it runs several coupled
sub-epidemics and aggregates them to the state level. The faithful port below
carries that structure. Each subpopulation has its own renewal recurrence and
its own reproduction number — a reference subpopulation whose weekly `log(Rt)`
follows a random walk, and the others tracking it through a stationary AR(1)
deviation. A single `@deffun` returns the whole `nt x S` infection matrix, so
both the aggregate admissions signal (`M * pop_frac`, a matrix-vector product)
and the per-subpopulation wastewater concentrations derive from one value —
avoiding the tuple return the original `generate_infections` uses.

Wastewater samples carry a subpopulation label as well as a day, so the
observation vector is *gathered* out of the `nt x S` concentration matrix at the
sampled `(subpopulation, day)` pairs (`ww_gather`). The admissions and
LOD-censored wastewater likelihoods are exactly as in the single-population
core.

```@raw html
<div class="atlas-comparison" data-atlas-comparison>
```

```@eval
Main.FeatureAtlasDocs.comparison(Main.FeatureAtlasDocs.example_module(:WastewaterMultiSubpop), raw"""
using StanBlocks

@deffun begin
    ww_conv_at(x::vector[nx], pmf::vector[np], t::int)::real = begin
        acc = 0.0
        kmax = min(np, t - 1)
        for k in 1:kmax
            acc = acc + x[t - k] * pmf[k]
        end
        acc
    end
    ww_conv(x::vector[nx], pmf::vector[np], out_n::int, off::int)::vector[out_n] = begin
        out::vector[out_n]
        for i in 1:out_n
            out[i] = ww_conv_at(x, pmf, i + off)
        end
        out
    end
    ww_rw(x0::real, sd::real, z::vector[m])::vector[m+1] = begin
        out::vector[m+1]
        out[1] = x0
        for i in 1:m
            out[i+1] = out[i] + sd * z[i]
        end
        out
    end
    "Stationary AR(1) around 0: dev[1] = sd/sqrt(1-ac^2) * z[1]; dev[t] = ac*dev[t-1] + sd*z[t]."
    ww_ar1(ac::real, sd::real, z::vector[m])::vector[m] = begin
        out::vector[m]
        out[1] = sd / sqrt(1.0 - ac * ac) * z[1]
        for t in 2:m
            out[t] = ac * out[t-1] + sd * z[t]
        end
        out
    end
    "Per-subpopulation renewal. Returns the nt x S infection matrix (each column
     one sub-epidemic's scan), so the aggregate and per-subpop wastewater both
     derive from it with no tuple return."
    subpop_infections(log_i0::vector[S], growth::vector[S], log_rt::matrix[nt, S],
                      gen_int::vector[gmax], fb_pmf::vector[fl],
                      feedback::real, uot::int)::matrix[nt, S] = begin
        M::matrix[nt, S]
        for s in 1:S
            for t in 1:uot
                M[t, s] = exp(log_i0[s] + growth[s] * (t - 1))
            end
            for t in (uot + 1):nt
                infness = 0.0
                kmax = min(gmax, t - 1)
                for k in 1:kmax
                    infness = infness + M[t - k, s] * gen_int[k]
                end
                fbk = 0.0
                fmax = min(fl, t - 1)
                for k in 1:fmax
                    fbk = fbk + M[t - k, s] * fb_pmf[k]
                end
                rt_eff = exp(log_rt[t, s] - feedback * fbk)
                M[t, s] = rt_eff * infness
            end
        end
        M
    end
    "Gather per-subpop wastewater log-concentration at observation (subpop, day) pairs."
    ww_gather(conc::matrix[nt, S], subpop::int[nobs], day::int[nobs])::vector[nobs] = begin
        out::vector[nobs]
        for i in 1:nobs
            out[i] = conc[day[i], subpop[i]]
        end
        out
    end
    "Daily per-subpop log-Rt = reference weekly RW + per-subpop AR(1) weekly
     deviation, expanded to daily via the week-of-day index. Returns nt x S."
    ww_subpop_daily_rt(ref_weekly::vector[nw], z::vector[nz], S::int, ac::real, sd::real,
                       week_of_day::int[nt])::matrix[nt, S] = begin
        W::matrix[nt, S]
        for s in 1:S
            dev = ww_ar1(ac, sd, segment(z, (s - 1) * nw + 1, nw))  # vector[nw] for subpop s
            for t in 1:nt
                wk = week_of_day[t]
                W[t, s] = ref_weekly[wk] + dev[wk]
            end
        end
        W
    end
    "Per-subpop shedding convolution -> nt x S log-concentration matrix."
    ww_shed_conc(M::matrix[nt, S], shed::vector[ns], log10_g::real, mwpd::real)::matrix[nt, S] = begin
        C::matrix[nt, S]
        for s in 1:S
            for t in 1:nt
                net = 0.0
                kmax = min(ns, t)
                for k in 1:kmax
                    net = net + M[t - k + 1, s] * shed[k]
                end
                C[t, s] = log(10.0) * log10_g + log(net + 1.0e-8) - log(mwpd)
            end
        end
        C
    end
end

ww_multi = @slic (;
    week_of_day = repeat(1:5, inner = 7),      # int[35] day -> week
    day_of_week = repeat(1:7, outer = 4),      # int[28] observed day-of-week
    hosp        = fill(20, 28),                # int[28] state admissions
    gen_int     = [0.15, 0.30, 0.30, 0.15, 0.10],
    fb_pmf      = [0.5, 0.3, 0.2],
    inf_to_hosp = [0.02, 0.1, 0.2, 0.25, 0.2, 0.12, 0.07, 0.04],
    shed        = [0.05, 0.15, 0.25, 0.2, 0.15, 0.1, 0.07, 0.03],
    pop_frac    = [0.6, 0.4],                  # 2 subpopulations
    ww_conc     = [3.4, 2.0, 4.1, 3.8, 2.0, 3.2, 4.4, 3.9, 2.0, 3.1, 4.0, 3.7],
    ww_lod      = fill(2.0, 12),
    ww_day      = [3, 7, 10, 14, 17, 21, 24, 5, 9, 13, 19, 23],   # observed day per ww sample
    ww_subpop   = [1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2],           # subpop per ww sample
    state_pop   = 1.0e5, mwpd = 1.0e6,
) begin
    nt      = dims(week_of_day)[1]
    ot      = dims(hosp)[1]
    uot     = nt - ot
    n_weeks = maximum(week_of_day)
    S       = dims(pop_frac)[1]

    # --- reference weekly log-Rt: random walk; per-subpop AR(1) deviations ---
    log_r0 ~ normal(0.0, 0.2)
    eta_sd ~ normal(0.0, 0.1; lower = 0.0)
    w_ref :: vector[n_weeks - 1] ~ std_normal()
    autoreg ~ beta(2.0, 2.0)
    sigma_subpop ~ normal(0.0, 0.1; lower = 0.0)
    z_subpop :: vector[n_weeks * S] ~ std_normal()            # per-subpop weekly innovations
    log_rt_ref_weekly = ww_rw(log_r0, eta_sd, w_ref)          # vector[n_weeks]
    # build the daily per-subpop log-Rt matrix
    log_rt_daily = ww_subpop_daily_rt(log_rt_ref_weekly, z_subpop, S, autoreg, sigma_subpop, week_of_day)

    # --- per-subpop renewal + aggregation ---
    log_i0   :: vector[S] ~ normal(-13.0, 1.0)
    growth   :: vector[S] ~ normal(0.0, 0.05)
    feedback ~ normal(0.0, 0.01; lower = 0.0)
    M = subpop_infections(log_i0, growth, log_rt_daily, gen_int, fb_pmf, feedback, uot)  # matrix[nt,S]
    state_inf = M * pop_frac                                   # vector[nt] aggregate per-capita

    # --- hospital admissions (state level): neg-binomial + day-of-week simplex ---
    ihr ~ beta(2.0, 20.0)
    phi ~ gamma(2.0, 0.1)
    hosp_wday_effect :: simplex[7] ~ dirichlet(rep_vector(1.0, 7))
    wday        = 7.0 * hosp_wday_effect[day_of_week]
    exp_hosp_pc = ww_conv(state_inf, inf_to_hosp, ot, uot)
    exp_hosp    = (state_pop * ihr) * exp_hosp_pc .* wday
    hosp ~ neg_binomial_2(exp_hosp, phi)

    # --- wastewater (per subpopulation): LOD-censored normal on log-conc ---
    log10_g ~ normal(9.0, 1.0)
    sigma_ww ~ normal(0.0, 1.0; lower = 0.0)
    conc_matrix = ww_shed_conc(M, shed, log10_g, mwpd)         # matrix[nt,S]
    model_conc  = ww_gather(conc_matrix, ww_subpop, ww_day)    # vector[nobs]
    ww_conc ~ censored(normal, model_conc, sigma_ww; lower = ww_lod)
end
""", :ww_multi)
```

```@raw html
</div>
```

## What this exercises

- A **`@deffun` scan** for the renewal recurrence and the weekly random walk —
  the sequential structure a `@slic` body and a `plate` cannot express — lowered
  to clean Stan `for` loops.
- **Fancy indexing** for the weekly→daily `Rt` expansion and the day-of-week
  simplex effect.
- A **`neg_binomial_2`** admissions likelihood and a **`censored(normal, …)`**
  wastewater likelihood in the same joint model, each with automatically emitted
  posterior-predictive and pointwise-likelihood generated quantities.
- The **`erfc`-stable** log-CDF for the below-LOD contribution — a more accurate
  gradient than the hand-written `target += normal_lcdf` in the original source.
- In the multi-subpopulation model: a **matrix-returning `@deffun`** for the
  per-subpopulation renewal (so the aggregate and the per-subpopulation
  wastewater derive from one value, avoiding a tuple return), a **matrix-vector
  aggregation** `M * pop_frac`, `segment` slicing of a flat innovation vector,
  and a **gather** of the observation vector out of the concentration matrix at
  the sampled `(subpopulation, day)` pairs.
