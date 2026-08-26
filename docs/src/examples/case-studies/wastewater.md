# Wastewater renewal model

This example ports the CDC
[`ww-inference-model`](https://github.com/CDCgov/ww-inference-model)
(`inst/stan/wwinference.stan`) — a joint model that infers latent infections
from **two** noisy signals at once: hospital admissions and pathogen
concentration in wastewater. It is a *renewal* model: today's infections are
generated from recent infections and a time-varying reproduction number, rather
than from a mechanistic compartmental system.

The published program is a single monolith with a runtime flag for each modeling
choice. The port takes the shape StanBlocks makes natural instead: a small
library of composable pieces, assembled at *construction* time. This page climbs
a **modeling ladder** — a base model, then rungs that swap one component at a
time — and ends at the full multi-subpopulation model. Every rung is evaluated
at documentation-build time, so the displayed Julia is the exact source that
produced the Stan beside it. The small arrays are build fixtures, not real
surveillance data.

!!! note "Companion port in BayesianRegressionModels.jl"
    The **same** CDC `ww-inference-model` is also ported in BRM, onto its
    StanBlocks backend (`@slic` + `@deffun`):
    [The full CDC `ww-inference-model`](https://nsiccha.github.io/BayesianRegressionModels.jl/dev/wastewater-cdc).
    Both target the identical upstream model — this page as a standalone
    StanBlocks modeling ladder, the BRM page from the `@brm` / StanBlocks layer
    boundary.

## Two observation streams, two removable submodels

The joint model observes two data streams. Each is an **observation submodel**
(`data ~ submodel(...)`): a self-contained node that carries its own parameters
*and* its own likelihood, added or dropped as a single line.

- `admissions_stream` carries the infection-hospitalization ratio, the
  negative-binomial dispersion, and a `simplex[7]` day-of-week reporting effect,
  and observes admissions with `neg_binomial_2`.
- `wastewater_stream` carries the genome-scaling and measurement noise, and
  observes log-concentration with `censored(normal, …; lower = lod)` — so
  below-limit-of-detection samples contribute the log-CDF automatically.

The renewal **core** produces the latent infection curve and leaves two things
open: the **Rt process** (an unbound `log_rt`) and the **shedding kernel** (a
`shed_kernel` that defaults to a data pmf). Each observation stream is one line;
each ladder rung below binds `log_rt` (or swaps `shed_kernel`) through
`Base.merge`, without touching the core or the streams.

## The Rt ladder — one core, three reproduction-number processes

The same core becomes three models by merging in a different Rt fragment. The
CDC source selects between these with an integer flag; here each is a value you
compose in. `spline_csr` uses the sparse `csr_matrix_times_vector` primitive on
a compressed-row spline basis; `rw_parametric` additionally swaps the data
shedding pmf for a parametric viral-shedding trajectory driven by peak-timing
parameters.

```@raw html
<div class="atlas-comparison" data-atlas-comparison data-stan-label="Rt ladder">
```

```@eval
Main.FeatureAtlasDocs.comparisons(Main.FeatureAtlasDocs.example_module(:CDCLadder), raw"""
using StanBlocks

@deffun begin
    cdc_conv_at(x::vector[nx], pmf::vector[np], t::int)::real = begin
        acc = 0.0
        kmax = min(np, t - 1)
        for k in 1:kmax
            acc = acc + x[t - k] * pmf[k]
        end
        acc
    end
    cdc_conv(x::vector[nx], pmf::vector[np], out_n::int, off::int)::vector[out_n] = begin
        out::vector[out_n]
        for i in 1:out_n
            out[i] = cdc_conv_at(x, pmf, i + off)
        end
        out
    end
    cdc_rw(x0::real, sd::real, z::vector[m])::vector[m+1] = begin
        out::vector[m+1]
        out[1] = x0
        for i in 1:m
            out[i+1] = out[i] + sd * z[i]
        end
        out
    end
    "diff-AR(1): the weekly *differences* follow an AR(1) around 0; the level is
     x0 + cumulative sum of the differences (the CDC reference-Rt process)."
    cdc_diff_ar1(x0::real, ar::real, sd::real, z::vector[m])::vector[m+1] = begin
        diffs::vector[m]
        diffs[1] = sd * z[1]
        for t in 2:m
            diffs[t] = ar * diffs[t-1] + sd * z[t]
        end
        lvl::vector[m+1]
        lvl[1] = x0
        for t in 1:m
            lvl[t+1] = lvl[t] + diffs[t]
        end
        lvl
    end
    "Parametric viral-shedding trajectory: a triangular-on-log profile peaking at
     day t_peak, declining to zero at dur_shed, normalised to sum 1."
    cdc_vl_trajectory(t_peak::real, viral_peak::real, dur_shed::real, n::int)::vector[n] = begin
        v::vector[n]
        for k in 1:n
            tk = k * 1.0
            frac = tk <= t_peak ? tk / t_peak : (dur_shed - tk) / (dur_shed - t_peak)
            height = frac > 0.0 ? frac : 0.0
            v[k] = exp(viral_peak * height)
        end
        v / sum(v)
    end
    cdc_renewal(log_i0::real, growth::real, log_rt::vector[nt],
                gen_int::vector[gmax], uot::int)::vector[nt] = begin
        infections::vector[nt]
        for t in 1:uot
            infections[t] = exp(log_i0 + growth * (t - 1))
        end
        for t in (uot + 1):nt
            infections[t] = exp(log_rt[t]) * cdc_conv_at(infections, gen_int, t)
        end
        infections
    end
end

# ── Observation submodels (§3 Form A): one removable node per stream ──
admissions_stream = @slic begin
    ihr ~ beta(2.0, 20.0)
    phi ~ gamma(2.0, 0.1)
    wday_effect :: simplex[7] ~ dirichlet(rep_vector(1.0, 7))
    wday = 7.0 * wday_effect[day_of_week]
    mu = (state_pop * ihr) * exp_hosp_pc .* wday
    obs ~ neg_binomial_2(mu, phi)
    return obs
end
wastewater_stream = @slic begin
    log10_g ~ normal(9.0, 1.0)
    sigma ~ normal(0.0, 1.0; lower = 0.0)
    model_conc = log(10.0) * log10_g + log(conc_pc + 1.0e-8) - log(mwpd)
    obs ~ censored(normal, model_conc, sigma; lower = lod)
    return obs
end

cdc_data = (;
    week_of_day = repeat(1:5, inner = 7),
    day_of_week = repeat(1:7, outer = 4),
    hosp        = fill(20, 28),
    gen_int     = [0.15, 0.30, 0.30, 0.15, 0.10],
    inf_to_hosp = [0.02, 0.1, 0.2, 0.25, 0.2, 0.12, 0.07, 0.04],
    shed        = [0.05, 0.15, 0.25, 0.2, 0.15, 0.1, 0.07, 0.03],
    ww_conc     = [3.4, 2.0, 4.1, 3.8, 2.0, 3.2, 4.4, 3.9, 2.0, 3.1, 4.0, 3.7,
                   3.6, 2.0, 4.2, 3.5, 3.9, 2.0, 4.0, 3.7, 3.3, 2.0, 4.1, 3.8,
                   2.0, 3.2, 4.4, 3.9],
    ww_lod      = fill(2.0, 28),
    # sparse spline basis (28 obs days x 6 basis fns), banded CSR triple:
    spl_w = vcat([[0.5, 0.5] for _ in 1:28]...),
    spl_v = vcat([[min(i, 5), min(i, 5) + 1] for i in 1:28]...),
    spl_u = collect(1:2:(2*28 + 1)),
    n_basis = 6, n_shed = 8,
    state_pop = 1.0e5, mwpd = 1.0e6,
)

# ── Renewal core with an UNBOUND `log_rt` (Rt axis) and a swappable shedding
#    kernel (`shed_kernel`, data by default). Each observation stream is one line. ──
cdc_core = @slic cdc_data begin
    nt  = dims(week_of_day)[1]
    ot  = dims(hosp)[1]
    uot = nt - ot
    n_weeks = maximum(week_of_day)
    log_i0 ~ normal(-13.0, 1.0)
    growth ~ normal(0.0, 0.05)
    infections  = cdc_renewal(log_i0, growth, log_rt, gen_int, uot)   # log_rt UNBOUND
    exp_hosp_pc = cdc_conv(infections, inf_to_hosp, ot, uot)
    shed_kernel = shed                                               # shedding axis (data default)
    conc_pc     = cdc_conv(infections, shed_kernel, ot, uot)
    hosp    ~ admissions_stream(; day_of_week, state_pop, exp_hosp_pc)
    ww_conc ~ wastewater_stream(; conc_pc, mwpd, lod = ww_lod)
end

# ── Swappable Rt-process fragments (each binds `log_rt`) ──
rt_rw = quote
    log_r0 ~ normal(0.0, 0.2)
    eta_sd ~ normal(0.0, 0.1; lower = 0.0)
    w :: vector[n_weeks - 1] ~ std_normal()
    log_rt = cdc_rw(log_r0, eta_sd, w)[week_of_day]
end
rt_diff_ar1 = quote
    log_r0 ~ normal(0.0, 0.2)
    ar ~ beta(2.0, 2.0)
    eta_sd ~ normal(0.0, 0.1; lower = 0.0)
    z :: vector[n_weeks - 1] ~ std_normal()
    log_rt = cdc_diff_ar1(log_r0, ar, eta_sd, z)[week_of_day]
end
rt_spline = quote
    beta :: vector[n_basis] ~ std_normal()
    log_rt = csr_matrix_times_vector(nt, n_basis, spl_w, spl_v, spl_u, beta)
end
# ── Swappable shedding fragment (parametric VL trajectory) ──
shed_parametric = quote
    t_peak ~ normal(3.0, 1.0; lower = 0.0)
    viral_peak ~ normal(2.0, 1.0)
    dur_shed ~ normal(8.0, 2.0; lower = 0.0)
    shed_kernel = cdc_vl_trajectory(t_peak, viral_peak, dur_shed, n_shed)
end

# ── The modeling ladder — each rung a composition on the previous ──
ladder = (;
    rw            = Base.merge(cdc_core, rt_rw),
    diff_ar1      = Base.merge(cdc_core, rt_diff_ar1),
    spline_csr    = Base.merge(cdc_core, rt_spline),
    rw_parametric = Base.merge(cdc_core, rt_rw, shed_parametric),
)
""", :ladder)
```

```@raw html
</div>
```

## The full multi-subpopulation model

The published model is not single-population: it runs several coupled
sub-epidemics and aggregates them to the state level. The capstone below carries
that structure faithfully and showcases the remaining features:

- a **`plate` over sub-epidemics** — each cell samples that subpopulation's own
  initial size, growth, and weekly AR(1) innovations and returns its infection
  column; the `vector[nt]` cell outputs collect into a `matrix[nt, S]`. The
  unavoidable within-subpop *time* recurrence (a scan) lives in a `@deffun` the
  cell calls; the subpopulation axis is the plate. This matches the companion
  BRM port, which expresses the same subpop axis as a `plate`;
- **per-subpopulation `Rt`** — a shared reference weekly random walk plus a
  stationary AR(1) deviation per subpopulation (its own innovations sampled
  inside the plate cell), and **infection feedback** in the renewal recurrence;
- **aggregation** to admissions by a matrix-vector product `M * pop_frac`, and
  per-subpopulation wastewater **gathered** at the sampled `(subpopulation, day)`
  pairs;
- a **per-site** wastewater stream — its own site offset and per-lab-site noise,
  indexed by each sample's site.

The two observation submodels are reused unchanged; only the wastewater stream
gains the per-site index.

```@raw html
<div class="atlas-comparison" data-atlas-comparison data-stan-label="Multi-subpopulation model">
```

```@eval
Main.FeatureAtlasDocs.comparison(Main.FeatureAtlasDocs.example_module(:CDCCapstone), raw"""
using StanBlocks

@deffun begin
    cdc_conv_at(x::vector[nx], pmf::vector[np], t::int)::real = begin
        acc = 0.0
        kmax = min(np, t - 1)
        for k in 1:kmax
            acc = acc + x[t - k] * pmf[k]
        end
        acc
    end
    cdc_conv(x::vector[nx], pmf::vector[np], out_n::int, off::int)::vector[out_n] = begin
        out::vector[out_n]
        for i in 1:out_n
            out[i] = cdc_conv_at(x, pmf, i + off)
        end
        out
    end
    cdc_rw(x0::real, sd::real, z::vector[m])::vector[m+1] = begin
        out::vector[m+1]
        out[1] = x0
        for i in 1:m
            out[i+1] = out[i] + sd * z[i]
        end
        out
    end
    cdc_ar1(ac::real, sd::real, z::vector[m])::vector[m] = begin
        out::vector[m]
        out[1] = sd / sqrt(1.0 - ac * ac) * z[1]
        for t in 2:m
            out[t] = ac * out[t-1] + sd * z[t]
        end
        out
    end
    "Single sub-epidemic renewal for ONE `plate` cell: the subpop's daily log-Rt
     (shared reference weekly RW + that subpop's AR(1) deviation) drives the
     feedback-renewal scan. Returns the subpop's daily infection vector[nt]. The
     subpop LOOP is the `plate` below; this @deffun owns only the unavoidable
     within-subpop TIME recurrence (a scan, which a plate cell cannot express)."
    cdc_subpop_cell(log_i0::real, growth::real, log_rt_ref::vector[nw], z::vector[nz],
                    ac::real, sd::real, week_of_day::int[nt],
                    gen_int::vector[gmax], fb_pmf::vector[fl], feedback::real, uot::int)::vector[nt] = begin
        dev = cdc_ar1(ac, sd, z)
        M::vector[nt]
        for t in 1:uot
            M[t] = exp(log_i0 + growth * (t - 1))
        end
        for t in (uot + 1):nt
            infness = 0.0
            kmax = min(gmax, t - 1)
            for k in 1:kmax
                infness = infness + M[t - k] * gen_int[k]
            end
            fbk = 0.0
            fmax = min(fl, t - 1)
            for k in 1:fmax
                fbk = fbk + M[t - k] * fb_pmf[k]
            end
            wk = week_of_day[t]
            rt_eff = exp((log_rt_ref[wk] + dev[wk]) - feedback * fbk)
            M[t] = rt_eff * infness
        end
        M
    end
    "Per-subpop net shedding convolution -> nt x S log-scale matrix (the log10_g
     genome-scaling is added downstream in the wastewater stream)."
    cdc_shed_conc(M::matrix[nt, S], shed::vector[ns], mwpd::real)::matrix[nt, S] = begin
        C::matrix[nt, S]
        for s in 1:S
            for t in 1:nt
                net = 0.0
                kmax = min(ns, t)
                for k in 1:kmax
                    net = net + M[t - k + 1, s] * shed[k]
                end
                C[t, s] = log(net + 1.0e-8) - log(mwpd)
            end
        end
        C
    end
    "Gather per-subpop concentration at observation (subpop, day) pairs."
    cdc_gather(conc::matrix[nt, S], subpop::int[nobs], day::int[nobs])::vector[nobs] = begin
        out::vector[nobs]
        for i in 1:nobs
            out[i] = conc[day[i], subpop[i]]
        end
        out
    end
end

# Observation stream submodels (reused, per §3 Form A).
admissions_stream = @slic begin
    ihr ~ beta(2.0, 20.0)
    phi ~ gamma(2.0, 0.1)
    wday_effect :: simplex[7] ~ dirichlet(rep_vector(1.0, 7))
    wday = 7.0 * wday_effect[day_of_week]
    mu = (state_pop * ihr) * exp_hosp_pc .* wday
    obs ~ neg_binomial_2(mu, phi)
    return obs
end
# Per-site wastewater stream: its own log10_g + a per-lab-site sigma, indexed by
# the observation's site (ww_site_mod / per-site variance = faithful detail).
wastewater_site_stream = @slic begin
    log10_g ~ normal(9.0, 1.0)
    site_sigma :: vector[n_sites] ~ normal(0.0, 1.0; lower = 0.0)
    site_mod :: vector[n_sites] ~ normal(0.0, 0.5)
    model_conc = log(10.0) * log10_g + model_log_conc + site_mod[ww_site]
    obs ~ censored(normal, model_conc, site_sigma[ww_site]; lower = lod)
    return obs
end

cdc_capstone = @slic (;
    week_of_day = repeat(1:5, inner = 7),
    day_of_week = repeat(1:7, outer = 4),
    hosp        = fill(20, 28),
    gen_int     = [0.15, 0.30, 0.30, 0.15, 0.10],
    fb_pmf      = [0.5, 0.3, 0.2],
    inf_to_hosp = [0.02, 0.1, 0.2, 0.25, 0.2, 0.12, 0.07, 0.04],
    shed        = [0.05, 0.15, 0.25, 0.2, 0.15, 0.1, 0.07, 0.03],
    pop_frac    = [0.6, 0.4],
    ww_conc     = [3.4, 2.0, 4.1, 3.8, 2.0, 3.2, 4.4, 3.9, 2.0, 3.1, 4.0, 3.7],
    ww_lod      = fill(2.0, 12),
    ww_day      = [3, 7, 10, 14, 17, 21, 24, 5, 9, 13, 19, 23],
    ww_subpop   = [1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2],
    ww_site     = [1, 1, 2, 2, 3, 3, 3, 1, 2, 2, 3, 3],
    n_sites     = 3, n_shed = 8,
    state_pop   = 1.0e5, mwpd = 1.0e6,
) begin
    nt  = dims(week_of_day)[1]
    ot  = dims(hosp)[1]
    uot = nt - ot
    n_weeks = maximum(week_of_day)
    S = dims(pop_frac)[1]

    # shared reference weekly log-Rt (random walk), captured by every subpop cell
    log_r0 ~ normal(0.0, 0.2)
    eta_sd ~ normal(0.0, 0.1; lower = 0.0)
    w_ref :: vector[n_weeks - 1] ~ std_normal()
    autoreg ~ beta(2.0, 2.0)
    sigma_subpop ~ normal(0.0, 0.1; lower = 0.0)
    feedback ~ normal(0.0, 0.01; lower = 0.0)
    log_rt_ref = cdc_rw(log_r0, eta_sd, w_ref)

    # per-subpop renewal as a `plate` over the S sub-epidemics: each cell samples
    # its OWN initial size / growth / weekly AR(1) innovations and returns that
    # subpop's infection column. Shared pieces (log_rt_ref, feedback, autoreg, …)
    # are captured; the vector[nt] cell outputs collect into I_mat :: matrix[nt, S].
    I_mat :: matrix[nt, S] ~ plate(; outer = (S,)) do s
        log_i0_s ~ normal(-13.0, 1.0)
        growth_s ~ normal(0.0, 0.05)
        z_s :: vector[n_weeks] ~ std_normal()
        cdc_subpop_cell(log_i0_s, growth_s, log_rt_ref, z_s, autoreg, sigma_subpop,
                        week_of_day, gen_int, fb_pmf, feedback, uot)
    end
    state_inf = I_mat * pop_frac       # aggregate per-capita infections

    exp_hosp_pc = cdc_conv(state_inf, inf_to_hosp, ot, uot)
    hosp ~ admissions_stream(; day_of_week, state_pop, exp_hosp_pc)

    conc_matrix    = cdc_shed_conc(I_mat, shed, mwpd)    # net shed; log10_g added in the stream
    model_log_conc = cdc_gather(conc_matrix, ww_subpop, ww_day)
    ww_conc ~ wastewater_site_stream(; model_log_conc, ww_site, n_sites, lod = ww_lod)
end
""", :cdc_capstone)
```

```@raw html
</div>
```

## What this exercises

- **Observation submodels** (`data ~ submodel(...)`) — the two data streams as
  self-contained, removable nodes, each with its own parameters and likelihood.
- **Composition** — `Base.merge` swaps the Rt process and the shedding kernel
  into one shared core, dissolving the source's runtime-flag monolith into a
  construction-time modeling ladder.
- **`plate` over sub-epidemics** — per-subpopulation `~` parameters (initial
  size, growth, weekly AR(1) innovations) introduced *inside* the cell, with a
  shared reference-Rt random walk captured from the enclosing scope; the
  `vector[nt]` cell outputs collect into a `matrix[nt, S]`, matching the BRM
  companion port's subpop-axis structure.
- The **sparse CSR** primitive `csr_matrix_times_vector` for the spline Rt
  process, and a **parametric shedding kernel** built in a `@deffun`.
- **`@deffun` scans** for the renewal recurrence, the random-walk and AR(1) /
  diff-AR(1) Rt processes, and the shedding convolutions.
- **`censored(normal, …)`** for below-LOD wastewater, whose emitted log-CDF
  routes through StanBlocks' `erfc`-stable helper — a more accurate lower-tail
  gradient than the hand-written `target += normal_lcdf` in the original source.
- **Fancy indexing**, matrix-vector **aggregation**, and a **gather** at the
  sampled `(subpopulation, day, site)` observation points.
