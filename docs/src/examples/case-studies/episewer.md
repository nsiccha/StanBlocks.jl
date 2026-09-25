# EpiSewer — a composable renewal library

[R `EpiSewer`](https://github.com/adrian-lison/EpiSewer) packs a large model
space into one config-driven Stan program: several reproduction-number
processes, several observation families, and seeding / shedding / limit-of-detection
/ digital-PCR / outlier options, each selected at runtime by an integer flag.
[`EpiSewer.jl`](https://github.com/seabbs/EpiSewer.jl) describes the *same* model
space the opposite way — as composable Turing components, with no monolith.

StanBlocks realizes both the same way, and it is the way the two designs already
point at: a small library of pieces assembled at **construction** time. The
runtime flags of the R monolith become plain Julia composition, and the pieces
line up one-to-one with `EpiSewer.jl`'s components. This page shows a slice of
that library — one renewal core, a swappable Rt process, and a swappable
observation family — and transpiles three assembled configurations to three
focused Stan programs. The arrays are build fixtures, not real data.

!!! note "Companion port in BayesianRegressionModels.jl"
    BRM ports the same EpiSewer renewal core through its formula interface:
    [Wastewater-based Rt inference (EpiSewer / `ww-inference-model`)](https://nsiccha.github.io/BayesianRegressionModels.jl/dev/wastewater).
    That page frames the renewal model as a `@brm` `kernel(...)` term; this page
    shows the pure-StanBlocks `@slic` / `@deffun` library the same backend emits.

## One renewal core, swappable Rt process and observation family

Two axes vary here, both by composition:

- **Rt process** — a weekly random walk, or a **sparse spline** basis evaluated
  with the `csr_matrix_times_vector` primitive (the R model's `R_model` 0/1/3
  spline processes, in compressed-row form).
- **Observation family** — an **observation submodel** per family:
  `obs_lod` measures log-concentration with an LOD-censored normal, and
  `obs_dpcr` measures digital-PCR **positive-partition counts**, using `to_int`
  for the real→integer partition total. Each family pairs the same renewal core
  with a different measurement node.

Each model is one `Base.merge` of the core with an Rt fragment; the monolith's
`if`-on-a-flag is gone.

```@raw html
<div class="atlas-comparison" data-atlas-comparison data-stan-label="EpiSewer library">
```

```@eval
Main.FeatureAtlasDocs.comparisons(Main.FeatureAtlasDocs.example_module(:EpiSewerLib), raw"""
using StanBlocks

@deffun begin
    es_conv_at(x::vector[nx], pmf::vector[np], t::int)::real = begin
        acc = 0.0
        kmax = min(np, t - 1)
        for k in 1:kmax
            acc = acc + x[t - k] * pmf[k]
        end
        acc
    end
    es_conv(x::vector[nx], pmf::vector[np], out_n::int, off::int)::vector[out_n] = begin
        out::vector[out_n]
        for i in 1:out_n
            out[i] = es_conv_at(x, pmf, i + off)
        end
        out
    end
    es_rw(x0::real, sd::real, z::vector[m])::vector[m+1] = begin
        out::vector[m+1]
        out[1] = x0
        for i in 1:m
            out[i+1] = out[i] + sd * z[i]
        end
        out
    end
    es_renewal(log_i0::real, growth::real, log_rt::vector[nt],
               gen_int::vector[gmax], uot::int)::vector[nt] = begin
        infections::vector[nt]
        for t in 1:uot
            infections[t] = exp(log_i0 + growth * (t - 1))
        end
        for t in (uot + 1):nt
            infections[t] = exp(log_rt[t]) * es_conv_at(infections, gen_int, t)
        end
        infections
    end
end

# ── Swappable observation-family submodels (§3 Form A) ──
# Concentration measured with LOD-censored normal.
obs_lod = @slic begin
    log10_g ~ normal(9.0, 1.0)
    sigma ~ normal(0.0, 1.0; lower = 0.0)
    model_conc = log(10.0) * log10_g + log(signal + 1.0e-8) - log(mwpd)
    obs ~ censored(normal, model_conc, sigma; lower = lod)
    return obs
end
# Digital-PCR positive-partition counts: real->int partition total via `to_int`.
obs_dpcr = @slic begin
    phi ~ gamma(2.0, 0.1)
    n_part = to_int(round(n_partitions))              # real -> int (data-side)
    lambda = (1.0e5 * signal) * n_part / mwpd
    obs ~ neg_binomial_2(lambda, phi)
    return obs
end

es_data = (;
    gen_int     = [0.15, 0.30, 0.30, 0.15, 0.10],
    shed        = [0.05, 0.15, 0.25, 0.2, 0.15, 0.1, 0.07, 0.03],
    week_of_day = repeat(1:5, inner = 7),
    conc        = [3.4, 2.0, 4.1, 3.8, 2.0, 3.2, 4.4, 3.9, 2.0, 3.1, 4.0, 3.7,
                   3.6, 2.0, 4.2, 3.5, 3.9, 2.0, 4.0, 3.7, 3.3, 2.0, 4.1, 3.8,
                   2.0, 3.2, 4.4, 3.9],
    lod         = fill(2.0, 28),
    partitions_pos = fill(1200, 28),
    n_partitions   = 20000.0,
    spl_w = vcat([[0.5, 0.5] for _ in 1:28]...),
    spl_v = vcat([[min(i, 5), min(i, 5) + 1] for i in 1:28]...),
    spl_u = collect(1:2:(2*28 + 1)),
    n_basis = 6, mwpd = 1.0e6,
)

# ── Renewal cores: one per observation family, each leaving the Rt process
#    (`log_rt`) open. The concentration and dPCR families pair the SAME renewal
#    core with a different observation submodel. ──
es_base_conc = @slic es_data begin
    nt  = dims(week_of_day)[1]
    ot  = dims(conc)[1]
    uot = nt - ot
    log_i0 ~ normal(-13.0, 1.0)
    growth ~ normal(0.0, 0.05)
    infections = es_renewal(log_i0, growth, log_rt, gen_int, uot)   # log_rt UNBOUND
    signal = es_conv(infections, shed, ot, uot)
    conc ~ obs_lod(; signal, mwpd, lod)
end
es_base_dpcr = @slic es_data begin
    nt  = dims(week_of_day)[1]
    ot  = dims(partitions_pos)[1]
    uot = nt - ot
    log_i0 ~ normal(-13.0, 1.0)
    growth ~ normal(0.0, 0.05)
    infections = es_renewal(log_i0, growth, log_rt, gen_int, uot)   # log_rt UNBOUND
    signal = es_conv(infections, shed, ot, uot)
    partitions_pos ~ obs_dpcr(; signal, n_partitions, mwpd)
end

# ── Swappable Rt processes ──
rt_rw = quote
    log_r0 ~ normal(0.0, 0.2)
    eta_sd ~ normal(0.0, 0.1; lower = 0.0)
    n_weeks = maximum(week_of_day)
    w :: vector[n_weeks - 1] ~ std_normal()
    log_rt = es_rw(log_r0, eta_sd, w)[week_of_day]
end
rt_spline = quote
    beta :: vector[n_basis] ~ std_normal()
    log_rt = csr_matrix_times_vector(nt, n_basis, spl_w, spl_v, spl_u, beta)
end

# ── The composable library — swap the Rt process (and the observation family) ──
es_models = (;
    rw_lod     = Base.merge(es_base_conc, rt_rw),      # RW Rt,     LOD concentration
    spline_lod = Base.merge(es_base_conc, rt_spline),  # CSR spline Rt, LOD concentration
    rw_dpcr    = Base.merge(es_base_dpcr, rt_rw),      # RW Rt,     dPCR counts (to_int)
)
""", :es_models)
```

```@raw html
</div>
```

## What this exercises

- **`to_int`** — the real→integer partition total in the digital-PCR observation
  family (`to_int(round(n_partitions))`), a data-side conversion that lands in
  `transformed data` where Stan's `data`-qualified-argument rule is satisfied.
- **`csr_matrix_times_vector`** — the sparse compressed-row spline basis for the
  spline Rt process, exposed directly rather than densified.
- **Observation submodels** — the concentration and dPCR measurement families as
  self-contained, swappable nodes.
- **`Base.merge` composition** — the R monolith's runtime module flags become
  construction-time assembly, mirroring `EpiSewer.jl`'s component design.

For the full joint renewal model — two coupled observation streams, a
multi-subpopulation structure, tuple-returning renewal and per-site effects —
see the [wastewater renewal model](wastewater.md) case study.
