"""
`stan_unc_names(model)` lists a program's unconstrained coordinate names as
BridgeStan's `param_unc_names` does, without compiling it, so a Julia consumer
of a fit's unconstrained draws can name them before or without a BridgeStan
build (snag `build-brm-s-coor-a2964345`). `stan_interface` carries the same list
as `"unconstrained"`. The expected names below are BridgeStan 2.9.0's, measured
on this model; the BridgeStan item checks them against the compiled program.
"""
@testsnippet UncNamesModel begin
    # Every declarable parameter center type, plain and as arrays (two- and
    # three-axis included), a one-element simplex and an empty vector. Every
    # parameter reaches the likelihood, so none moves to generated quantities.
    unc_names_model() = @slic (; K = 3, N = 2, M = 4, K1 = 1, Z = 0, alpha = [1.0, 1.0, 1.0],
            alpha1 = [1.0], y = 0.5) begin
        a ~ normal(0, 1)
        b::real[N] ~ normal(0, 1)
        b2::real[N, K]
        v::vector[K] ~ normal(0, 1; lower = 0.0)
        rv::row_vector[K] ~ normal(0, 1)
        mt::matrix[N, K] ~ normal(0, 1)
        av::vector[N, K]
        aav::vector[N, M, K]
        am::matrix[N, K, M]
        s::simplex[K] ~ dirichlet(alpha)
        as::simplex[N, K] ~ dirichlet(alpha)
        s1::simplex[K1] ~ dirichlet(alpha1)
        z0::vector[Z] ~ normal(0, 1)
        o::ordered[K] ~ normal(0, 1)
        po::positive_ordered[K] ~ exponential(1.0)
        ao::ordered[N, K] ~ normal(0, 1)
        Lc::cholesky_factor_corr[K] ~ lkj_corr_cholesky(2.0)
        La::cholesky_factor_corr[N, K] ~ lkj_corr_cholesky(2.0)
        Lv::cholesky_factor_cov[K]
        Sg::cov_matrix[K]
        C::corr_matrix[K]
        G::row_stochastic_matrix[N, M] ~ normal(0, 1)
        H::column_stochastic_matrix[N, M] ~ normal(0, 1)
        y ~ normal(a + sum(b) + b2[1, 1] + sum(v) + sum(rv) + sum(mt) + sum(av[1]) +
            sum(aav[1, 1]) + sum(am[1]) + sum(s) + sum(as[1]) + sum(s1) + sum(z0) + sum(o) +
            sum(po) + sum(ao[1]) + sum(Lc) + sum(La[1]) + sum(Lv) + sum(Sg) + sum(C) +
            sum(G) + sum(H), 1.0)
    end
end

@testitem "slic: stan_unc_names lists BridgeStan's unconstrained names without compiling" tags=[:slic, :descriptor, :interface, :regression] setup=[StanBlocksImports, UncNamesModel] begin
    m = unc_names_model()
    d = stan_descriptor(m)
    # BridgeStan's `param_unc_names` for this program. The typed array of
    # ordered vectors is emitted under a plate name, read from the descriptor.
    ao = string(only(o.name for o in d.outputs if o.kind === :parameter && startswith(string(o.name), "ao")))
    expected = split(replace("""
        a b.1 b.2 b2.1.1 b2.2.1 b2.1.2 b2.2.2 b2.1.3 b2.2.3 v.1 v.2 v.3 rv.1 rv.2 rv.3
        mt.1.1 mt.2.1 mt.1.2 mt.2.2 mt.1.3 mt.2.3 av.1.1 av.2.1 av.1.2 av.2.2 av.1.3 av.2.3
        aav.1.1.1 aav.2.1.1 aav.1.2.1 aav.2.2.1 aav.1.3.1 aav.2.3.1 aav.1.4.1 aav.2.4.1
        aav.1.1.2 aav.2.1.2 aav.1.2.2 aav.2.2.2 aav.1.3.2 aav.2.3.2 aav.1.4.2 aav.2.4.2
        aav.1.1.3 aav.2.1.3 aav.1.2.3 aav.2.2.3 aav.1.3.3 aav.2.3.3 aav.1.4.3 aav.2.4.3
        am.1.1.1 am.2.1.1 am.1.2.1 am.2.2.1 am.1.3.1 am.2.3.1 am.1.1.2 am.2.1.2 am.1.2.2
        am.2.2.2 am.1.3.2 am.2.3.2 am.1.1.3 am.2.1.3 am.1.2.3 am.2.2.3 am.1.3.3 am.2.3.3
        am.1.1.4 am.2.1.4 am.1.2.4 am.2.2.4 am.1.3.4 am.2.3.4 s.1 s.2 as.1.1 as.2.1 as.1.2
        as.2.2 o.1 o.2 o.3 po.1 po.2 po.3 AO.1.1 AO.2.1 AO.1.2 AO.2.2 AO.1.3 AO.2.3
        Lc.1 Lc.2 Lc.3 La.1.1 La.2.1 La.1.2 La.2.2 La.1.3 La.2.3 Lv.1 Lv.2 Lv.3 Lv.4 Lv.5
        Lv.6 Sg.1 Sg.2 Sg.3 Sg.4 Sg.5 Sg.6 C.1 C.2 C.3 G.1.1 G.2.1 G.1.2 G.2.2 G.1.3 G.2.3
        H.1.1 H.1.2 H.1.3 H.1.4
        """, "AO." => ao * "."))
    @test stan_unc_names(m) == expected
    @test stan_unc_names(d) == expected
    @test stan_interface(m)["unconstrained"] == expected
    # `simplex[1]` and `vector[0]` have no unconstrained coordinate.
    @test !any(n -> startswith(n, "s1") || startswith(n, "z0"), expected)

    # A rebound model names its rebound sizes.
    rebound = StanBlocks.stan_model(m)(; N = 1, K = 2, alpha = [1.0, 1.0])
    names = stan_unc_names(rebound)
    @test filter(n -> startswith(n, "as."), names) == ["as.1.1"]
    @test filter(n -> startswith(n, "La."), names) == ["La.1.1"]
    @test filter(n -> startswith(n, "Sg."), names) == ["Sg.1", "Sg.2", "Sg.3"]

    # A program without parameters has no names (BridgeStan lists `[""]`).
    none = @slic (; y = 0.5) begin
        y ~ normal(0.0, 1.0)
    end
    @test stan_unc_names(none) == String[]
    @test stan_interface(none)["unconstrained"] == String[]
end

@testitem "slic: stan_unc_names equals BridgeStan's param_unc_names" tags=[:slic, :descriptor, :interface, :ragged, :bridgestan] setup=[StanBlocksImports, UncNamesModel] begin
    bridgestan_names(m) = BridgeStan.param_unc_names(
        BridgeStan.StanModel(stan_library(stan_code(m)), stan_data_json(m)))

    m = unc_names_model()
    @test stan_unc_names(m) == bridgestan_names(m)
    rebound = StanBlocks.stan_model(m)(; N = 1, K = 2, alpha = [1.0, 1.0])
    @test stan_unc_names(rebound) == bridgestan_names(rebound)

    # Parameters the compiler declares: a ragged simplex's free coordinates, a
    # stick-frame simplex under a sampled concentration, and a plate cell's
    # vectors in flat memory.
    emitted = @slic (; Ks = [2, 3], K = 4, Kc = [1, 3], y = [1, 2, 2, 3], w = 0.2) begin
        p::simplex[Ks] ~ flat()
        alpha ~ lognormal(0, 1)
        beta::simplex[K] ~ dirichlet(rep_vector(1.0, K))
        theta::simplex[K] ~ dirichlet(alpha * K * beta)
        y ~ categorical(theta)
        r ~ plate(; outer = length(Kc)) do g
            z::vector[Kc[g]] ~ std_normal()
            z
        end
        w ~ normal(sum(p[1]) + sum(r[2]), 1.0)
    end
    @test stan_unc_names(emitted) == bridgestan_names(emitted)
    @test stan_interface(emitted)["unconstrained"] == bridgestan_names(emitted)
    regrouped = StanBlocks.stan_model(emitted)(; Ks = [3, 1, 2], Kc = [2, 2, 2])
    @test stan_unc_names(regrouped) == bridgestan_names(regrouped)
end
