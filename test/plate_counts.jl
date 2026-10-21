"""
A plate count read off a container is spelled as data (snag
`per-cell-scan-le-8b202289`). Without `outer=`, a do-form plate counts its first
positional; `eachindex(x)`, a container clause `@plate for v in x` and an
explicit `outer=length(x)` count `x` too. Every top-level declaration and
transformed-data index table the plate emits is sized by that count, and Stan
admits only data there. For a parameter `lev::vector[n]` the count used to be
`num_elements(lev)` — a non-data size in `parameters`, out of scope in
`transformed data` — so the documented spelling transpiled and stanc rejected it.
It is now `lev`'s declared count `n`: the program is the one `outer=(n,)` emits.
"""
@testitem "slic: a plate counted off a parameter positional sizes its storage with data" tags=[:slic, :plate, :scan, :regression, :stanc] setup=[StanBlocksImports, StanBlocksTestSetup] begin
    using .StanBlocksTestSetup: stanc_compiles
    ys = [[0.12, 0.31, 0.2], [-0.05, 0.1, 0.42], [0.3, -0.2, 0.05]]
    y3 = [0.1, 0.2, 0.3]
    same(a, b) = stan_code(a) == stan_code(b)

    # The reported shapes: a per-cell-length scan, latent and observed, in a
    # do-form plate whose first positional is a parameter.
    latent = @slic (; ys, n=3) begin
        phi ~ uniform(-1, 1); sigma ~ exponential(1); lev ~ normal(0, 1; n=n)
        res ~ plate(lev, ys) do l, y
            @scan begin
                h[1] ~ normal(l, 1)
                for t in 2:length(y)
                    h[t] ~ normal(phi * h[t-1], sigma)
                end
            end
            y ~ normal(h, 1)
            nothing
        end
    end
    latent_n = @slic (; ys, n=3) begin
        phi ~ uniform(-1, 1); sigma ~ exponential(1); lev ~ normal(0, 1; n=n)
        res ~ plate(lev, ys; outer=(n,)) do l, y
            @scan begin
                h[1] ~ normal(l, 1)
                for t in 2:length(y)
                    h[t] ~ normal(phi * h[t-1], sigma)
                end
            end
            y ~ normal(h, 1)
            nothing
        end
    end
    @test !occursin("num_elements(lev)", stan_code(latent))
    @test same(latent, latent_n)
    @test stanc_compiles(latent)

    observed = @slic (; ys, n=3) begin
        phi ~ uniform(-1, 1); sigma ~ exponential(1); lev ~ normal(0, 1; n=n)
        res ~ plate(lev, ys) do l, y
            @scan begin
                y[1] ~ normal(l, sigma)
                for t in 2:length(y)
                    y[t] ~ normal(l + phi * (y[t-1] - l), sigma)
                end
            end
        end
    end
    observed_n = @slic (; ys, n=3) begin
        phi ~ uniform(-1, 1); sigma ~ exponential(1); lev ~ normal(0, 1; n=n)
        res ~ plate(lev, ys; outer=(n,)) do l, y
            @scan begin
                y[1] ~ normal(l, sigma)
                for t in 2:length(y)
                    y[t] ~ normal(l + phi * (y[t-1] - l), sigma)
                end
            end
        end
    end
    @test !occursin("num_elements(lev)", stan_code(observed))
    @test same(observed, observed_n)
    @test stanc_compiles(observed)

    # No scan needed: a fresh cell parameter is declared in `parameters`.
    dense = @slic (; y3, n=3) begin
        lev ~ normal(0, 1; n=n)
        res ~ plate(lev, y3) do l, yi
            t ~ normal(l, 1); yi ~ normal(t, 1); t
        end
    end
    dense_n = @slic (; y3, n=3) begin
        lev ~ normal(0, 1; n=n)
        res ~ plate(lev, y3; outer=n) do l, yi
            t ~ normal(l, 1); yi ~ normal(t, 1); t
        end
    end
    @test same(dense, dense_n)
    @test stanc_compiles(dense)

    # Only parameter positionals, and a transformed parameter: the declared count.
    only_param = @slic (; y3, n=3) begin
        lev ~ normal(0, 1; n=n)
        res ~ plate(lev) do l
            t ~ normal(l, 1); t
        end
        y3 ~ normal(res, 1)
    end
    only_param_n = @slic (; y3, n=3) begin
        lev ~ normal(0, 1; n=n)
        res ~ plate(lev; outer=n) do l
            t ~ normal(l, 1); t
        end
        y3 ~ normal(res, 1)
    end
    @test same(only_param, only_param_n)
    @test stanc_compiles(only_param)
    explicit = @slic (; y3, n=3) begin
        lev ~ normal(0, 1; n=n)
        res ~ plate(lev; outer=length(lev)) do l
            t ~ normal(l, 1); t
        end
        y3 ~ normal(res, 1)
    end
    @test same(explicit, only_param_n)
    transformed = @slic (; y3, n=3) begin
        lev ~ normal(0, 1; n=n)
        mu = 2.0 * lev
        res ~ plate(mu, y3) do m, yi
            t ~ normal(m, 1); yi ~ normal(t, 1); t
        end
    end
    @test !occursin("num_elements(mu)", stan_code(transformed))
    @test stanc_compiles(transformed)

    # The annotated loops: `eachindex(lev)`, a container clause, `2:length(lev)`.
    each = @slic (; y3, n=3) begin
        lev ~ normal(0, 1; n=n)
        @plate for i in eachindex(lev)
            x[i] ~ normal(lev[i], 1)
        end
        y3 ~ normal(x, 1)
    end
    each_n = @slic (; y3, n=3) begin
        lev ~ normal(0, 1; n=n)
        @plate for i in 1:n
            x[i] ~ normal(lev[i], 1)
        end
        y3 ~ normal(x, 1)
    end
    @test same(each, each_n)
    @test stanc_compiles(each)
    container = @slic (; n=3) begin
        lev ~ normal(0, 1; n=n)
        @plate for l in lev
            t ~ normal(l, 1)
        end
    end
    @test !occursin("num_elements(lev)", stan_code(container))
    @test stanc_compiles(container)
    scan = @slic (; y3, n=3) begin
        lev ~ normal(0, 1; n=n)
        @scan begin
            h[1] ~ normal(lev[1], 1)
            for t in 2:length(lev)
                h[t] ~ normal(h[t-1] + lev[t], 1)
            end
        end
        y3 ~ normal(h, 1)
    end
    scan_n = @slic (; y3, n=3) begin
        lev ~ normal(0, 1; n=n)
        @scan begin
            h[1] ~ normal(lev[1], 1)
            for t in 2:n
                h[t] ~ normal(h[t-1] + lev[t], 1)
            end
        end
        y3 ~ normal(h, 1)
    end
    @test same(scan, scan_n)
    @test stanc_compiles(scan)

    # A data positional keeps its own `length` spelling.
    data_first = @slic (; y3) begin
        res ~ plate(y3) do yi
            t ~ normal(0, 1); yi ~ normal(t, 1); t
        end
    end
    @test occursin("num_elements(y3)", stan_code(data_first))
    @test stanc_compiles(data_first)
end
