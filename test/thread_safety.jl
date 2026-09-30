using TestItemRunner

@testitem "thread safety: named submodels own their namespace" tags=[:slic, :regression] begin
    using StanBlocks

    function workspace(offset)
        # Same names even for the modules: name-based tokens cannot isolate this.
        mod = Module(:RepeatedWorkspace)
        Core.eval(mod, :(using StanBlocks))
        Core.eval(mod, quote
            @slic latent(scale) = begin
                z ~ normal($offset, scale)
                return z
            end
        end)
        mod
    end
    a, b = workspace(1.25), workspace(2.75)
    @test typeof(a.latent) !== typeof(b.latent)
    @test Base.invokelatest(a.latent, 1.0).mod === a
    @test Base.invokelatest(b.latent, 1.0).mod === b
    model(mod) = Core.eval(mod, :(@slic (; y=0.0) begin
        mu ~ latent(1.0)
        y ~ normal(mu, 1.0)
    end))
    code_a = Base.invokelatest(stan_code, model(a))
    code_b = Base.invokelatest(stan_code, model(b))
    @test occursin("1.25", code_a) && !occursin("2.75", code_a)
    @test occursin("2.75", code_b) && !occursin("1.25", code_b)
    # Adding a method in one module preserves its existing binding and the other.
    binding = a.latent
    Core.eval(a, :(@slic latent(scale, offset) = begin
        z ~ normal(offset, scale)
        return z
    end))
    @test a.latent === binding
    @test Base.invokelatest(a.latent, 1.0, 0.5).mod === a
    @test Base.invokelatest(stan_code, model(b)) == code_b
end

@testitem "thread safety: concurrent runtime families and old tasks" tags=[:slic, :regression] begin
    using StanBlocks
    @test Threads.nthreads() >= 2
    data = (; y=[0.25, -0.5])
    before = deepcopy(data)
    function bundle(offset)
        definition = :(shift(x::real)::real = begin
            x + $offset
        end)
        compile_slic_bundle(data, [:(latent() = begin
            z ~ normal(shift(0.0), 1.0)
            return z
        end)], quote
            mu ~ latent()
            y ~ normal(mu, 1.0)
        end; udf_definitions=[(; definition, markers=:juliacompat)])
    end
    # Distinct definitions share the same Julia-signature registry, while each
    # private workspace must retain its own helper and named submodel semantics.
    expected = [bundle(i + 0.125).code for i in 1:8]
    actual = fetch.([Threads.@spawn(bundle(i + 0.125).code) for i in 1:8])
    @test actual == expected
    @test data == before

    mod = Module(:OldTaskDefinitions)
    Core.eval(mod, :(using StanBlocks))
    ready = Channel{Nothing}(1)
    models = Channel{Any}(1)
    old_task = Threads.@spawn begin
        put!(ready, nothing)
        stan_code(take!(models))
    end
    take!(ready)
    slic = slic_eval(mod, quote
        @deffun @juliacompat offset(x::real)::real = x + 3.25
        @slic latent() = begin
            z ~ normal(offset(0.0), 1.0)
            return z
        end
        @slic (; y=0.0) begin
            mu ~ latent()
            y ~ normal(mu, 1.0)
        end
    end)
    put!(models, slic)
    @test occursin("3.25", fetch(old_task))
    @test Base.invokelatest(mod.offset, 2.0) == 5.25
    @test stan_code(slic) == stan_code(stan_model(slic))

    failed = Module(:FailedDefinitions)
    Core.eval(failed, :(using StanBlocks))
    @test_throws Exception slic_eval(failed, quote
        @deffun begin
            @juliacompat collision(x::vector[n])::vector[n] = x
            @juliacompat collision(x::row_vector[n])::row_vector[n] = x
        end
    end)
    # A failed definition must release the gate and leave other namespaces usable.
    @test bundle(1.125).code == expected[1]
end

@testitem "thread safety: parallel readers and exclusive publication" tags=[:slic, :regression] begin
    using StanBlocks
    @test Threads.nthreads() >= 2
    entered, release = Channel{Nothing}(2), Channel{Nothing}(2)
    readers = [Threads.@spawn(StanBlocks._with_slic_read() do
        put!(entered, nothing)
        take!(release)
        # A waiting writer cannot deadlock an already admitted nested reader.
        StanBlocks._with_slic_read(() -> :read)
    end) for _ in 1:2]
    take!(entered)
    take!(entered) # Both readers must enter BEFORE either one is released.
    writer_started, writer_entered = Channel{Nothing}(1), Channel{Nothing}(1)
    writer = Threads.@spawn begin
        put!(writer_started, nothing)
        StanBlocks._with_slic_definitions() do
            put!(writer_entered, nothing)
            StanBlocks._with_slic_definitions() do
                StanBlocks._with_slic_read(() -> :write)
            end
        end
    end
    take!(writer_started)
    @test !isready(writer_entered)
    put!(release, nothing)
    put!(release, nothing)
    @test fetch.(readers) == [:read, :read]
    @test fetch(writer) == :write
    @test isready(writer_entered)
    @test_throws ErrorException StanBlocks._with_slic_read(() -> error("reader"))
    @test_throws ErrorException StanBlocks._with_slic_definitions(() -> error("writer"))
    @test_throws ArgumentError StanBlocks._with_slic_read() do
        StanBlocks._with_slic_definitions(() -> nothing)
    end
    @test StanBlocks._with_slic_read(() -> :released) == :released
end
