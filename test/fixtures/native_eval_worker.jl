# Two content-equal `stan_instantiate` calls return separate instances of one
# published library. Evaluate both on every thread at once, comparing each
# result bit for bit with a serial evaluation. Run with `--threads` above one.
using StanBlocks, BridgeStan, LogDensityProblems
ENV["STANBLOCKS_BUILD_DIR"] = ARGS[1]
seconds = parse(Float64, ARGS[2])
Threads.nthreads() > 1 || error("native_eval_worker needs more than one thread")
K, N = 8, 400
g = [mod1(i, K) for i in 1:N]
y = [sin(i) for i in 1:N]
model = stan_model(@slic (; y, g, K) begin
    mu ~ normal(0.0, 1.0)
    tau ~ lognormal(0.0, 1.0)
    a ~ normal(mu, tau; n = K)
    y ~ normal(a[g], 1.0)
end)
one, two = fetch.([Threads.@spawn(stan_instantiate(model)) for _ in 1:2])
@assert one.model.stanmodel != two.model.stanmodel
threaded = occursin("STAN_THREADS=true", BridgeStan.model_info(one.model))
dim = LogDensityProblems.dimension(one)
points = [[0.1 * sin(k * j) for j in 1:dim] for k in 1:8]
reference = [LogDensityProblems.logdensity_and_gradient(one, x) for x in points]
evals, mismatches = Threads.Atomic{Int}(0), Threads.Atomic{Int}(0)
deadline = time() + seconds
workers = map(1:2Threads.nthreads()) do j
    Threads.@spawn begin
        problem = isodd(j) ? one : two
        k = 0
        while time() < deadline
            k += 1
            i = mod1(j + k, length(points))
            lp, grad = LogDensityProblems.logdensity_and_gradient(problem, points[i])
            lp === reference[i][1] && grad == reference[i][2] || Threads.atomic_add!(mismatches, 1)
            Threads.atomic_add!(evals, 1)
        end
    end
end
foreach(wait, workers)
println("RESULT evals=$(evals[]) mismatches=$(mismatches[]) threads=$(Threads.nthreads()) threaded=$threaded")
