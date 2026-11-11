module DocsExamples
using StanBlocks, Distributions, LogDensityProblems, AlgebraOfVega
import HTMXObjects # Loads AlgebraOfVega's captioned, theme-aware HTML extension.
using BridgeStan, Markdown, SHA, TOML, Statistics, Printf

const ROOT = dirname(@__DIR__)
const MODELS = Dict{String,Module}()
const RESULTS = Dict{String,Any}()
const FIGURES = Dict{String,String}()
const RECEIPTS = Dict{String,Dict{String,Any}}()
const BLUE = "#2a78d6"
const ORANGE = "#eb6834"

function example_module(name)
    get!(MODELS, name) do
        mod = Module(Symbol("Docs_", name))
        Base.include(mod, joinpath(ROOT, "research", name, "model.jl"))
        mod
    end
end

function model_block(name)
    mod = example_module(name)
    model = getfield(mod, :model)
    code = Base.invokelatest(stan_code, model)
    checked = stanc_check(code)
    checked.ok || error("Stan compilation failed for $name:\n$(checked.output)")
    source = read(joinpath(ROOT, "research", name, "model.jl"), String)
    Markdown.parse("```julia\n" * source * "\n```\n\n### Generated Stan\n\n```stan\n" * code * "\n```")
end

function table(headers, rows)
    Markdown.parse("| " * join(headers, " | ") * " |\n| " *
        join(fill("---", length(headers)), " | ") * " |\n" *
        join(["| " * join(string.(row), " | ") * " |" for row in rows], "\n"))
end
number(x; digits=3) = string(round(x; digits))

function golf()
    get!(RESULTS, "golf") do
        dir = joinpath(ROOT, "research", "golf")
        receipt = TOML.parsefile(joinpath(dir, "receipt.toml"))
        summary_path = joinpath(dir, "summary.toml")
        key = bytes2hex(sha256(vcat(read(joinpath(dir, "model.jl")), read(joinpath(dir, "fit.jl")))))
        @assert receipt["key"] == key "Golf input/generator changed: regenerate its summary"
        @assert receipt["summary_sha256"] == bytes2hex(sha256(read(summary_path)))
        mod = example_module("golf")
        code = Base.invokelatest(stan_code, mod.model)
        @assert receipt["stan_source_sha256"] == bytes2hex(sha256(code)) "Golf Stan program changed: regenerate its summary"
        (; summary=TOML.parsefile(summary_path), receipt, mod)
    end
end

function golf_data()
    g = golf().mod.golf_data
    table(["Distances", "Attempts", "Successful putts", "Distance range (ft)"],
          [(length(g.distance), sum(g.attempts), sum(g.sunk), "$(minimum(g.distance))–$(maximum(g.distance))")])
end
function golf_summary()
    values = golf().summary["sigma_deg"]
    table(["Parameter", "Median", "Central 95% interval"],
          [("Angular SD (degrees)", number(values[2]), "$(number(values[1]))–$(number(values[3]))")])
end
function golf_diagnostics()
    r = golf().receipt
    table(["Check", "Result"], [
        ("Grid / refined grid (points)", "$(r["grid_points"]) / $(r["refined_grid_points"])"),
        ("Median change on refinement (rad)", @sprintf("%.3g", r["median_refinement_rad"])),
        ("Range of Stan − reference log density", @sprintf("%.3g", r["density_difference_range"])),
        ("Grid boundary probability mass", @sprintf("%.3g", r["boundary_mass"])),
    ])
end

function pooling()
    get!(RESULTS, "pooling") do
        mod = example_module("pooling")
        d = mod.school_data
        variance = 1 ./ (1 ./ d.standard_error.^2 .+ 1/d.population_sd^2)
        mean = variance .* (d.effect ./ d.standard_error.^2 .+ d.population_mean/d.population_sd^2)
        sd = sqrt.(variance)
        weight = variance ./ d.standard_error.^2
        problem = Base.invokelatest(stan_instantiate, mod.model; nan_on_error=false)
        @assert LogDensityProblems.dimension(problem) == length(d.effect)
        reference(x) = sum(logpdf.(Normal(d.population_mean, d.population_sd), x)) +
            sum(logpdf.(Normal.(x, d.standard_error), d.effect))
        points = [zeros(length(mean)), mean, mean .+ sd]
        deltas = [LogDensityProblems.logdensity(problem, x) - reference(x) for x in points]
        error = maximum(abs.(deltas .- first(deltas)))
        @assert error < 1e-8
        (; mod, mean, sd, weight, density_error=error)
    end
end
function pooling_summary()
    p = pooling()
    table(["School", "Observed effect (points)", "SE (points)", "Data weight", "Conditional mean (points)", "Conditional SD (points)"],
        [(i, p.mod.school_data.effect[i], p.mod.school_data.standard_error[i],
          number(p.weight[i]), number(p.mean[i]), number(p.sd[i])) for i in eachindex(p.mean)])
end
function pooling_diagnostics()
    p=pooling()
    table(["Check", "Result"], [
        ("Stan − reference density range", @sprintf("%.3g", p.density_error)),
        ("All weights between zero and one", all(0 .< p.weight .< 1)),
        ("All conditional SDs below observation SEs", all(p.sd .< p.mod.school_data.standard_error)),
    ])
end

function epidemic()
    get!(RESULTS, "epidemic") do
        mod = example_module("epidemic")
        problem = Base.invokelatest(stan_instantiate, mod.model; nan_on_error=false)
        @assert LogDensityProblems.dimension(problem) == 0
        names = BridgeStan.param_names(problem.model; include_tp=true, include_gq=true)
        values = BridgeStan.param_constrain(problem.model, Float64[]; include_tp=true, include_gq=true,
            rng=BridgeStan.StanRNG(problem.model, 1))
        descriptor = Base.invokelatest(stan_descriptor, mod.model)
        output = only(o for o in descriptor.outputs if o.name == :trajectory)
        trajectory = fill(NaN, length(mod.sir_data.times), 3)
        for (name, value) in zip(names, values)
            matched = match(r"^trajectory\.(\d+)\.(\d+)$", name)
            isnothing(matched) && continue
            trajectory[parse(Int,matched[1]),parse(Int,matched[2])] = value
        end
        @assert all(isfinite,trajectory)
        infection_rate = values[only(findall(==("infection_rate"),names))]
        removal_rate = values[only(findall(==("removal_rate"),names))]
        population_error = maximum(abs.(sum(trajectory; dims=2) .- mod.sir_data.population))
        @assert population_error < 1e-4
        @assert minimum(trajectory) > -1e-6
        @assert all(diff(trajectory[:,1]) .<= 1e-6)
        initial = mod.sir_data.initial
        incidence = -diff(vcat(initial[1], trajectory[:,1]))
        (; mod, trajectory, incidence, population_error, output, infection_rate, removal_rate)
    end
end
function epidemic_summary()
    e=epidemic()
    peak=argmax(e.trajectory[:,2])
    table(["Quantity", "Value"], [
        ("Infection rate (per day)", number(e.infection_rate)),
        ("Removal rate (per day)", number(e.removal_rate)),
        ("Reproduction number at full susceptibility", number(e.infection_rate/e.removal_rate)),
        ("Mean infectious period (days)", number(1/e.removal_rate)),
        ("Largest sampled prevalence (people)", number(e.trajectory[peak,2]; digits=1)),
        ("Time of largest sampled prevalence (days)", e.mod.sir_data.times[peak]),
    ])
end
function epidemic_diagnostics()
    e=epidemic()
    table(["Check", "Result"], [
        ("Maximum population residual (people)", @sprintf("%.3g", e.population_error)),
        ("Minimum compartment count (people)", number(minimum(e.trajectory); digits=6)),
        ("Susceptible count never increases", all(diff(e.trajectory[:,1]) .<= 1e-6)),
    ])
end

function figure(name)
    if name == "golf"
        rows = golf().summary["predictions"]
        data_rows = [(distance=r["distance_ft"], observed=r["observed"],
                      prediction=r["probability_median"], lo=r["replicate_lower"], hi=r["replicate_upper"]) for r in rows]
        points = data(data_rows) * mapping(:distance => "Distance (ft)", :observed => "Success proportion") * visual(Scatter; color=ORANGE)
        line = data(data_rows) * mapping(:distance => "Distance (ft)", :prediction => "Success proportion") * visual(Lines; color=BLUE)
        interval = data(data_rows) * mapping(:distance => "Distance (ft)", :lo => "Success proportion", :hi) * visual(Rangebars; color=BLUE)
        spec = (interval + line + points) * config(width=560, height=300)
        caption = "Observed success proportions (orange), posterior median success probability (blue), and central 95% intervals for replicate proportions at the observed attempt counts. Systematic deviations assess the angle-only assumption."
    elseif name == "pooling"
        p=pooling()
        z=quantile(Normal(), 0.975)
        rows = [(school=i, estimate=p.mean[i], lower=p.mean[i]-z*p.sd[i], upper=p.mean[i]+z*p.sd[i], observed=p.mod.school_data.effect[i]) for i in eachindex(p.mean)]
        interval = data(rows) * mapping(:school => "School", :lower => "Treatment effect (points)", :upper) * visual(Rangebars; color=BLUE)
        estimates = data(rows) * mapping(:school => "School", :estimate => "Treatment effect (points)") * visual(Scatter; color=BLUE)
        observations = data(rows) * mapping(:school => "School", :observed => "Treatment effect (points)") * visual(Scatter; color=ORANGE)
        spec = (interval + estimates + observations) * config(width=560, height=300)
        caption = "Observed effects (orange) and conditional posterior means with central 95% intervals (blue). The population mean and standard deviation are fixed inputs; these intervals exclude uncertainty in those inputs."
    elseif name == "epidemic"
        e=epidemic()
        rows = [(day=e.mod.sir_data.times[i], infectious=e.trajectory[i,2], incidence=e.incidence[i]) for i in eachindex(e.incidence)]
        prevalence = data(rows) * mapping(:day => "Time (days)", :infectious => "People") * visual(Lines; color=BLUE)
        incidence = data(rows) * mapping(:day => "Time (days)", :incidence => "People") * visual(Lines; color=ORANGE)
        spec = (prevalence + incidence) * config(width=560, height=300)
        step=first(diff(e.mod.sir_data.times))
        caption = "Infectious prevalence at each time (blue) and new infections during the preceding $(step)-day interval (orange). This is one trajectory conditional on a seeded draw of the rates from their stated priors; it is not a fit or an uncertainty band."
    else
        error("Unknown figure $name")
    end
    FIGURES[name] = to_html(spec, caption; plot_id="stanblocks-$name", source=:vendor, base="../vendor", runtime=:linked)
    nothing
end

function reproduce(name)
    path=joinpath(ROOT,"research",name,"model.jl")
    hash=bytes2hex(sha256(read(path)))
    if name != "golf"
        mod = example_module(name)
        checks = name == "pooling" ? Dict("density_difference_range" => pooling().density_error) :
            Dict("maximum_population_residual_people" => epidemic().population_error)
        RECEIPTS[name] = merge(Dict{String,Any}("method" => "documentation build calculation",
            "julia" => string(VERSION), "stanblocks_version" => string(pkgversion(StanBlocks)),
            "model_source_sha256" => hash,
            "stan_source_sha256" => bytes2hex(sha256(Base.invokelatest(stan_code, mod.model)))), checks)
    end
    receipt = name == "golf" ? "Fit key: `$(golf().receipt["key"])`. Created: $(golf().receipt["created_utc"]) UTC." :
        "Build calculation; no fitted parameters or cached draws. Source SHA-256: `$hash`. [Numerical receipt](../research/$name/receipt.toml.txt)."
    # Reproduction is the only collapsed material; scientific results remain in the page.
    Markdown.parse("$receipt\n\n" *
        "[Source](../research/$name/model.jl.txt) · [Instructions](../research/$name/README.txt)\n\n" *
        (name=="golf" ? "[Fit summary](../research/golf/summary.toml.txt) · [Receipt](../research/golf/receipt.toml.txt) · [Generator](../research/golf/fit.jl.txt)\n\n" : "") *
        "The shell command below reproduces this result from the repository root. " *
        (name=="golf" ? "It reruns the fit and is intentionally excluded from CI.\n\n```sh\njulia --project=docs research/golf/fit.jl\n```" : "It runs the documentation examples and regenerates the figure.\n\n```sh\njulia --project=docs docs/make.jl\n```"))
end

function publish_assets(public)
    mkpath(joinpath(public,"figures"))
    for (name, html) in FIGURES
        write(joinpath(public,"figures","$name.html"),html)
    end
    cp(vega_vendor_dir(), joinpath(public,"vendor"); force=true)
    for name in keys(MODELS)
        source=joinpath(ROOT,"research",name)
        target=joinpath(public,"research",name)
        mkpath(target)
        for file in readdir(source)
            dest=file=="README.md" ? "README.txt" : file * ".txt"
            cp(joinpath(source,file),joinpath(target,dest);force=true)
        end
        if haskey(RECEIPTS,name)
            open(joinpath(target,"receipt.toml.txt"),"w") do io
                TOML.print(io, RECEIPTS[name])
            end
        end
    end
end
end
