module PosteriorDBWeb

using HTMXObjects
using DynamicObjects
import PosteriorDB
using StanBlocks
import StanBlocks: slic_implementation
import StanBlocks.stan: transpiles, compiles, stan_code, stan_model, instantiate
using LogDensityProblems
using Statistics, Random
using BridgeStan, StanLogDensityProblems, JSON
using TestModules
using Treebars: prepare_progress!, with_prepared_progress, polling_fetchindex

# include("test/runtests.jl")

pdb() = PosteriorDB.database()


# --- Test helpers ---

function try_transpile(post)
    # try
        code = stan_code(post)
        (ok=true, error=nothing, stacktrace=nothing, code=code)
    # catch e
    #     error()
    #     bt = catch_backtrace()
    #     wrapped = e isa StanBlocksError ? e : StanBlocksError(:transpile, string(post), (e, bt))
    #     error_msg = first(split(sprint(showerror, wrapped), "\n"))
    #     full_trace = sprint(showerror, wrapped)
    #     (ok=false, error=error_msg, stacktrace=full_trace, code=nothing)
    # end
end

function try_compile(post)
    # try
        problem = instantiate(post)
        (ok=true, error=nothing, stacktrace=nothing, dimension=LogDensityProblems.dimension(problem))
    # catch e
    #     error()
    #     bt = catch_backtrace()
    #     wrapped = e isa StanBlocksError ? e : StanBlocksError(:compile, string(post), (e, bt))
    #     error_msg = first(split(sprint(showerror, wrapped), "\n"))
    #     full_trace = sprint(showerror, wrapped)
    #     (ok=false, error=error_msg, stacktrace=full_trace, dimension=nothing)
    # end
end

function compare_logdensities(reference, test; stat_f=median, m=200)
    n = LogDensityProblems.dimension(test)
    X = randn((n, m))
    eval_safe(problem, x) = LogDensityProblems.logdensity(problem, x) # try ... catch; NaN end
    reference_lpdfs = [eval_safe(reference, x) for x in eachcol(X)]
    test_lpdfs = [eval_safe(test, x) for x in eachcol(X)]
    finite_idxs = filter(i -> isfinite(reference_lpdfs[i] + test_lpdfs[i]), 1:m)
    length(finite_idxs) == 0 && return nothing
    ref = reference_lpdfs[finite_idxs]
    tst = test_lpdfs[finite_idxs]
    adjusted = tst .+ median(ref - tst)
    (;
        absolute_constant_difference=stat_f(abs.(ref .- tst)),
        relative_remaining_difference=stat_f(abs.((ref .- adjusted) ./ ref))
    )
end

function try_correct(posterior)
    # try
        post = slic_implementation(posterior)
        problem = instantiate(post)
        ref_stan_path = PosteriorDB.path(PosteriorDB.implementation(PosteriorDB.model(posterior), "stan"))
        ref_data = PosteriorDB.load(PosteriorDB.dataset(posterior), String)
        ref_problem = StanLogDensityProblems.StanProblem(ref_stan_path, ref_data; nan_on_error=true, make_args=["STAN_THREADS=true"])
        result = compare_logdensities(ref_problem, problem)
        isnothing(result) && return (ok=false, error="No finite evaluations", stacktrace=nothing, stats=nothing)
        ok = result.relative_remaining_difference < 1e-6
        (ok=ok, error=ok ? nothing : "Relative difference: $(result.relative_remaining_difference)", stacktrace=nothing, stats=result)
    # catch e
    #     error()
    #     bt = catch_backtrace()
    #     wrapped = e isa StanBlocksError ? e : StanBlocksError(:evaluate, string(posterior), (e, bt))
    #     error_msg = first(split(sprint(showerror, wrapped), "\n"))
    #     full_trace = sprint(showerror, wrapped)
    #     (ok=false, error=error_msg, stacktrace=full_trace, stats=nothing)
    # end
end

# --- Web app ---

function status_str(cached, ok)
    !cached ? "-" : ok ? "PASS" : "FAIL"
end

function status_cell(cached, ok)
    !cached && return h.td("-"; style="color:gray")
    ok ? h.td("PASS"; style="color:green;font-weight:bold") :
         h.td("FAIL"; style="color:red;font-weight:bold")
end

function status_cell_clickable(cached, ok, check_url, detail_id)
    base_style = "cursor:pointer;"
    if !cached
        return h.td("-"; class="check-cell", style=base_style * "color:gray",
            hx_get=check_url, hx_target="#$detail_id", hx_swap="innerHTML",
            _="on htmx:afterOnLoad if not me.classList.contains('batch') then remove .hidden from #$detail_id end remove .batch from me")
    elseif !ok
        return h.td("FAIL"; class="check-cell", style=base_style * "color:red;font-weight:bold",
            hx_get=check_url, hx_target="#$detail_id", hx_swap="innerHTML",
            _="on htmx:afterOnLoad if not me.classList.contains('batch') then remove .hidden from #$detail_id end remove .batch from me")
    else
        return h.td("PASS"; class="check-cell", style=base_style * "color:green;font-weight:bold",
            _="on click toggle .hidden on #$detail_id")
    end
end

function split_posterior_name(pn)
    parts = split(pn, "-"; limit=2)
    length(parts) == 2 ? (parts[1], parts[2]) : (pn, "")
end


# --- Sandbox gallery (read-only view of `web/sandbox/`) ----------------------
#
# Reuses the existing on-disk sandbox dir as a read-only `HTMXObjects.Gallery`.
# The interactive `/sandbox` editor is left untouched — the gallery exposes
# the *same files* via the new `gallery_grid` primitive so the docs
# `record_gallery` flow can ship the entire sandbox as static recordings.
const _SANDBOX_GALLERY_DIR = joinpath(dirname(dirname(@__DIR__)), "web", "sandbox")
const _sandbox_gallery = Gallery(_SANDBOX_GALLERY_DIR)

# `cache_type=:parallel` so the IP cache survives across the per-request
# `@htmx` instances polling for `record_gallery` progress (see AoV's
# `GalleryAppData` for the canonical shape).
@dynamicstruct struct SbAppData
    """
    `record_gallery(record_dir, record_base)` — IP. Drives
    `HTMXObjects.record!` against a fresh `AppContext()` to dump every
    sandbox-gallery URL into `record_dir`, with per-path `prepare_progress!`
    markers so the route can poll a live progress tree. Returns a
    NamedTuple summary `(; n_html, n_js, n_json, n_other, n_paths,
    record_dir, record_base)`.
    """
    record_gallery(record_dir::String, record_base::String) = begin
        ids = [it.id for it in _sandbox_gallery.items]
        paths = vcat(
            ["/gallery"],
            ["/sandbox_view/$id" for id in ids],
        )

        phases = [prepare_progress!(__status__; description=p) for p in paths]

        isdir(record_dir) && rm(record_dir; recursive=true)
        mkpath(record_dir)

        # `HTMXObjects.record!` re-`route!`s the app with recording closures.
        # The live AppContext registration is clobbered while this loop runs;
        # restore it via `route!(app)` in the `finally` so subsequent live
        # requests keep going to the editor / live UI.
        app = AppContext()
        HTMXObjects.route!(app; record_dir, record_base)
        router = HTMXObjects.CONTEXT[].service.router
        try
            for (path, phase) in zip(paths, phases)
                with_prepared_progress(phase) do _
                    HTMXObjects._drive_record_path(router, path, Pair{String,String}[])
                    HTMXObjects._drive_record_path(router, path, ["HX-Request" => "true"])
                end
            end
        finally
            HTMXObjects.route!(app)
        end

        n_html = 0; n_js = 0; n_json = 0; n_other = 0
        for (root, _, fs) in walkdir(record_dir)
            for f in fs
                ext = lowercase(splitext(f)[2])
                ext == ".html" ? (n_html += 1) :
                ext == ".js"   ? (n_js   += 1) :
                ext == ".json" ? (n_json += 1) :
                                 (n_other += 1)
            end
        end
        (; n_html, n_js, n_json, n_other, n_paths=length(paths), record_dir, record_base)
    end
end

const APPDATA = SbAppData(; cache_type=:parallel)


@htmx struct AppContext

    # Long-running operations (e.g. `record_gallery`) live in the parallel-
    # cached `APPDATA` so their per-request progress trees survive between
    # `polling_fetchindex` polls. Reach into `__appdata__.record_gallery`
    # directly at the call site rather than destructuring `(; record_gallery)
    # = __appdata__` — the destructuring binds `record_gallery` as a
    # property of `AppContext`, which collides with the `@get record_gallery`
    # handler below on `compute_property(AppContext, Val{:record_gallery})`
    # and routes the IP cache lookup to the wrong struct.
    __appdata__ = APPDATA

    cache_path = joinpath(dirname(dirname(@__DIR__)), "web", "cache")
    status_path = joinpath(dirname(dirname(@__DIR__)), "web", "status")

    status_file(check, pn) = joinpath(status_path, "$(check)_$(pn).status")
    write_status(check, pn, ok::Bool) = begin
        isdir(status_path) || mkpath(status_path)
        write(status_file[check, pn], ok ? "PASS" : "FAIL")
    end

    @cached posterior_names = sort([
        pn for pn in PosteriorDB.posterior_names(pdb())
        if !isnothing(slic_implementation(PosteriorDB.posterior(pdb(), pn)))
    ])

    @cached transpile_result(pn) = begin
        post = slic_implementation(PosteriorDB.posterior(pdb(), pn))
        r = try_transpile(post)
        write_status["transpile", pn, r.ok]
        r
    end

    @cached compile_result(pn) = begin
        post = slic_implementation(PosteriorDB.posterior(pdb(), pn))
        r = try_compile(post)
        write_status["compile", pn, r.ok]
        r
    end

    @cached correct_result(pn) = begin
        posterior = PosteriorDB.posterior(pdb(), pn)
        r = try_correct(posterior)
        write_status["correct", pn, r.ok]
        r
    end

    # Disk-only status read for the overview: avoids deserializing full result
    # NamedTuples (with stacktrace + code). Sidecars are written whenever a
    # `@cached` result body runs. For pre-existing `.sjl` caches with no sidecar,
    # call `/backfill_status` once to populate them.
    disk_status(check, pn) = begin
        sp = status_file[check, pn]
        isfile(sp) || return (false, false)
        (true, read(sp, String) == "PASS")
    end

    overview_row(pn) = begin
        local dataset, model
        dataset, model = split_posterior_name(pn)
        t_cached, t_ok = disk_status["transpile", pn]
        c_cached, c_ok = disk_status["compile", pn]
        r_cached, r_ok = disk_status["correct", pn]
        detail_id = "detail-$pn"
        toggle = "on click toggle .hidden on #$detail_id"
        [h.tr(
            h.td(pn; style="cursor:pointer", _=toggle),
            h.td(dataset; style="cursor:pointer", _=toggle),
            h.td(model; style="cursor:pointer", _=toggle),
            status_cell_clickable(t_cached, t_ok, "/check_transpile/$pn", detail_id),
            status_cell_clickable(c_cached, c_ok, "/check_compile/$pn", detail_id),
            status_cell_clickable(r_cached, r_ok, "/check_correct/$pn", detail_id);
            id="row-$pn",
        ),
        h.tr(; id=detail_id, class="hidden")]
    end

    @get index = h.div(
        h.h2("PosteriorDB Models ($(length(posterior_names)) implemented)"),
        h.input(;
            type="search",
            id="search",
            placeholder="Filter posteriors...",
            _="on input set query to my value.toLowerCase() for row in <tr/> in #posterior-tbody if row.textContent.toLowerCase() contains query remove .hidden from row else add .hidden to row end end on keydown[key is 'Enter'] halt the event for row in <tr[id^='row-']/> in #posterior-tbody if row matches ':not(.hidden)' set target to null for cell in <td.check-cell/> in row if target is null and cell.textContent.trim() is not 'PASS' set target to cell end end if target is not null add .batch to target send click to target end end end",
            style="margin-bottom:1rem;width:100%;",
        ),
        h.table(class="striped"; role="grid")(
            h.thead(
                h.tr(
                    h.th("Posterior"; _="on click call sortTable(0, me)", style="cursor:pointer"),
                    h.th("Dataset"; _="on click call sortTable(1, me)", style="cursor:pointer"),
                    h.th("Model"; _="on click call sortTable(2, me)", style="cursor:pointer"),
                    h.th("Transpiles"; _="on click call sortTable(3, me)", style="cursor:pointer"),
                    h.th("Compiles"; _="on click call sortTable(4, me)", style="cursor:pointer"),
                    h.th("Correct"; _="on click call sortTable(5, me)", style="cursor:pointer"),
                )
            ),
            h.tbody(vcat([overview_row[pn] for pn in posterior_names]...)...; id="posterior-tbody")
        ),
        h.style(".hidden { display: none; } tr[id^=row-]:hover { background: var(--pico-table-row-stripped-background-color); } details summary { cursor: pointer; font-weight: 600; margin-bottom: 0.5rem; }"),
    )

    result_section(label, result) = begin
        isnothing(result) && return ""
        h.div(; style="margin-bottom:0.5rem")(
            h.p(h.strong(label, ": "), status_badge(result.ok ? :done : :failed; label=result.ok ? "PASS" : "FAIL")),
            isnothing(result.error) ? "" : h.p(h.strong("Error: "), h.code(result.error)),
            isnothing(result.stacktrace) ? "" : h.details(
                h.summary("Full stacktrace"),
                h.pre(result.stacktrace; style="white-space:pre-wrap;max-height:300px;overflow:auto;font-size:0.8em")
            ),
            hasproperty(result, :code) && !isnothing(result.code) ? h.details(
                h.summary("Generated Stan code"),
                h.pre(result.code; style="white-space:pre-wrap;max-height:400px;overflow:auto;font-size:0.85em");
                open=""
            ) : "",
            hasproperty(result, :stats) && !isnothing(result.stats) ? h.p(
                h.strong("Stats: "),
                "abs diff = $(result.stats.absolute_constant_difference), rel diff = $(result.stats.relative_remaining_difference)"
            ) : "",
        )
    end

    model_detail_content(pn) = begin
        t_cached = @is_cached transpile_result[pn]
        c_cached = @is_cached compile_result[pn]
        r_cached = @is_cached correct_result[pn]
        t_result = t_cached ? transpile_result[pn] : nothing
        c_result = c_cached ? compile_result[pn] : nothing
        r_result = r_cached ? correct_result[pn] : nothing
        # Pick worst color for border
        all_ok = all(r -> isnothing(r) || r.ok, [t_result, c_result, r_result])
        any_fail = any(r -> !isnothing(r) && !r.ok, [t_result, c_result, r_result])
        border_color = any_fail ? "var(--pico-del-color)" : all_ok ? "var(--pico-ins-color)" : "var(--pico-muted-border-color)"
        h.td(; colspan="6", style="padding:0;border:none")(
            h.div(; style="padding:1rem 1.5rem;background:var(--pico-card-background-color);border-left:4px solid $border_color;margin:0.25rem 0")(
                h.h4(pn),
                result_section["Transpiles", t_result],
                result_section["Compiles", c_result],
                result_section["Correct", r_result],
            )
        )
    end

    plain_overview = begin
        header = rpad("Posterior", 60) * rpad("Transpiles", 12) * rpad("Compiles", 12) * "Correct"
        lines = [header, "-"^length(header)]
        for pn in posterior_names
            t_cached = @is_cached transpile_result[pn]
            t_ok = t_cached ? transpile_result[pn].ok : false
            c_cached = @is_cached compile_result[pn]
            c_ok = c_cached ? compile_result[pn].ok : false
            r_cached = @is_cached correct_result[pn]
            r_ok = r_cached ? correct_result[pn].ok : false
            push!(lines, rpad(pn, 60) * rpad(status_str(t_cached, t_ok), 12) * rpad(status_str(c_cached, c_ok), 12) * status_str(r_cached, r_ok))
        end
        join(lines, "\n")
    end

    plain_result_section(label, result) = begin
        isnothing(result) && return "$label: -"
        parts = ["$label: $(result.ok ? "PASS" : "FAIL")"]
        isnothing(result.error) || push!(parts, "  Error: $(result.error)")
        hasproperty(result, :stacktrace) && !isnothing(result.stacktrace) && push!(parts, "", "  --- Stacktrace ---", result.stacktrace)
        hasproperty(result, :code) && !isnothing(result.code) && push!(parts, "", "  --- Stan code ---", result.code)
        hasproperty(result, :stats) && !isnothing(result.stats) && push!(parts, "  Stats: abs_diff=$(result.stats.absolute_constant_difference), rel_diff=$(result.stats.relative_remaining_difference)")
        hasproperty(result, :dimension) && !isnothing(result.dimension) && push!(parts, "  Dimension: $(result.dimension)")
        join(parts, "\n")
    end

    plain_model(pn) = begin
        t_cached = @is_cached transpile_result[pn]
        c_cached = @is_cached compile_result[pn]
        r_cached = @is_cached correct_result[pn]
        t_result = t_cached ? transpile_result[pn] : nothing
        c_result = c_cached ? compile_result[pn] : nothing
        r_result = r_cached ? correct_result[pn] : nothing
        parts = ["# $pn", "",
            plain_result_section["Transpiles", t_result], "",
            plain_result_section["Compiles", c_result], "",
            plain_result_section["Correct", r_result],
        ]
        join(parts, "\n")
    end

    __page__(content) = htmx(
        h.body(
            h.header(class="container")(
                h.h1("StanBlocks.jl PosteriorDB Dashboard"),
                h.nav(h.a("PosteriorDB"; href=__self__), " | ", h.a("Tests"; href=__self__/"tests"), " | ", h.a("Sandbox"; href=__self__/"sandbox"), " | ", h.a("Gallery"; href=__self__/"gallery")),
            ),
            h.main(class="container")(h.div(content; id="content"))
        );
        pico_version="2",
        extra_head=(
            h.style(":root { --pico-font-size: 100%; } .hidden { display: none; } .full-width { max-width: 100%; padding: 0 1rem; }"),
            h.link(rel="stylesheet", href="https://cdn.jsdelivr.net/npm/prismjs@1/themes/prism-tomorrow.min.css"),
            h.script(src="https://cdn.jsdelivr.net/npm/prismjs@1/prism.min.js"),
            h.script(src="https://cdn.jsdelivr.net/npm/prismjs@1/components/prism-julia.min.js"),
            h.script(src="https://cdn.jsdelivr.net/npm/prismjs@1/components/prism-stan.min.js"),
            h.script("document.addEventListener('DOMContentLoaded',function(){document.body.addEventListener('htmx:afterSettle',function(){document.querySelectorAll('code[class*=\"language-\"]').forEach(function(el){if(!el.querySelector('span.token')){Prism.highlightElement(el);}});});});"),
            sortable_table_js(),
        ),
    )

    # --- Test routes (via @include — registers under /tests/) ---
    @include tests = TestRoutes(; __req__, test_module=@__MODULE__)

    @get clear_cache = begin
        foreach(rm, filter(f -> endswith(f, ".sjl"), readdir(cache_path; join=true)))
        h.p("Cache cleared")
    end

    # Backfill status sidecars from existing `.sjl` caches. Only touches
    # already-cached results — never triggers a fresh computation.
    @get backfill_status = begin
        n = 0
        for pn in posterior_names
            if !isfile(status_file["transpile", pn]) && @is_cached(transpile_result[pn])
                write_status["transpile", pn, transpile_result[pn].ok]; n += 1
            end
            if !isfile(status_file["compile", pn]) && @is_cached(compile_result[pn])
                write_status["compile", pn, compile_result[pn].ok]; n += 1
            end
            if !isfile(status_file["correct", pn]) && @is_cached(correct_result[pn])
                write_status["correct", pn, correct_result[pn].ok]; n += 1
            end
        end
        h.p("Backfilled $n sidecar(s)")
    end

    @get clear_cache_model(pn) = begin
        @clear_cache! transpile_result[pn]
        h.p("Cache cleared for $pn")
    end

    updated_summary_row(pn) = begin
        local dataset, model
        dataset, model = split_posterior_name(pn)
        t_cached = @is_cached transpile_result[pn]
        t_ok = t_cached ? transpile_result[pn].ok : false
        c_cached = @is_cached compile_result[pn]
        c_ok = c_cached ? compile_result[pn].ok : false
        r_cached = @is_cached correct_result[pn]
        r_ok = r_cached ? correct_result[pn].ok : false
        detail_id = "detail-$pn"
        toggle = "on click toggle .hidden on #$detail_id"
        h.tr(
            h.td(pn; style="cursor:pointer", _=toggle),
            h.td(dataset; style="cursor:pointer", _=toggle),
            h.td(model; style="cursor:pointer", _=toggle),
            status_cell_clickable(t_cached, t_ok, "/check_transpile/$pn", detail_id),
            status_cell_clickable(c_cached, c_ok, "/check_compile/$pn", detail_id),
            status_cell_clickable(r_cached, r_ok, "/check_correct/$pn", detail_id);
            id="row-$pn",
            hx_swap_oob="outerHTML:#row-$pn",
        )
    end

    @get check_transpile(pn) = begin
        force_check["transpile", pn]
        [model_detail_content[pn], h.template(updated_summary_row[pn])]
    end

    @get check_compile(pn) = begin
        force_check["compile", pn]
        [model_detail_content[pn], h.template(updated_summary_row[pn])]
    end

    @get check_correct(pn) = begin
        force_check["correct", pn]
        [model_detail_content[pn], h.template(updated_summary_row[pn])]
    end

    reference_stan(pn) = begin
        posterior = PosteriorDB.posterior(pdb(), pn)
        ref_path = PosteriorDB.path(PosteriorDB.implementation(PosteriorDB.model(posterior), "stan"))
        read(ref_path, String)
    end

    @get ref(pn) = reference_stan[pn]

    check_status(check, pn) = if check == "transpile"
        c = @is_cached transpile_result[pn]
        (c, c ? transpile_result[pn].ok : false)
    elseif check == "compile"
        c = @is_cached compile_result[pn]
        (c, c ? compile_result[pn].ok : false)
    elseif check == "correct"
        c = @is_cached correct_result[pn]
        (c, c ? correct_result[pn].ok : false)
    else
        (false, false)
    end

    filtered_names(check, status) = begin
        names = String[]
        for pn in posterior_names
            cached, ok = check_status[check, pn]
            show = if status == "pass"; cached && ok
            elseif status == "fail"; cached && !ok
            elseif status == "unchecked"; !cached
            else; true
            end
            show && push!(names, pn)
        end
        names
    end

    @get filter(check, status="all") = join(filtered_names[check, status], "\n")

    # Force (re)compute a single check, clearing failed cache first
    force_check(check, pn) = if check == "transpile"
        @is_cached(transpile_result[pn]) && !transpile_result[pn].ok && @clear_cache! transpile_result[pn]
        transpile_result[pn]
    elseif check == "compile"
        @is_cached(compile_result[pn]) && !compile_result[pn].ok && @clear_cache! compile_result[pn]
        compile_result[pn]
    elseif check == "correct"
        @is_cached(correct_result[pn]) && !correct_result[pn].ok && @clear_cache! correct_result[pn]
        correct_result[pn]
    end

    # Batch recheck: /recheck/{check}/{status}
    # Clears failed caches and retriggers computation for all matching models
    @get recheck(check, status="fail") = begin
        targets = filtered_names[check, status]
        results = String[]
        for pn in targets
            r = force_check[check, pn]
            push!(results, "$(r.ok ? "PASS" : "FAIL") $pn$(r.ok ? "" : " -- $(r.error)")")
        end
        join(results, "\n")
    end

    @get model(pn) = h.div(; id="content")(model_detail_content[pn])

    # --- Sandbox: interactive SLIC editor ---

    sandbox_path = joinpath(dirname(dirname(@__DIR__)), "web", "sandbox")

    sandbox_snippets() = begin
        isdir(sandbox_path) || mkpath(sandbox_path)
        snippets = Pair{String,String}[]
        for f in sort(readdir(sandbox_path))
            endswith(f, ".jl") || continue
            name = f[1:end-3]
            code = read(joinpath(sandbox_path, f), String)
            push!(snippets, name => code)
        end
        snippets
    end

    sandbox_save(name, code) = begin
        isdir(sandbox_path) || mkpath(sandbox_path)
        write(joinpath(sandbox_path, name * ".jl"), code)
    end

    sandbox_auto_name(code) = begin
        # Try to extract a meaningful name from the code
        m = match(r"@slic\s+(?:\([^)]*\)\s+)?begin\s*\n\s*(\w+)", code)
        base = isnothing(m) ? "snippet" : m[1]
        existing = first.(sandbox_snippets())
        name = base
        i = 1
        while name in existing
            i += 1
            name = "$(base)_$i"
        end
        name
    end

    sandbox_result(code) = begin
        result = begin # try
            if !isdefined(Main, :StanBlocks)
                Core.eval(Main, :(const StanBlocks = $StanBlocks))
                Base.eval(Main, :(using .StanBlocks))
            end
            exprs = Meta.parseall(code).args
            rv = nothing
            for expr in exprs
                expr isa LineNumberNode && continue
                rv = Base.eval(Main, expr)
            end
            m = rv::StanBlocks.stan.SlicModel
            sc = Base.invokelatest(stan_code, m)
            (ok=true, error=nothing, stacktrace=nothing, code=sc)
        # catch e
        #     error()
        #     bt = catch_backtrace()
        #     wrapped = e isa StanBlocksError ? e : StanBlocksError(:transpile, "sandbox", (e, bt))
        #     error_msg = first(split(sprint(showerror, wrapped), "\n"))
        #     trace = sprint(showerror, wrapped, nothing)
        #     full_trace = sprint(showerror, wrapped)
        #     (ok=false, error=error_msg, stacktrace=trace, full_stacktrace=full_trace, code=nothing)
        end
        result
    end

    stanc_check(stan_code_str) = begin
        stanc = "/home/niko/.cmdstan/cmdstan-2.37.0/bin/stanc"
        tmpfile = tempname() * ".stan"
        # try
            write(tmpfile, stan_code_str)
            io = IOBuffer()
            ok = success(pipeline(`$stanc --warn-pedantic $tmpfile`; stderr=io, stdout=io))
            rv = (ok=ok, output=String(take!(io)))
            isfile(tmpfile) && rm(tmpfile)
            rv
        # catch e
        #     error()
        #     (ok=false, output=sprint(showerror, e))
        # finally
        #     isfile(tmpfile) && rm(tmpfile)
        # end
    end

    sandbox_output(name, result) = h.div(; id="sandbox-output")(
        if result.ok
            h.div(
                h.p(h.strong("Transpilation: "), h.span("PASS"; style="color:green;font-weight:bold")),
                h.pre(h.code(result.code; class="language-stan"); style="max-height:600px;overflow:auto"),
            )
        else
            e.div(
                h.p(h.strong("Transpilation: "), h.span("FAIL"; style="color:red;font-weight:bold")),
                h.p(h.strong("Error: "), h.code(result.error)),
                isnothing(result.stacktrace) ? "" : h.pre(result.stacktrace; style="white-space:pre-wrap;max-height:300px;overflow:auto"),
                hasproperty(result, :full_stacktrace) && !isnothing(result.full_stacktrace) ? h.details(
                    h.summary("Full stacktrace (internal)"),
                    h.pre(result.full_stacktrace; style="white-space:pre-wrap;max-height:400px;overflow:auto"),
                ) : "",
            )
        end,
    )

    sandbox_editor(code; name="", target="#sandbox-output", standalone=false) = h.form(;
        hx_post=__self__/"sandbox_run", hx_target=target, hx_swap="outerHTML",
    )(
        name == "" ? "" : h.input(; name="name", type="hidden", value=name),
        standalone ? h.input(; name="standalone", type="hidden", value="1") : "",
        h.textarea(code; name="code", rows="15",
            style="font-family:monospace;width:100%;resize:vertical;margin:0",
            _="on keydown[(shiftKey or ctrlKey) and key is 'Enter'] halt the event send submit to closest <form/> on keydown[key is 'Escape'] halt the event set editor to closest .snippet-editor add .hidden to editor remove .hidden from previous <pre/> from editor"),
    )

    snippet_code_block(name, code, card_id; standalone=false) = h.div(
        h.pre(h.code(code; class="language-julia");
            style="max-height:200px;overflow:auto;cursor:default",
            title="Ctrl+click to edit",
            _="on click[ctrlKey or shiftKey] halt the event set editor to the next .snippet-editor add .hidden to me remove .hidden from editor focus() the first <textarea/> in editor"),
        h.div(; class="hidden snippet-editor")(
            sandbox_editor(code; name, target="#$card_id", standalone),
        ),
    )

    # Read a sidecar file `<name>.jl.<suffix>` (PASS/FAIL status, stanc
    # output, …) from `sandbox_path`. Used by both the live editor flow
    # and the read-only `/gallery` view to surface cached results
    # without re-evaluating snippets.
    sandbox_read_sidecar(name, suffix) = let p = joinpath(sandbox_path, "$name.jl.$suffix")
        isfile(p) ? read(p, String) : nothing
    end
    sandbox_read_stanc(name) = sandbox_read_sidecar(name, "stanc")
    sandbox_read_status(name) = sandbox_read_sidecar(name, "status")

    stanc_badge(name) = begin
        sc = sandbox_read_stanc(name)
        isnothing(sc) && return ""
        sc == "OK" ? h.span("stanc ✓"; style="color:green;font-size:0.8em", title="Stan compiler (stanc) accepted the generated code") :
            h.span("stanc ✗"; style="color:red;font-size:0.8em", title="Stan compiler (stanc) rejected the generated code — see error below")
    end

    stanc_error_block(name) = begin
        sc = sandbox_read_stanc(name)
        (isnothing(sc) || sc == "OK") && return ""
        h.div(; style="margin:0.4rem 0")(
            h.div("stanc rejected the generated Stan code:"; style="color:red;font-size:0.85em;font-weight:bold;margin-bottom:0.2rem"),
            h.pre(sc; style="white-space:pre-wrap;max-height:200px;overflow:auto;font-size:0.85em;margin:0;padding:0.3rem 0.5rem;border-left:2px solid red;background:var(--pico-code-background-color,#f6f6f6)"),
        )
    end

    snippet_card(name, code; standalone=false) = begin
        result = sandbox_result(code)
        write(joinpath(sandbox_path, name * ".jl.status"), result.ok ? "PASS" : result.error)
        if result.ok
            sc = stanc_check(result.code)
            write(joinpath(sandbox_path, name * ".jl.stanc"), sc.ok ? "OK" : sc.output)
        end
        should_fail = sandbox_should_fail(name)
        border_color = result.ok ? "var(--pico-ins-color)" :
            should_fail ? "orange" : "var(--pico-del-color)"
        card_id = "snippet-$name"
        refresh_url = standalone ? __self__/"sandbox_view/$name" : __self__/"sandbox_refresh/$name"
        status_badge = result.ok ? h.span("PASS"; style="color:green") :
            should_fail ? h.span("XFAIL"; style="color:orange") :
            h.span("FAIL"; style="color:red")
        expect_label = should_fail ? "xfail" : "should pass"
        h.div(; id=card_id, style="border-left:4px solid $border_color;padding:0.5rem 1rem;margin-bottom:0.5rem;background:var(--pico-card-background-color);min-width:0;overflow:hidden")(
            h.div(; style="display:flex;align-items:center;gap:0.5rem;flex-wrap:wrap")(
                h.strong(name; contenteditable="true", style="outline:none;min-width:3em",
                    _="on blur if my textContent.trim() is not '$name' fetch /sandbox_rename/$name?to=\${my.textContent.trim()} then put the result into #$card_id.outerHTML"),
                status_badge, stanc_badge(name),
                h.button("↻"; type="button", style="margin:0;padding:0.1rem 0.4rem",
                    hx_get=refresh_url, hx_target="#$card_id", hx_swap="outerHTML"),
                h.button(expect_label; type="button", style="margin:0;padding:0.1rem 0.4rem;font-size:0.8em",
                    hx_get=__self__/"sandbox_toggle_expect/$name", hx_target="#$card_id", hx_swap="outerHTML"),
                h.button("✕"; type="button", style="margin:0;padding:0.1rem 0.4rem;color:var(--pico-del-color)",
                    hx_delete=__self__/"sandbox_delete/$name", hx_target="#$card_id", hx_swap="outerHTML",
                    hx_confirm="Delete snippet '$name'?"),
                standalone ? "" : h.a("⧉"; href=__self__/"sandbox_view/$name", target="_blank", style="margin:0;padding:0.1rem 0.4rem;text-decoration:none"),
                h.a("macro"; href=__self__/"sandbox_macroexpand/$name", target="_blank", style="margin:0;padding:0.1rem 0.4rem;text-decoration:none", title="Show macroexpanded code"),
            ),
            snippet_code_block(name, code, card_id; standalone),
            stanc_error_block(name),
            standalone ? sandbox_output(name, result) :
            h.details(
                h.summary(result.ok ? "Stan code" : "Error"),
                sandbox_output(name, result),
            ),
        )
    end

    sandbox_read_expect(name) = begin
        ep = joinpath(sandbox_path, name * ".jl.expect")
        isfile(ep) ? read(ep, String) : "pass"
    end

    sandbox_should_fail(name) = sandbox_read_expect(name) == "fail"

    sandbox_sort_key(name) = begin
        status = sandbox_read_status(name)
        should_fail = sandbox_should_fail(name)
        stanc = sandbox_read_stanc(name)
        ok = isnothing(status) ? nothing : status == "PASS"
        stanc_ok = isnothing(stanc) ? nothing : stanc == "OK"
        if !isnothing(ok) && !ok && !should_fail
            0  # unexpected transpile failure — first
        elseif isnothing(ok)
            1  # untried — second
        elseif ok && !isnothing(stanc_ok) && !stanc_ok
            2  # transpiles but stanc fails — third
        elseif ok
            3  # fully passing — middle
        else
            4  # expected failure — last
        end
    end

    snippet_card_lazy(name, code) = begin
        card_id = "snippet-$name"
        status = sandbox_read_status(name)
        should_fail = sandbox_should_fail(name)
        border_color = isnothing(status) ? "gray" :
            status == "PASS" ? "var(--pico-ins-color)" :
            should_fail ? "orange" : "var(--pico-del-color)"
        status_badge = isnothing(status) ? "" :
            status == "PASS" ? h.span("PASS"; style="color:green") :
            should_fail ? h.span("XFAIL"; style="color:orange") :
            h.span("FAIL"; style="color:red", title=status)
        expect_label = should_fail ? "xfail" : "should pass"
        h.div(; id=card_id, style="border-left:4px solid $border_color;padding:0.5rem 1rem;margin-bottom:0.5rem;background:var(--pico-card-background-color);min-width:0;overflow:hidden")(
            h.div(; style="display:flex;align-items:center;gap:0.5rem;flex-wrap:wrap")(
                h.strong(name),
                status_badge, stanc_badge(name),
                h.button("↻"; type="button", style="margin:0;padding:0.1rem 0.4rem",
                    hx_get=__self__/"sandbox_refresh/$name", hx_target="#$card_id", hx_swap="outerHTML"),
                h.button(expect_label; type="button", style="margin:0;padding:0.1rem 0.4rem;font-size:0.8em",
                    hx_get=__self__/"sandbox_toggle_expect/$name", hx_target="#$card_id", hx_swap="outerHTML"),
                h.button("✕"; type="button", style="margin:0;padding:0.1rem 0.4rem;color:var(--pico-del-color)",
                    hx_delete=__self__/"sandbox_delete/$name", hx_target="#$card_id", hx_swap="outerHTML",
                    hx_confirm="Delete snippet '$name'?"),
                h.a("⧉"; href=__self__/"sandbox_view/$name", target="_blank", style="margin:0;padding:0.1rem 0.4rem;text-decoration:none"),
            ),
            snippet_code_block(name, code, card_id),
            stanc_error_block(name),
        )
    end

    snippet_list() = h.div(; id="snippet-list", style="display:grid;grid-template-columns:repeat(3,1fr);gap:0.5rem")(
        [snippet_card_lazy(name, code) for (name, code) in sort(sandbox_snippets(), by=p -> (sandbox_sort_key(first(p)), first(p)))]...
    )

    default_slic_code = """@slic (;y=randn(10)) begin
    mu ~ normal(0, 1)
    sigma ~ gamma(1, 1)
    y ~ normal(mu, sigma)
end"""

    # `/gallery` — read-only view of `web/sandbox/` rendered via
    # `HTMXObjects.gallery_grid`. Same files as `/sandbox` but no editor /
    # CRUD; this is what the docs `record_gallery` flow ships as static
    # recordings.
    sandbox_gallery_card(item) = let code = item.code_string
        status = sandbox_read_status(item.id)
        ok_pass = status == "PASS"
        h.article(; class="htmxo-gallery-card", id=item.id,
                    style="margin:0; padding:0.5rem; min-width:0; overflow:hidden;")(
            h.header(; class="htmxo-gallery-card-header",
                       style="padding:0 0 0.25rem; margin:0; display:flex; align-items:baseline; gap:0.5rem; flex-wrap:wrap;")(
                h.a(item.title; class="htmxo-gallery-card-title",
                    href=__self__/"sandbox_view/$(item.id)", target="_blank",
                    style="font-weight:bold; font-size:0.9em;"),
                isnothing(status) ?
                    h.span("?"; style="color:gray;font-size:0.75em") :
                ok_pass ?
                    h.span("PASS"; style="color:green;font-size:0.75em;font-weight:bold") :
                    h.span("FAIL"; style="color:red;font-size:0.75em;font-weight:bold"),
                stanc_badge(item.id),
            ),
            isempty(item.description) ? h.span() :
                h.p(item.description; class="htmxo-gallery-card-description",
                    style="font-size:0.8em;color:var(--pico-muted-color);margin:0 0 0.25rem;"),
            h.details(; class="htmxo-gallery-card-code")(
                h.summary("Julia"; style="font-size:0.8em;"),
                h.pre(h.code(code; class="language-julia");
                      style="background:var(--pico-code-background-color); padding:0.5rem; border-radius:0.25rem; overflow-x:auto; font-size:0.75em; max-height:200px;"),
            ),
            (!isnothing(status) && !ok_pass) ?
                h.div(; style="font-size:0.8em;color:var(--pico-del-color);margin-top:0.25rem")(
                    h.code(strip(status))) : h.span(),
            stanc_error_block(item.id),
        )
    end

    @get gallery = htmx(
        h.main(; class="container-fluid", style="padding:1rem 2rem;")(
            h.h2("StanBlocks Sandbox Gallery"),
            h.p("Read-only view of ", h.code("web/sandbox/"),
                " — same snippets the ", h.a("editor"; href=__self__/"sandbox"),
                " serves, rendered as a flat gallery grid for the docs deploy."),
            gallery_grid(_sandbox_gallery.items;
                section_titles=_sandbox_gallery.section_titles,
                card_renderer=sandbox_gallery_card, columns=2),
        );
        extra_head=(h.script(; src="https://cdn.jsdelivr.net/npm/prismjs@1/prism.min.js"),
                    h.script(; src="https://cdn.jsdelivr.net/npm/prismjs@1/components/prism-julia.min.js"),
                    h.script(; src="https://cdn.jsdelivr.net/npm/prismjs@1/components/prism-stan.min.js"),
                    h.link(; rel="stylesheet", href="https://cdn.jsdelivr.net/npm/prismjs@1/themes/prism-tomorrow.min.css")),
    )

    # `/record_gallery` — drive the AppData IP `record_gallery` to dump
    # `/gallery` + every `/sandbox_view/<id>` URL into
    # `docs/src/public/live-sb/`. Long-running so it goes through
    # `polling_fetchindex`. Override the deploy URL prefix via
    # `RECORD_BASE_PREFIX` env var (default `/StanBlocks.jl/dev/live-sb`).
    @get record_gallery(; record_dir::String="", record_base::String="", force::Bool=false) = begin
        rd = isempty(record_dir) ?
            joinpath(dirname(dirname(@__DIR__)), "docs", "src", "public", "live-sb") :
            record_dir
        rb = isempty(record_base) ?
            get(ENV, "RECORD_BASE_PREFIX", "/StanBlocks.jl/dev/live-sb") :
            record_base

        polling_fetchindex(__appdata__.record_gallery, rd, rb;
                           poll_url=__route__,
                           label="Recording sandbox gallery",
                           force) do summary
            h.article(
                h.header(h.h2("Gallery recorded")),
                h.p("Wrote ", h.code(string(summary.n_paths)),
                    " routes (× full + HX shapes) into ",
                    h.code(summary.record_dir), "."),
                h.ul(
                    h.li(h.code(string(summary.n_html)), " .html"),
                    h.li(h.code(string(summary.n_js)),   " .js"),
                    h.li(h.code(string(summary.n_json)), " .json"),
                    h.li(h.code(string(summary.n_other)), " other"),
                ),
                h.p(h.strong("Next: "),
                    h.code("git add docs/src/public/live-sb && git commit && git push"),
                    " — CI deploys the rest."),
                h.p("Re-record (overwrites cache): ",
                    h.a("/record_gallery?force=true";
                        href=query_url(__self__/"record_gallery"; force=true))),
            )
        end
    end

    @get sandbox = h.div(
        h.style("#content { max-width: 100%; } main.container { max-width: 100%; padding: 0 1rem; }"),
        h.h2("SLIC Sandbox"),
        h.div(; style="display:flex;align-items:center;gap:1rem")(
            h.h3("Saved Snippets"; style="margin:0"),
            h.button("Compile Failures"; type="button", style="padding:0.2rem 0.6rem",
                hx_get=__self__/"sandbox_compile_failures", hx_target="#compile-all-output", hx_swap="innerHTML"),
            h.button("Spot Check (5)"; type="button", style="padding:0.2rem 0.6rem",
                hx_get=__self__/"sandbox_spot_check", hx_target="#compile-all-output", hx_swap="innerHTML"),
            h.button("Compile All"; type="button", style="padding:0.2rem 0.6rem",
                hx_get=__self__/"sandbox_compile_all", hx_target="#compile-all-output", hx_swap="innerHTML"),
            h.button("Verify stanc"; type="button", style="padding:0.2rem 0.6rem",
                hx_get=__self__/"sandbox_stanc_all", hx_target="#compile-all-output", hx_swap="innerHTML"),
        ),
        snippet_list(),
        h.div(; id="compile-all-output")(""),
        h.h3("New"),
        sandbox_editor(default_slic_code),
        h.div(; id="sandbox-output")(""),
    )

    @post sandbox_run(; code="", name="", standalone="") = begin
        from_card = !isempty(strip(name))
        name = strip(name)
        isempty(name) && (name = sandbox_auto_name(code))
        sandbox_save(name, code)
        if from_card
            snippet_card(name, code; standalone=standalone=="1")
        else
            result = sandbox_result(code)
            [sandbox_output(name, result), h.template(snippet_list)]
        end
    end

    @get sandbox_refresh(name) = begin
        code = read(joinpath(sandbox_path, name * ".jl"), String)
        snippet_card(name, code)
    end

    @get sandbox_refresh_all = snippet_list

    @get sandbox_rename(name; to="") = begin
        to = strip(to)
        isempty(to) && return snippet_card(name, read(joinpath(sandbox_path, name * ".jl"), String))
        old_path = joinpath(sandbox_path, name * ".jl")
        new_path = joinpath(sandbox_path, to * ".jl")
        isfile(old_path) && mv(old_path, new_path; force=true)
        code = read(new_path, String)
        snippet_card(to, code)
    end

    @get sandbox_view(name) = begin
        code = read(joinpath(sandbox_path, name * ".jl"), String)
        snippet_card(name, code; standalone=true)
    end

    @get sandbox_macroexpand(name) = begin
        code = read(joinpath(sandbox_path, name * ".jl"), String)
        mod = Module(:Sandbox)
        Core.eval(mod, :(const StanBlocks = $StanBlocks))
        Base.eval(mod, :(using .StanBlocks))
        exprs = Meta.parseall(code).args
        expanded = String[]
        for expr in exprs
            expr isa LineNumberNode && continue
            ex = # try
                Base.eval(mod, :(macroexpand($mod, $(QuoteNode(expr)))))
            # catch e
            #     error()
            #     expr
            # end
            push!(expanded, sprint(Base.show_unquoted, ex))
        end
        h.pre(h.code(join(expanded, "\n\n"); class="language-julia"))
    end

    compile_snippets(snippets) = begin
        results = map(snippets) do (name, code)
            t0 = time()
            result = sandbox_result(code)
            write(joinpath(sandbox_path, name * ".jl.status"), result.ok ? "PASS" : result.error)
            dt = time() - t0
            (name=name, ok=result.ok, error=result.ok ? nothing : result.error, time=dt)
        end
        n_pass = count(r -> r.ok, results)
        n_fail = length(results) - n_pass
        total_time = sum(r -> r.time, results)
        h.div(
            h.p(
                h.strong("$(n_pass)/$(length(results)) passed"),
                n_fail > 0 ? h.span(" ($n_fail failed)"; style="color:red") : "",
                h.span(" in $(round(total_time; digits=2))s"; style="color:gray"),
            ),
            [h.div(; style="padding:0.1rem 0;font-family:monospace;font-size:0.85em")(
                r.ok ? h.span("PASS"; style="color:green") : h.span("FAIL"; style="color:red"),
                " ", r.name,
                h.span(" $(round(r.time*1000; digits=0))ms"; style="color:gray"),
                r.ok ? "" : h.span(" - ", r.error; style="color:red;font-size:0.9em"),
            ) for r in results]...,
        )
    end

    @get sandbox_compile_all = compile_snippets(sandbox_snippets())

    @get sandbox_compile_failures = begin
        to_compile = filter(sandbox_snippets()) do (name, _)
            status = sandbox_read_status(name)
            should_fail = sandbox_should_fail(name)
            isnothing(status) || (status != "PASS" && !should_fail)
        end
        isempty(to_compile) ? h.p("No unexpected failures or untried snippets") : compile_snippets(to_compile)
    end

    @get sandbox_spot_check(; n="5") = begin
        num = parse(Int, n)
        passes = filter(sandbox_snippets()) do (name, _)
            status = sandbox_read_status(name)
            !isnothing(status) && status == "PASS"
        end
        sample = passes[Random.randperm(length(passes))[1:min(num, length(passes))]]
        isempty(sample) ? h.p("No passing snippets to spot check") : compile_snippets(sample)
    end

    @get sandbox_stanc_check(name) = begin
        code = read(joinpath(sandbox_path, name * ".jl"), String)
        result = sandbox_result(code)
        if !result.ok
            h.div(h.span("Cannot check: transpilation failed"; style="color:red"))
        else
            sc = stanc_check(result.code)
            if sc.ok
                h.div(h.span("stanc: OK"; style="color:green"),
                    isempty(sc.output) ? "" : h.pre(sc.output; style="font-size:0.85em;max-height:200px;overflow:auto"))
            else
                h.div(h.span("stanc: FAIL"; style="color:red"),
                    h.pre(sc.output; style="font-size:0.85em;max-height:200px;overflow:auto;white-space:pre-wrap"))
            end
        end
    end

    @get sandbox_stanc_all = begin
        snippets = sandbox_snippets()
        results = map(snippets) do (name, code)
            result = sandbox_result(code)
            if !result.ok
                (name=name, ok=nothing, output="transpile failed")
            else
                sc = stanc_check(result.code)
                (name=name, ok=sc.ok, output=sc.output)
            end
        end
        verified = count(r -> r.ok === true, results)
        failed = count(r -> r.ok === false, results)
        skipped = count(r -> isnothing(r.ok), results)
        h.div(
            h.p(
                h.strong("stanc: $verified verified"),
                failed > 0 ? h.span(", $failed failed"; style="color:red") : "",
                skipped > 0 ? h.span(", $skipped skipped"; style="color:gray") : "",
            ),
            [h.div(; style="padding:0.1rem 0;font-family:monospace;font-size:0.85em")(
                isnothing(r.ok) ? h.span("SKIP"; style="color:gray") :
                r.ok ? h.span("OK"; style="color:green") : h.span("FAIL"; style="color:red"),
                " ", r.name,
                (!isnothing(r.ok) && !r.ok) ? h.span(" - ", r.output; style="color:red;font-size:0.9em") : "",
            ) for r in results]...,
        )
    end

    @get sandbox_toggle_expect(name) = begin
        ep = joinpath(sandbox_path, name * ".jl.expect")
        current = sandbox_read_expect(name)
        write(ep, current == "fail" ? "pass" : "fail")
        code = read(joinpath(sandbox_path, name * ".jl"), String)
        snippet_card_lazy(name, code)
    end

    @delete sandbox_delete(name) = begin
        fp = joinpath(sandbox_path, name * ".jl")
        isfile(fp) && rm(fp)
        sp = joinpath(sandbox_path, name * ".jl.status")
        isfile(sp) && rm(sp)
        ep = joinpath(sandbox_path, name * ".jl.expect")
        isfile(ep) && rm(ep)
        sc = joinpath(sandbox_path, name * ".jl.stanc")
        isfile(sc) && rm(sc)
        ""
    end
end

function __init__()
    route!(AppContext())
end

end # module PosteriorDBWeb
