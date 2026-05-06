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
using Treebars: prepare_progress!, with_prepared_progress, polling_fetchindex, initialize_progress!

# include("test/runtests.jl")


pdb() = PosteriorDB.database()


# --- Test helpers (throw on failure; route safety wrapper renders the error article) ---

transpile_check(post) = (code=stan_code(post),)

compile_check(post) = (dimension=LogDensityProblems.dimension(instantiate(post)),)

function compare_logdensities(reference, test; stat_f=median, m=200)
    n = LogDensityProblems.dimension(test)
    X = randn((n, m))
    reference_lpdfs = [LogDensityProblems.logdensity(reference, x) for x in eachcol(X)]
    test_lpdfs = [LogDensityProblems.logdensity(test, x) for x in eachcol(X)]
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

function correct_check(posterior)
    post = slic_implementation(posterior)
    problem = instantiate(post)
    ref_stan_path = PosteriorDB.path(PosteriorDB.implementation(PosteriorDB.model(posterior), "stan"))
    ref_data = PosteriorDB.load(PosteriorDB.dataset(posterior), String)
    ref_problem = StanLogDensityProblems.StanProblem(ref_stan_path, ref_data; nan_on_error=true, make_args=["STAN_THREADS=true"])
    result = compare_logdensities(ref_problem, problem)
    isnothing(result) && error("No finite evaluations")
    result.relative_remaining_difference < 1e-6 ||
        error("Relative difference: $(result.relative_remaining_difference)")
    (stats=result,)
end

# --- Web app ---

function status_str(status::Symbol)
    status == :ready ? "PASS" : status == :started ? "FAIL" : "-"
end

function status_cell_clickable(status::Symbol, check_url, detail_id)
    if status == :unstarted
        return h.td("-"; class="check-cell u-pointer u-text-muted",
            hx_get=check_url, hx_target="#$detail_id", hx_swap="innerHTML",
            _="on htmx:afterOnLoad if not me.classList.contains('batch') then remove .u-hidden from #$detail_id end remove .batch from me")
    elseif status == :started
        return h.td("FAIL"; class="check-cell u-pointer u-text-error u-text-bold",
            hx_get=check_url, hx_target="#$detail_id", hx_swap="innerHTML",
            _="on htmx:afterOnLoad if not me.classList.contains('batch') then remove .u-hidden from #$detail_id end remove .batch from me")
    else
        return h.td("PASS"; class="check-cell u-pointer u-text-success u-text-bold",
            _="on click toggle .u-hidden on #$detail_id")
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
    # `polling_fetchindex` renders progress via `htmx_render(__status__)`;
    # without an initialized tree the rendering callback hits
    # `htmx_render(::Nothing)` on the first call. `:state` selects the
    # text-tree backend used by the polling progress UI.
    __status__ = initialize_progress!(:state; description="StanBlocks sandbox")

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

    @cached posterior_names = sort([
        pn for pn in PosteriorDB.posterior_names(pdb())
        if !isnothing(slic_implementation(PosteriorDB.posterior(pdb(), pn)))
    ])

    @cached transpile_result(pn) = transpile_check(slic_implementation(PosteriorDB.posterior(pdb(), pn)))
    @cached compile_result(pn)   = compile_check(slic_implementation(PosteriorDB.posterior(pdb(), pn)))
    @cached correct_result(pn)   = correct_check(PosteriorDB.posterior(pdb(), pn))

    # `@cache_status` inspects the `.sjl` file presence/size — `:unstarted` if
    # absent, `:started` if empty (touched at compute start, left empty when
    # the body throws), `:ready` if successfully serialized. Drives the
    # dashboard without deserializing the full result tuple.
    transpile_status(pn) = @cache_status transpile_result[pn]
    compile_status(pn)   = @cache_status compile_result[pn]
    correct_status(pn)   = @cache_status correct_result[pn]

    check_status_sym(check, pn) = check == "transpile" ? transpile_status[pn] :
                                  check == "compile"   ? compile_status[pn] :
                                  check == "correct"   ? correct_status[pn] : :unstarted

    overview_row(pn) = begin
        local dataset, model
        dataset, model = split_posterior_name(pn)
        detail_id = "detail-$pn"
        toggle = "on click toggle .u-hidden on #$detail_id"
        [h.tr(
            h.td(pn; class="u-pointer", _=toggle),
            h.td(dataset; class="u-pointer", _=toggle),
            h.td(model; class="u-pointer", _=toggle),
            status_cell_clickable(transpile_status[pn], "/check_transpile/$pn", detail_id),
            status_cell_clickable(compile_status[pn], "/check_compile/$pn", detail_id),
            status_cell_clickable(correct_status[pn], "/check_correct/$pn", detail_id);
            id="row-$pn",
        ),
        h.tr(; id=detail_id, class="u-hidden")]
    end

    @get index = h.div(
        h.h2("PosteriorDB Models ($(length(posterior_names)) implemented)"),
        h.input(;
            type="search",
            id="search",
            placeholder="Filter posteriors...",
            _="on input set query to my value.toLowerCase() for row in <tr/> in #posterior-tbody if row.textContent.toLowerCase() contains query remove .u-hidden from row else add .u-hidden to row end end on keydown[key is 'Enter'] halt the event for row in <tr[id^='row-']/> in #posterior-tbody if row matches ':not(.u-hidden)' set target to null for cell in <td.check-cell/> in row if target is null and cell.textContent.trim() is not 'PASS' set target to cell end end if target is not null add .batch to target send click to target end end end",
            class="u-w-full u-mb-4",
        ),
        h.table(class="striped"; role="grid")(
            h.thead(
                h.tr(
                    h.th("Posterior"; _="on click call sortTable(0, me)", class="u-pointer"),
                    h.th("Dataset"; _="on click call sortTable(1, me)", class="u-pointer"),
                    h.th("Model"; _="on click call sortTable(2, me)", class="u-pointer"),
                    h.th("Transpiles"; _="on click call sortTable(3, me)", class="u-pointer"),
                    h.th("Compiles"; _="on click call sortTable(4, me)", class="u-pointer"),
                    h.th("Correct"; _="on click call sortTable(5, me)", class="u-pointer"),
                )
            ),
            h.tbody(reduce(vcat, [overview_row[pn] for pn in posterior_names]; init=[])...; id="posterior-tbody")
        ),
        h.style("tr[id^=row-]:hover { background: var(--pico-table-row-stripped-background-color); } details summary { cursor: pointer; font-weight: 600; margin-bottom: 0.5rem; }"),
    )

    # `:ready` → render success info; `:started` → "FAIL" with re-run link
    # (clicking surfaces the underlying error via the route safety wrapper);
    # `:unstarted` → empty (overview cell already provides the run link).
    result_section(label, status::Symbol, result_url, retry_url) = begin
        status == :unstarted && return ""
        if status == :started
            return h.div(; class="u-mb-2")(
                h.p(h.strong(label, ": "), status_badge(:failed; label="FAIL"), " ",
                    h.a("Re-run to view error"; hx_get=retry_url, hx_target="closest tr", hx_swap="outerHTML")),
            )
        end
        result = if label == "Transpiles"; transpile_result[result_url]
                 elseif label == "Compiles"; compile_result[result_url]
                 else; correct_result[result_url] end
        h.div(; class="u-mb-2")(
            h.p(h.strong(label, ": "), status_badge(:done; label="PASS")),
            hasproperty(result, :code) ? h.details(
                h.summary("Generated Stan code"),
                h.pre(result.code; class="u-pre-wrap u-scroll-y-lg u-text-sm");
                open=""
            ) : "",
            hasproperty(result, :stats) ? h.p(
                h.strong("Stats: "),
                "abs diff = $(result.stats.absolute_constant_difference), rel diff = $(result.stats.relative_remaining_difference)"
            ) : "",
        )
    end

    model_detail_content(pn) = begin
        ts = transpile_status[pn]
        cs = compile_status[pn]
        rs = correct_status[pn]
        any_fail = ts == :started || cs == :started || rs == :started
        all_ready = ts != :started && cs != :started && rs != :started
        status_class = any_fail ? "u-status-callout u-status-error" :
                       all_ready ? "u-status-callout u-status-success" :
                       "u-status-callout"
        h.td(; colspan="6", class="pdb-detail-cell")(
            h.div(; class=status_class)(
                h.h4(pn),
                result_section["Transpiles", ts, pn, "/check_transpile/$pn"],
                result_section["Compiles", cs, pn, "/check_compile/$pn"],
                result_section["Correct", rs, pn, "/check_correct/$pn"],
            )
        )
    end

    plain_overview = begin
        header = rpad("Posterior", 60) * rpad("Transpiles", 12) * rpad("Compiles", 12) * "Correct"
        lines = [header, "-"^length(header)]
        for pn in posterior_names
            push!(lines, rpad(pn, 60) *
                rpad(status_str(transpile_status[pn]), 12) *
                rpad(status_str(compile_status[pn]),   12) *
                         status_str(correct_status[pn]))
        end
        join(lines, "\n")
    end

    plain_result_section(label, status::Symbol, pn) = begin
        status == :unstarted && return "$label: -"
        status == :started && return "$label: FAIL (re-run /check_$(lowercase(label))/$pn for error)"
        parts = ["$label: PASS"]
        result = label == "Transpiles" ? transpile_result[pn] :
                 label == "Compiles" ? compile_result[pn] : correct_result[pn]
        hasproperty(result, :code) && push!(parts, "", "  --- Stan code ---", result.code)
        hasproperty(result, :stats) && push!(parts, "  Stats: abs_diff=$(result.stats.absolute_constant_difference), rel_diff=$(result.stats.relative_remaining_difference)")
        hasproperty(result, :dimension) && push!(parts, "  Dimension: $(result.dimension)")
        join(parts, "\n")
    end

    plain_model(pn) = join([
        "# $pn", "",
        plain_result_section["Transpiles", transpile_status[pn], pn], "",
        plain_result_section["Compiles",   compile_status[pn],   pn], "",
        plain_result_section["Correct",    correct_status[pn],   pn],
    ], "\n")

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
            h.style("""
                :root { --pico-font-size: 100%; }
                .full-width { max-width: 100%; padding: 0 1rem; }
                .pdb-code-large { max-height: 600px; overflow: auto; }
                .pdb-detail-cell { padding: 0; border: none; }
                .pdb-snippet-input { font-family: monospace; width: 100%; resize: vertical; margin: 0; }
                .pdb-snippet-output { max-height: 200px; overflow: auto; cursor: default; }
                .pdb-snippet-header { display: flex; align-items: center; gap: 0.5rem; flex-wrap: wrap; }
                .pdb-name-input { outline: none; min-width: 3em; }
                .pdb-icon-btn { margin: 0; padding: 0.1rem 0.4rem; }
                .pdb-icon-del { color: var(--pico-del-color); }
                .pdb-snippet-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 0.5rem; }
                .pdb-snippet-result-row { padding: 0.1rem 0; font-family: monospace; font-size: 0.85em; }
            """),
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

    @get clear_cache_model(pn) = begin
        @clear_cache! transpile_result[pn]
        @clear_cache! compile_result[pn]
        @clear_cache! correct_result[pn]
        h.p("Cache cleared for $pn")
    end

    updated_summary_row(pn) = begin
        local dataset, model
        dataset, model = split_posterior_name(pn)
        detail_id = "detail-$pn"
        toggle = "on click toggle .u-hidden on #$detail_id"
        h.tr(
            h.td(pn; class="u-pointer", _=toggle),
            h.td(dataset; class="u-pointer", _=toggle),
            h.td(model; class="u-pointer", _=toggle),
            status_cell_clickable(transpile_status[pn], "/check_transpile/$pn", detail_id),
            status_cell_clickable(compile_status[pn], "/check_compile/$pn", detail_id),
            status_cell_clickable(correct_status[pn], "/check_correct/$pn", detail_id);
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

    filtered_names(check, status) = begin
        names = String[]
        for pn in posterior_names
            s = check_status_sym[check, pn]
            show = status == "pass" ? s == :ready :
                   status == "fail" ? s == :started :
                   status == "unchecked" ? s == :unstarted : true
            show && push!(names, pn)
        end
        names
    end

    @get filter(check, status="all") = join(filtered_names[check, status], "\n")

    # Force (re)compute a single check, clearing the empty `:started` marker
    # left behind by a previously-failed run so the next access recomputes.
    force_check(check, pn) = begin
        if check == "transpile"
            transpile_status[pn] == :started && @clear_cache! transpile_result[pn]
            transpile_result[pn]
        elseif check == "compile"
            compile_status[pn] == :started && @clear_cache! compile_result[pn]
            compile_result[pn]
        elseif check == "correct"
            correct_status[pn] == :started && @clear_cache! correct_result[pn]
            correct_result[pn]
        end
    end

    # Batch recheck: /recheck/{check}/{status}. Errors bubble through the
    # route safety wrapper; PASSes are reported in the response body.
    @get recheck(check, status="fail") = begin
        results = String[]
        for pn in filtered_names[check, status]
            force_check[check, pn]
            push!(results, "PASS $pn")
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

    # Eval user-typed SLIC code and transpile. Throws on parse / transpile
    # failure — the per-card render wraps this in `safely(; obj=__self__)`
    # so the error is contained to the failing card.
    sandbox_result(code) = begin
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
        (code=Base.invokelatest(stan_code, m),)
    end

    stanc_check(stan_code_str) = begin
        stanc = "/home/niko/.cmdstan/cmdstan-2.37.0/bin/stanc"
        tmpfile = tempname() * ".stan"
        write(tmpfile, stan_code_str)
        io = IOBuffer()
        ok = success(pipeline(`$stanc --warn-pedantic $tmpfile`; stderr=io, stdout=io))
        rm(tmpfile; force=true)
        (ok=ok, output=String(take!(io)))
    end

    sandbox_output(result) = h.div(; id="sandbox-output")(
        h.div(
            h.p(h.strong("Transpilation: "), h.span("PASS"; class="u-text-success u-text-bold")),
            h.pre(h.code(result.code; class="language-stan"); class="pdb-code-large"),
        ),
    )

    sandbox_editor(code; name="", target="#sandbox-output", standalone=false) = h.form(;
        hx_post=__self__/"sandbox_run", hx_target=target, hx_swap="outerHTML",
    )(
        name == "" ? "" : h.input(; name="name", type="hidden", value=name),
        standalone ? h.input(; name="standalone", type="hidden", value="1") : "",
        h.textarea(code; name="code", rows="15",
            class="pdb-snippet-input",
            _="on keydown[(shiftKey or ctrlKey) and key is 'Enter'] halt the event send submit to closest <form/> on keydown[key is 'Escape'] halt the event set editor to closest .snippet-editor add .u-hidden to editor remove .u-hidden from previous <pre/> from editor"),
    )

    snippet_code_block(name, code, card_id; standalone=false) = h.div(
        h.pre(h.code(code; class="language-julia");
            class="pdb-snippet-output",
            title="Ctrl+click to edit",
            _="on click[ctrlKey or shiftKey] halt the event set editor to the next .snippet-editor add .u-hidden to me remove .u-hidden from editor focus() the first <textarea/> in editor"),
        h.div(; class="u-hidden snippet-editor")(
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
    sandbox_read_stan(name) = sandbox_read_sidecar(name, "stan")


    stanc_badge(name) = begin
        sc = sandbox_read_stanc(name)
        isnothing(sc) && return ""
        sc == "OK" ?
            h.span("stanc ✓"; class="htmxo-gallery-card-status is-ok",
                title="Stan compiler (stanc) accepted the generated code") :
            h.span("stanc ✗"; class="htmxo-gallery-card-status is-error",
                title="Stan compiler (stanc) rejected the generated code — see error below")
    end

    stanc_error_block(name) = begin
        sc = sandbox_read_stanc(name)
        (isnothing(sc) || sc == "OK") && return ""
        h.div(class="htmxo-gallery-card-error")(
            h.strong("stanc rejected the generated Stan code:"),
            h.pre(h.code(sc); class="htmxo-gallery-card-code"),
        )
    end

    snippet_card(name, code; standalone=false) = safely(; obj=__self__) do
        result = sandbox_result(code)
        write(joinpath(sandbox_path, name * ".jl.status"), "PASS")
        write(joinpath(sandbox_path, name * ".jl.stan"), result.code)
        sc = stanc_check(result.code)
        write(joinpath(sandbox_path, name * ".jl.stanc"), sc.ok ? "OK" : sc.output)
        should_fail = sandbox_should_fail(name)
        status_class = "u-status-callout u-status-success"
        card_id = "snippet-$name"
        refresh_url = standalone ? __self__/"sandbox_view/$name" : __self__/"sandbox_refresh/$name"
        status_badge = should_fail ? h.span("XPASS"; class="u-text-warning",
                                            title="Expected failure but transpiled — toggle 'should pass' if intentional") :
                                     h.span("PASS"; class="u-text-success")
        expect_label = should_fail ? "xfail" : "should pass"
        h.div(; id=card_id, class=status_class)(
            h.div(; class="pdb-snippet-header")(
                h.strong(name; contenteditable="true", class="pdb-name-input",
                    _="on blur if my textContent.trim() is not '$name' fetch /sandbox_rename/$name?to=\${my.textContent.trim()} then put the result into #$card_id.outerHTML"),
                status_badge, stanc_badge(name),
                h.button("↻"; type="button", class="pdb-icon-btn",
                    hx_get=refresh_url, hx_target="#$card_id", hx_swap="outerHTML"),
                h.button(expect_label; type="button", class="pdb-icon-btn u-text-xs",
                    hx_get=__self__/"sandbox_toggle_expect/$name", hx_target="#$card_id", hx_swap="outerHTML"),
                h.button("✕"; type="button", class="pdb-icon-btn pdb-icon-del",
                    hx_delete=__self__/"sandbox_delete/$name", hx_target="#$card_id", hx_swap="outerHTML",
                    hx_confirm="Delete snippet '$name'?"),
                standalone ? "" : h.a("⧉"; href=__self__/"sandbox_view/$name", target="_blank", class="pdb-icon-btn u-link-plain"),
                h.a("macro"; href=__self__/"sandbox_macroexpand/$name", target="_blank", class="pdb-icon-btn u-link-plain", title="Show macroexpanded code"),
            ),
            snippet_code_block(name, code, card_id; standalone),
            stanc_error_block(name),
            standalone ? sandbox_output(result) :
            h.details(
                h.summary("Stan code"),
                sandbox_output(result),
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
        status_class = isnothing(status) ? "u-status-callout" :
            status == "PASS" ? "u-status-callout u-status-success" :
            should_fail ? "u-status-callout u-status-warning" : "u-status-callout u-status-error"
        status_badge = isnothing(status) ? "" :
            status == "PASS" ? h.span("PASS"; class="u-text-success") :
            should_fail ? h.span("XFAIL"; class="u-text-warning") :
            h.span("FAIL"; class="u-text-error", title=status)
        expect_label = should_fail ? "xfail" : "should pass"
        h.div(; id=card_id, class=status_class)(
            h.div(; class="pdb-snippet-header")(
                h.strong(name),
                status_badge, stanc_badge(name),
                h.button("↻"; type="button", class="pdb-icon-btn",
                    hx_get=__self__/"sandbox_refresh/$name", hx_target="#$card_id", hx_swap="outerHTML"),
                h.button(expect_label; type="button", class="pdb-icon-btn u-text-xs",
                    hx_get=__self__/"sandbox_toggle_expect/$name", hx_target="#$card_id", hx_swap="outerHTML"),
                h.button("✕"; type="button", class="pdb-icon-btn pdb-icon-del",
                    hx_delete=__self__/"sandbox_delete/$name", hx_target="#$card_id", hx_swap="outerHTML",
                    hx_confirm="Delete snippet '$name'?"),
                h.a("⧉"; href=__self__/"sandbox_view/$name", target="_blank", class="pdb-icon-btn u-link-plain"),
            ),
            snippet_code_block(name, code, card_id),
            stanc_error_block(name),
        )
    end

    snippet_list() = h.div(; id="snippet-list", class="pdb-snippet-grid")(
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
    sandbox_gallery_card(item) = safely(; obj=__self__, req=__req__) do
        code        = item.code_string
        status      = sandbox_read_status(item.id)
        ok_pass     = status == "PASS"
        status_class = isnothing(status) ? "is-unknown" :
                       ok_pass ? "is-ok" : "is-error"
        status_text = isnothing(status) ? "?" :
                      ok_pass ? "PASS" : "FAIL"
        # Stan source is shown only when cached (`.jl.stan` written by
        # `snippet_card` on editor view, or by the recording flow).
        # Eval-into-`Main` is too expensive + state-polluting to run
        # 132× on every gallery render.
        stan_src = sandbox_read_stan(item.id)
        h.article(; class="htmxo-gallery-card", id=item.id)(
            h.h4(; class="htmxo-gallery-card-title")(
                h.a(item.title; href=__self__/"sandbox_view/$(item.id)", target="_blank"),
                h.span(status_text; class="htmxo-gallery-card-status $status_class"),
                stanc_badge(item.id),
            ),
            isempty(item.description) ? h.span() :
                h.p(item.description; class="htmxo-gallery-card-description"),
            h.h5("SLIC source"; class="htmxo-gallery-card-subheading"),
            h.pre(h.code(code; class="language-julia"); class="htmxo-gallery-card-code"),
            isnothing(stan_src) ? h.span() : h.div(
                h.h5("Generated Stan"; class="htmxo-gallery-card-subheading"),
                h.pre(h.code(stan_src; class="language-stan"); class="htmxo-gallery-card-code"),
            ),
            (!isnothing(status) && !ok_pass) ?
                h.p(; class="htmxo-gallery-card-error")(h.code(strip(status))) : h.span(),
            stanc_error_block(item.id),
        )
    end

    @get gallery = htmx(
        h.main(; class="container-fluid")(
            gallery_grid(_sandbox_gallery.items;
                section_titles=_sandbox_gallery.section_titles,
                card_renderer=sandbox_gallery_card),
        );
        extra_head=(htmxo_gallery_styles(), htmxo_syntax_head()...),
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
        h.div(; class="u-flex-wide")(
            h.h3("Saved Snippets"; class="u-m-0"),
            h.button("Compile Failures"; type="button", class="u-btn-xs",
                hx_get=__self__/"sandbox_compile_failures", hx_target="#compile-all-output", hx_swap="innerHTML"),
            h.button("Spot Check (5)"; type="button", class="u-btn-xs",
                hx_get=__self__/"sandbox_spot_check", hx_target="#compile-all-output", hx_swap="innerHTML"),
            h.button("Compile All"; type="button", class="u-btn-xs",
                hx_get=__self__/"sandbox_compile_all", hx_target="#compile-all-output", hx_swap="innerHTML"),
            h.button("Verify stanc"; type="button", class="u-btn-xs",
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
            [sandbox_output(result), h.template(snippet_list)]
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
            ex = Base.eval(mod, :(macroexpand($mod, $(QuoteNode(expr)))))
            push!(expanded, sprint(Base.show_unquoted, ex))
        end
        h.pre(h.code(join(expanded, "\n\n"); class="language-julia"))
    end

    # Per-snippet: returns (name, ok::Bool, msg::String, dt::Float64). On
    # success writes the PASS sidecar; on failure stores the one-line error
    # message in the FAIL sidecar (so the read-only gallery can still flag
    # which snippets are broken). The route safety wrapper isn't a fit here
    # because we want the BATCH summary to render even if some entries fail.
    _compile_one(name, code) = begin
        t0 = time()
        ok, msg = try
            r = sandbox_result(code)
            write(joinpath(sandbox_path, name * ".jl.stan"), r.code)
            true, ""
        catch e
            first(split(sprint(showerror, e), "\n"))::String |> m -> (false, m)
        end
        write(joinpath(sandbox_path, name * ".jl.status"), ok ? "PASS" : msg)
        (name=name, ok=ok, msg=msg, dt=time() - t0)
    end

    compile_snippets(snippets) = begin
        results = [_compile_one(name, code) for (name, code) in snippets]
        n_pass = count(r -> r.ok, results)
        n_fail = length(results) - n_pass
        total_time = sum(r -> r.dt, results; init=0.0)
        h.div(
            h.p(
                h.strong("$(n_pass)/$(length(results)) passed"),
                n_fail > 0 ? h.span(" ($n_fail failed)"; class="u-text-error") : "",
                h.span(" in $(round(total_time; digits=2))s"; class="u-text-muted"),
            ),
            [h.div(; class="pdb-snippet-result-row")(
                r.ok ? h.span("PASS"; class="u-text-success") : h.span("FAIL"; class="u-text-error"),
                " ", r.name,
                h.span(" $(round(r.dt*1000; digits=0))ms"; class="u-text-muted"),
                r.ok ? "" : h.span(" - ", r.msg; class="u-text-error u-text-sm"),
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
        sc = stanc_check(result.code)
        if sc.ok
            h.div(h.span("stanc: OK"; class="u-text-success"),
                isempty(sc.output) ? "" : h.pre(sc.output; class="u-text-sm u-scroll-y"))
        else
            h.div(h.span("stanc: FAIL"; class="u-text-error"),
                h.pre(sc.output; class="u-text-sm u-scroll-y u-pre-wrap"))
        end
    end

    # Batch stanc check: relies on the cached `.jl.stan` sidecar from a prior
    # compile run to avoid re-transpiling. Snippets without a cached `.stan`
    # are skipped (run "Compile All" / "Compile Failures" first).
    @get sandbox_stanc_all = begin
        snippets = sandbox_snippets()
        results = map(snippets) do (name, _)
            stan_src = sandbox_read_stan(name)
            if isnothing(stan_src)
                (name=name, ok=nothing, output="no cached .stan — compile first")
            else
                sc = stanc_check(stan_src)
                (name=name, ok=sc.ok, output=sc.output)
            end
        end
        verified = count(r -> r.ok === true, results)
        failed = count(r -> r.ok === false, results)
        skipped = count(r -> isnothing(r.ok), results)
        h.div(
            h.p(
                h.strong("stanc: $verified verified"),
                failed > 0 ? h.span(", $failed failed"; class="u-text-error") : "",
                skipped > 0 ? h.span(", $skipped skipped"; class="u-text-muted") : "",
            ),
            [h.div(; class="pdb-snippet-result-row")(
                isnothing(r.ok) ? h.span("SKIP"; class="u-text-muted") :
                r.ok ? h.span("OK"; class="u-text-success") : h.span("FAIL"; class="u-text-error"),
                " ", r.name,
                (!isnothing(r.ok) && !r.ok) ? h.span(" - ", r.output; class="u-text-error u-text-sm") : "",
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
