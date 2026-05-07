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

    # All per-posterior data + rendering. Three sibling sub-DOs (transpile,
    # compile, correct) share shape (label, check_url, @cached result, status)
    # but have distinct check bodies. The indexed `view(kind)` dispatches via
    # `__parent__` for shared rendering (cell, section) — keeps the render
    # logic in one place without losing per-check identity on the data side.
    @struct posterior(pn) = begin
        pdb_posterior = PosteriorDB.posterior(pdb(), pn)

        detail_id = "detail-$pn"
        toggle    = "on click toggle @hidden on #$detail_id"

        dataset_name, model_name = split_posterior_name(pn)

        @struct transpile = begin
            label     = "Transpiles"
            check_url = "/check_transpile/$pn"
            @cached result = transpile_check(slic_implementation(pdb_posterior))
            status = @cache_status result
        end

        @struct compile = begin
            label     = "Compiles"
            check_url = "/check_compile/$pn"
            @cached result = compile_check(slic_implementation(pdb_posterior))
            status = @cache_status result
        end

        @struct correct = begin
            label     = "Correct"
            check_url = "/check_correct/$pn"
            @cached result = correct_check(pdb_posterior)
            status = @cache_status result
        end

        # Rendering for one check kind (`:transpile` | `:compile` | `:correct`).
        # Dispatches via `__parent__` so cell + section live in one place. The
        # check sub-DOs (above) own the data identity; this IP owns the view
        # derivations.
        @struct view(kind::Symbol) = begin
            c        = getproperty(__parent__, kind)
            target   = "#$(__parent__.detail_id)"
            on_load  = "on htmx:afterOnLoad if not me.hasAttribute('data-batch') then remove @hidden from $target end remove @data-batch from me"

            cell = if c.status == :unstarted
                h.td("-"; data_status="muted",
                    hx_get=c.check_url, hx_target=target, hx_swap="innerHTML", _=on_load)
            elseif c.status == :started
                h.td("FAIL"; data_status="error",
                    hx_get=c.check_url, hx_target=target, hx_swap="innerHTML", _=on_load)
            else
                h.td("PASS"; data_status="success",
                    _="on click toggle @hidden on $target")
            end

            section = if c.status == :unstarted
                ""
            elseif c.status == :started
                h.section(
                    h.p(h.strong(c.label, ": "), status_badge(:failed; label="FAIL"), " ",
                        h.a("Re-run to view error"; hx_get=c.check_url, hx_target="closest tr", hx_swap="outerHTML")),
                )
            else
                h.section(
                    h.p(h.strong(c.label, ": "), status_badge(:done; label="PASS")),
                    hasproperty(c.result, :code) ? h.details(
                        h.summary("Generated Stan code"),
                        h.pre(c.result.code; class="pdb-stan-output");
                        open=""
                    ) : "",
                    hasproperty(c.result, :stats) ? h.p(
                        h.strong("Stats: "),
                        "abs diff = $(c.result.stats.absolute_constant_difference), rel diff = $(c.result.stats.relative_remaining_difference)"
                    ) : "",
                )
            end

        end

        # Dashboard banner color: red if any check has actually failed; green
        # only when ALL three are ready; otherwise neutral (some unstarted).
        banner_status =
            any(s == :started for s in (transpile.status, compile.status, correct.status)) ? "error" :
            all(s == :ready   for s in (transpile.status, compile.status, correct.status)) ? "success" :
            "accent"

        summary_cells = [
            h.td(pn; _=toggle),
            h.td(dataset_name; _=toggle),
            h.td(model_name; _=toggle),
            view(:transpile).cell,
            view(:compile).cell,
            view(:correct).cell,
        ]

        row = [
            h.tr(summary_cells...; id="row-$pn"),
            h.tr(; id=detail_id, hidden=""),
        ]

        updated_summary = h.tr(summary_cells...;
            id="row-$pn", hx_swap_oob="outerHTML:#row-$pn")

        detail = h.td(; colspan="6")(
            h.div(; class="htmxo-status-banner", data_status=banner_status)(
                h.h4(pn),
                view(:transpile).section,
                view(:compile).section,
                view(:correct).section,
            )
        )

        reference_stan = read(
            PosteriorDB.path(PosteriorDB.implementation(PosteriorDB.model(pdb_posterior), "stan")),
            String)

        # Force (re)compute one check, clearing the empty `:started` marker
        # left behind by a previously-failed run so the next access recomputes.
        force_check(kind::Symbol) = begin
            c = getproperty(__self__, kind)
            c.status == :started && @clear_cache! c.result
            c.result
        end

        clear_cache!() = begin
            @clear_cache! transpile.result
            @clear_cache! compile.result
            @clear_cache! correct.result
            nothing
        end
    end

    @get index = h.div(
        h.h2("PosteriorDB Models ($(length(posterior_names)) implemented)"),
        h.input(;
            type="search",
            id="search",
            placeholder="Filter posteriors...",
            _="on input set query to my value.toLowerCase() for row in <tr/> in #posterior-tbody if row.textContent.toLowerCase() contains query remove @hidden from row else add @hidden to row end end on keydown[key is 'Enter'] halt the event for row in <tr[id^='row-']/> in #posterior-tbody if row matches ':not([hidden])' set target to null for cell in <td[data-status]:not([data-status='success'])/> in row if target is null set target to cell end end if target is not null add @data-batch to target send click to target end end end",
        ),
        h.table(class="striped htmxo-sortable-table"; role="grid")(
            h.thead(
                h.tr(
                    h.th("Posterior"; _="on click call sortTable(0, me)"),
                    h.th("Dataset"; _="on click call sortTable(1, me)"),
                    h.th("Model"; _="on click call sortTable(2, me)"),
                    h.th("Transpiles"; _="on click call sortTable(3, me)"),
                    h.th("Compiles"; _="on click call sortTable(4, me)"),
                    h.th("Correct"; _="on click call sortTable(5, me)"),
                )
            ),
            h.tbody(reduce(vcat, [posterior(pn).row for pn in posterior_names]; init=[])...; id="posterior-tbody")
        ),
    )

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

                .htmxo-status-banner section { margin-bottom: 0.5rem; }
                .pdb-stan-output { white-space: pre-wrap; max-height: 400px; overflow: auto; font-size: 0.85em; }

                .pdb-sandbox-toolbar { display: flex; align-items: center; gap: 1rem; flex-wrap: wrap; }
                .pdb-sandbox-toolbar > h3 { margin: 0; }
                .pdb-sandbox-toolbar > button { padding: 0.2rem 0.6rem; }

                .pdb-code-large { max-height: 600px; overflow: auto; }
                .pdb-snippet-input { font-family: monospace; width: 100%; resize: vertical; margin: 0; }
                .pdb-snippet-output { max-height: 200px; overflow: auto; cursor: default; }
                .pdb-snippet-header { display: flex; align-items: center; gap: 0.5rem; flex-wrap: wrap; }
                .pdb-name-input { outline: none; min-width: 3em; }
                .pdb-icon-btn { margin: 0; padding: 0.1rem 0.4rem; }
                .pdb-icon-btn[data-variant="text"] { font-size: 0.75em; }
                .pdb-icon-btn[data-variant="del"]  { color: var(--pico-del-color); }
                .pdb-icon-link { text-decoration: none; color: inherit; padding: 0.1rem 0.4rem; }
                .pdb-snippet-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 0.5rem; }
                .pdb-snippet-result-row { padding: 0.1rem 0; font-family: monospace; font-size: 0.85em; }
                .pdb-stanc-output { font-size: 0.85em; max-height: 300px; overflow: auto; white-space: pre-wrap; }
            """),
            h.link(rel="stylesheet", href="https://cdn.jsdelivr.net/npm/prismjs@1/themes/prism-tomorrow.min.css"),
            h.script(src="https://cdn.jsdelivr.net/npm/prismjs@1/prism.min.js"),
            h.script(src="https://cdn.jsdelivr.net/npm/prismjs@1/components/prism-julia.min.js"),
            h.script(src="https://cdn.jsdelivr.net/npm/prismjs@1/components/prism-stan.min.js"),
            h.script("document.addEventListener('DOMContentLoaded',function(){document.body.addEventListener('htmx:afterSettle',function(){document.querySelectorAll('code[class*=\"language-\"]').forEach(function(el){if(!el.querySelector('span.token')){Prism.highlightElement(el);}});});});"),
            sortable_table_js(),
            sortable_table_styles(),
        ),
    )

    # --- Test routes (via @include — registers under /tests/) ---
    @include tests = TestRoutes(; __req__, test_module=@__MODULE__)

    @get clear_cache = begin
        foreach(rm, filter(f -> endswith(f, ".sjl"), readdir(cache_path; join=true)))
        h.p("Cache cleared")
    end

    @get clear_cache_model(pn) = begin
        posterior(pn).clear_cache!()
        h.p("Cache cleared for $pn")
    end

    @get check_transpile(pn) = begin
        p = posterior(pn)
        p.force_check(:transpile)
        [p.detail, h.template(p.updated_summary)]
    end

    @get check_compile(pn) = begin
        p = posterior(pn)
        p.force_check(:compile)
        [p.detail, h.template(p.updated_summary)]
    end

    @get check_correct(pn) = begin
        p = posterior(pn)
        p.force_check(:correct)
        [p.detail, h.template(p.updated_summary)]
    end

    @get ref(pn) = posterior(pn).reference_stan

    filtered_names(check, status) = begin
        kind = Symbol(check)
        names = String[]
        for pn in posterior_names
            s = getproperty(posterior(pn), kind).status
            show = status == "pass" ? s == :ready :
                   status == "fail" ? s == :started :
                   status == "unchecked" ? s == :unstarted : true
            show && push!(names, pn)
        end
        names
    end

    @get filter(check, status="all") = join(filtered_names(check, status), "\n")

    # Batch recheck: /recheck/{check}/{status}. Errors bubble through the
    # route safety wrapper; PASSes are reported in the response body.
    @get recheck(check, status="fail") = begin
        kind = Symbol(check)
        results = String[]
        for pn in filtered_names(check, status)
            posterior(pn).force_check(kind)
            push!(results, "PASS $pn")
        end
        join(results, "\n")
    end

    @get model(pn) = h.div(posterior(pn).detail)

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
            h.p(h.strong("Transpilation: "), h.span("PASS"; data_status="success")),
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
            _="on keydown[(shiftKey or ctrlKey) and key is 'Enter'] halt the event send submit to closest <form/> on keydown[key is 'Escape'] halt the event set editor to closest .pdb-snippet-editor add @hidden to editor remove @hidden from previous <pre/> from editor"),
    )

    snippet_code_block(name, code, card_id; standalone=false) = h.div(
        h.pre(h.code(code; class="language-julia");
            class="pdb-snippet-output",
            title="Ctrl+click to edit",
            _="on click[ctrlKey or shiftKey] halt the event set editor to the next .pdb-snippet-editor add @hidden to me remove @hidden from editor focus() the first <textarea/> in editor"),
        h.div(; class="pdb-snippet-editor", hidden="")(
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
            h.span("stanc ✓"; data_status="success",
                title="Stan compiler (stanc) accepted the generated code") :
            h.span("stanc ✗"; data_status="error",
                title="Stan compiler (stanc) rejected the generated code — see error below")
    end

    stanc_error_block(name) = begin
        sc = sandbox_read_stanc(name)
        (isnothing(sc) || sc == "OK") && return ""
        h.div(class="htmxo-card-error")(
            h.strong("stanc rejected the generated Stan code:"),
            h.pre(h.code(sc)),
        )
    end

    snippet_card(name, code; standalone=false) = safely(; obj=__self__) do
        result = sandbox_result(code)
        write(joinpath(sandbox_path, name * ".jl.status"), "PASS")
        write(joinpath(sandbox_path, name * ".jl.stan"), result.code)
        sc = stanc_check(result.code)
        write(joinpath(sandbox_path, name * ".jl.stanc"), sc.ok ? "OK" : sc.output)
        should_fail = sandbox_should_fail(name)
        card_id = "snippet-$name"
        refresh_url = standalone ? __self__/"sandbox_view/$name" : __self__/"sandbox_refresh/$name"
        badge = should_fail ? h.span("XPASS"; data_status="warning",
                                            title="Expected failure but transpiled — toggle 'should pass' if intentional") :
                                     h.span("PASS"; data_status="success")
        expect_label = should_fail ? "xfail" : "should pass"
        h.div(; id=card_id, class="htmxo-status-banner", data_status="success")(
            h.div(; class="pdb-snippet-header")(
                h.strong(name; contenteditable="true", class="pdb-name-input",
                    _="on blur if my textContent.trim() is not '$name' fetch /sandbox_rename/$name?to=\${my.textContent.trim()} then put the result into #$card_id.outerHTML"),
                badge, stanc_badge(name),
                h.button("↻"; type="button", class="pdb-icon-btn",
                    hx_get=refresh_url, hx_target="#$card_id", hx_swap="outerHTML"),
                h.button(expect_label; type="button", class="pdb-icon-btn", data_variant="text",
                    hx_get=__self__/"sandbox_toggle_expect/$name", hx_target="#$card_id", hx_swap="outerHTML"),
                h.button("✕"; type="button", class="pdb-icon-btn", data_variant="del",
                    hx_delete=__self__/"sandbox_delete/$name", hx_target="#$card_id", hx_swap="outerHTML",
                    hx_confirm="Delete snippet '$name'?"),
                standalone ? "" : h.a("⧉"; href=__self__/"sandbox_view/$name", target="_blank", class="pdb-icon-link"),
                h.a("macro"; href=__self__/"sandbox_macroexpand/$name", target="_blank", class="pdb-icon-link", title="Show macroexpanded code"),
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
        banner_status = isnothing(status) ? "accent" :
            status == "PASS" ? "success" :
            should_fail ? "warning" : "error"
        badge = isnothing(status) ? "" :
            status == "PASS" ? h.span("PASS"; data_status="success") :
            should_fail ? h.span("XFAIL"; data_status="warning") :
            h.span("FAIL"; data_status="error", title=status)
        expect_label = should_fail ? "xfail" : "should pass"
        h.div(; id=card_id, class="htmxo-status-banner", data_status=banner_status)(
            h.div(; class="pdb-snippet-header")(
                h.strong(name),
                badge, stanc_badge(name),
                h.button("↻"; type="button", class="pdb-icon-btn",
                    hx_get=__self__/"sandbox_refresh/$name", hx_target="#$card_id", hx_swap="outerHTML"),
                h.button(expect_label; type="button", class="pdb-icon-btn", data_variant="text",
                    hx_get=__self__/"sandbox_toggle_expect/$name", hx_target="#$card_id", hx_swap="outerHTML"),
                h.button("✕"; type="button", class="pdb-icon-btn", data_variant="del",
                    hx_delete=__self__/"sandbox_delete/$name", hx_target="#$card_id", hx_swap="outerHTML",
                    hx_confirm="Delete snippet '$name'?"),
                h.a("⧉"; href=__self__/"sandbox_view/$name", target="_blank", class="pdb-icon-link"),
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
        status_data = isnothing(status) ? "muted" :
                      ok_pass ? "success" : "error"
        status_text = isnothing(status) ? "?" :
                      ok_pass ? "PASS" : "FAIL"
        # Stan source is shown only when cached (`.jl.stan` written by
        # `snippet_card` on editor view, or by the recording flow).
        # Eval-into-`Main` is too expensive + state-polluting to run
        # 132× on every gallery render.
        stan_src = sandbox_read_stan(item.id)
        h.article(; id=item.id)(
            h.h4(
                h.a(item.title; href=__self__/"sandbox_view/$(item.id)", target="_blank"),
                h.span(status_text; data_status=status_data),
                stanc_badge(item.id),
            ),
            isempty(item.description) ? h.span() : h.p(item.description),
            h.h5("SLIC source"),
            h.pre(h.code(code; class="language-julia")),
            isnothing(stan_src) ? h.span() : h.div(
                h.h5("Generated Stan"),
                h.pre(h.code(stan_src; class="language-stan")),
            ),
            (!isnothing(status) && !ok_pass) ?
                h.p(; class="htmxo-card-error")(h.code(strip(status))) : h.span(),
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
        h.div(; class="pdb-sandbox-toolbar")(
            h.h3("Saved Snippets"),
            h.button("Compile Failures"; type="button",
                hx_get=__self__/"sandbox_compile_failures", hx_target="#compile-all-output", hx_swap="innerHTML"),
            h.button("Spot Check (5)"; type="button",
                hx_get=__self__/"sandbox_spot_check", hx_target="#compile-all-output", hx_swap="innerHTML"),
            h.button("Compile All"; type="button",
                hx_get=__self__/"sandbox_compile_all", hx_target="#compile-all-output", hx_swap="innerHTML"),
            h.button("Verify stanc"; type="button",
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
                n_fail > 0 ? h.span(" ($n_fail failed)"; data_status="error") : "",
                h.small(" in $(round(total_time; digits=2))s"),
            ),
            [h.div(; class="pdb-snippet-result-row")(
                h.span(r.ok ? "PASS" : "FAIL"; data_status=r.ok ? "success" : "error"),
                " ", r.name,
                h.small(" $(round(r.dt*1000; digits=0))ms"),
                r.ok ? "" : h.small(" - ", r.msg; data_status="error"),
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
            h.div(h.span("stanc: OK"; data_status="success"),
                isempty(sc.output) ? "" : h.pre(sc.output; class="pdb-stanc-output"))
        else
            h.div(h.span("stanc: FAIL"; data_status="error"),
                h.pre(sc.output; class="pdb-stanc-output"))
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
                failed > 0 ? h.span(", $failed failed"; data_status="error") : "",
                skipped > 0 ? h.span(", $skipped skipped"; data_status="muted") : "",
            ),
            [h.div(; class="pdb-snippet-result-row")(
                isnothing(r.ok) ? h.span("SKIP"; data_status="muted") :
                r.ok ? h.span("OK"; data_status="success") : h.span("FAIL"; data_status="error"),
                " ", r.name,
                (!isnothing(r.ok) && !r.ok) ? h.small(" - ", r.output; data_status="error") : "",
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
