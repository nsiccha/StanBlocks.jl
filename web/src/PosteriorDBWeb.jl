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


# --- Stanc badge/error helpers (used in stanc @include, card.index, and gallery_card) ---

# Render a stanc pass/fail badge from a Bool. `nothing` → empty string (no sidecar yet).
stanc_badge(ok::Bool) = ok ?
    h.span("stanc ✓"; data_status="success",
        title="Stan compiler (stanc) accepted the generated code") :
    h.span("stanc ✗"; data_status="error",
        title="Stan compiler (stanc) rejected the generated code — see error below")
stanc_badge(::Nothing) = ""
stanc_badge(sidecar::AbstractString) = stanc_badge(sidecar == "OK")

# Render a stanc error block. Returns `""` for nothing/empty/"OK";
# otherwise renders the error output in a card-error div.
stanc_error_block(::Nothing) = ""
function stanc_error_block(sidecar::AbstractString)
    (isempty(sidecar) || sidecar == "OK") && return ""
    h.div(class="htmxo-card-error")(
        h.strong("stanc rejected the generated Stan code:"),
        h.pre(h.code(sidecar)),
    )
end


# --- Test helpers (throw on failure; route safety wrapper renders the error article) ---

transpile_check(post) = (code=stan_code(post),)

# Locate the `stanc` binary. Resolution order:
#   1. $STANC_PATH env var (explicit override)
#   2. BridgeStan's shipped binary (bin/stanc relative to its source path)
#   3. Hardcoded fallback for Niko's machine (last-resort, keeps old behavior)
function _stanc_bin()
    if haskey(ENV, "STANC_PATH")
        return ENV["STANC_PATH"]
    end
    bs_path = BridgeStan.get_bridgestan_path(; download=false)
    if !isempty(bs_path)
        candidate = joinpath(bs_path, "bin", "stanc")
        isfile(candidate) && return candidate
    end
    # Last-resort fallback (Niko's machine, cmdstan 2.37.0)
    "/home/niko/.cmdstan/cmdstan-2.37.0/bin/stanc"
end

# Run `stanc --warn-pedantic` on a stan source string and return
# `(ok::Bool, output::String)`. Pure utility on a string arg — used by
# multiple sandbox sites (live editor card, batch verification) where the
# stan source comes from different identities (live transpile vs cached
# sidecar), so it has no natural single-owner to attach to.
function stanc_check(stan_code_str::AbstractString)
    stanc = _stanc_bin()
    tmpfile = tempname() * ".stan"
    write(tmpfile, stan_code_str)
    io = IOBuffer()
    ok = success(pipeline(`$stanc --warn-pedantic $tmpfile`; stderr=io, stdout=io))
    rm(tmpfile; force=true)
    (ok=ok, output=String(take!(io)))
end

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

# --- Sandbox gallery (read-only view of `web/sandbox/`) ----------------------
#
# Reuses the existing on-disk sandbox dir as a read-only `HTMXObjects.Gallery`.
# The interactive `/sandbox` editor is left untouched — the gallery exposes
# the *same files* via the new `gallery_grid` primitive so the docs
# `record_gallery` flow can ship the entire sandbox as static recordings.
_SANDBOX_GALLERY_DIR = joinpath(dirname(dirname(@__DIR__)), "web", "sandbox")
_sandbox_gallery = Gallery(_SANDBOX_GALLERY_DIR)

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
            ["/sandbox/gallery"],
            ["/sandbox/snippet/$id" for id in ids],
        )

        phases = [prepare_progress!(__status__; description=p) for p in paths]

        isdir(record_dir) && rm(record_dir; recursive=true)
        mkpath(record_dir)

        # `HTMXObjects.record!` re-`route!`s the app with recording closures.
        # The live AppContext registration is clobbered while this loop runs;
        # restore it via `route!(app)` in the `finally` so subsequent live
        # requests keep going to the editor / live UI.
        app = AppContext()
        route!(app; record_dir, record_base)
        router = HTMXObjects.CONTEXT[].service.router
        try
            for (path, phase) in zip(paths, phases)
                with_prepared_progress(phase) do _
                    HTMXObjects._drive_record_path(router, path, Pair{String,String}[])
                    HTMXObjects._drive_record_path(router, path, ["HX-Request" => "true"])
                end
            end
        finally
            route!(app)
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

    # All per-posterior data, rendering, and routes. Mounted at
    # `/posterior/<name>/...`. Three sibling sub-DOs (transpile, compile,
    # correct) share shape (label, check_url, @cached result, status) but
    # have distinct check bodies. `check_action(kind)` dispatches via
    # `__self__` for shared cell/section/force rendering — keeps the render
    # logic in one place without losing per-check identity on the data side.
    @include posterior(name::String) = begin
        # `_convert_param(::AbstractString, ::Type{<:AbstractString})` is a
        # no-op in HTMXObjects, so `name::String` only affects
        # dispatch — the value still arrives as `SubString`. PosteriorDB
        # requires `String`.
        pdb_posterior = PosteriorDB.posterior(pdb(), String(name))

        detail_id = "detail-$name"
        toggle    = "on click toggle @hidden on #$detail_id"

        dataset_name, model_name = let parts = split(name, "-"; limit=2)
            length(parts) == 2 ? (parts[1], parts[2]) : (name, "")
        end

        @struct transpile = begin
            label     = "Transpiles"
            check_url = "/posterior/$name/check/transpile"
            @cached result = transpile_check(slic_implementation(pdb_posterior))
            status = @cache_status result
        end

        @struct compile = begin
            label     = "Compiles"
            check_url = "/posterior/$name/check/compile"
            @cached result = compile_check(slic_implementation(pdb_posterior))
            status = @cache_status result
        end

        @struct correct = begin
            label     = "Correct"
            check_url = "/posterior/$name/check/correct"
            @cached result = correct_check(pdb_posterior)
            status = @cache_status result
        end

        # Per-kind derivations: cell + section rendering, plus the
        # side-effecting `force` (clears stale `:started` cache, returns
        # result). All three share `(kind)` so they live in one bundle
        # instead of as flat siblings on the posterior.
        @struct check_action(kind::Symbol) = begin
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

            force = begin
                c.status == :started && @clear_cache! c.result
                c.result
            end
        end

        # Dashboard banner color: red if any check has actually failed; green
        # only when ALL three are ready; otherwise neutral (some unstarted).
        banner_status =
            any(s == :started for s in (transpile.status, compile.status, correct.status)) ? "error" :
            all(s == :ready   for s in (transpile.status, compile.status, correct.status)) ? "success" :
            "accent"

        summary_cells = [
            h.td(name; _=toggle),
            h.td(dataset_name; _=toggle),
            h.td(model_name; _=toggle),
            check_action(:transpile).cell,
            check_action(:compile).cell,
            check_action(:correct).cell,
        ]

        row = [
            h.tr(summary_cells...; id="row-$name"),
            h.tr(; id=detail_id, hidden=""),
        ]

        updated_summary = h.tr(summary_cells...;
            id="row-$name", hx_swap_oob="outerHTML:#row-$name")

        detail = h.td(; colspan="6")(
            h.div(; class="htmxo-status-banner", data_status=banner_status)(
                h.h4(name),
                check_action(:transpile).section,
                check_action(:compile).section,
                check_action(:correct).section,
            )
        )

        # Per-check route bundle: /posterior/<name>/check/<kind>. The IP
        # cache makes repeated requests idempotent (the underlying
        # `@cached result` re-runs only when the on-disk cache is cleared).
        @include check = begin
            do_check(kind::Symbol) = begin
                __parent__.check_action(kind).force
                [__parent__.detail, h.template(__parent__.updated_summary)]
            end
            @get transpile() = do_check(:transpile)
            @get compile()   = do_check(:compile)
            @get correct()   = do_check(:correct)
        end

        @get clear_cache() = begin
            @clear_cache! transpile.result
            @clear_cache! compile.result
            @clear_cache! correct.result
            h.p("Cache cleared for $name")
        end

        # Read reference Stan on-demand (only /ref route needs it, not the index).
        @get ref() = read(
            PosteriorDB.path(PosteriorDB.implementation(PosteriorDB.model(pdb_posterior), "stan")),
            String)

        @get model() = h.div(detail)
    end

    @get index() = h.div(
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
                h.nav(h.a("PosteriorDB"; href=__self__), " | ", h.a("Tests"; href=__self__/"tests"), " | ", h.a("Sandbox"; href=__self__/"sandbox"), " | ", h.a("Gallery"; href=__self__/"sandbox/gallery"), " | ", h.a("Structure"; href=__self__/"structure")),
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

                /* Wide-layout pages opt in by emitting their root with `data-layout="wide"`. */
                main.container:has([data-layout="wide"]) { max-width: 100%; padding: 0 1rem; }
                #content:has(> [data-layout="wide"])    { max-width: 100%; }
            """),
            htmxo_gallery_styles(),
            htmxo_syntax_head()...,
            h.script("document.addEventListener('DOMContentLoaded',function(){document.body.addEventListener('htmx:afterSettle',function(){document.querySelectorAll('code[class*=\"language-\"]').forEach(function(el){if(!el.querySelector('span.token')){Prism.highlightElement(el);}});});});"),
            sortable_table_js(),
            sortable_table_styles(),
        ),
    )

    # --- Test routes (via @include — registers under /tests/) ---
    @include tests = TestRoutes(; __req__, test_module=@__MODULE__)

    @get clear_cache() = begin
        foreach(rm, Base.filter(f -> endswith(f, ".sjl"), readdir(cache_path; join=true)))
        h.p("Cache cleared")
    end

    # DO/HTMXO type-tree introspection. `DynamicObjects.structure` returns a
    # `Node{:pre, …}` that's `Base.showable("text/html", …)` — Cobweb auto-
    # renders HTML-showable children, so embedding it directly works.
    @get structure() = h.div(
        h.h2("AppContext type structure"),
        h.p(h.small("DynamicObjects.structure(AppContext) — bond colors mark identical worst-case dependency sets")),
        DynamicObjects.structure(AppContext),
    )

    # Per-check-kind routes for batch operations across all posteriors.
    # Mounted at `/checks/<check>/{filter|recheck}[/<status>]`.
    @include checks(check::String) = begin
        kind = Symbol(check)

        names_with(status::String) = begin
            out = String[]
            for pn in __parent__.posterior_names
                s = getproperty(__parent__.posterior(pn), kind).status
                show = status == "pass"      ? s == :ready    :
                       status == "fail"      ? s == :started  :
                       status == "unchecked" ? s == :unstarted : true
                show && push!(out, pn)
            end
            out
        end

        @get filter(status::String="all") = join(names_with(status), "\n")

        # Batch recheck. Errors bubble through the route safety wrapper;
        # PASSes are reported in the response body.
        @get recheck(status::String="fail") = begin
            results = String[]
            for pn in names_with(status)
                __parent__.posterior(pn).check_action(kind).force
                push!(results, "PASS $pn")
            end
            join(results, "\n")
        end
    end

    # --- Sandbox: interactive SLIC editor + read-only gallery ---
    #
    # All sandbox-related state, helpers, and routes live in this `@include`
    # bundle. URLs mount under `/sandbox/...`; per-snippet routes live under
    # `/sandbox/snippet/<name>/...` via the indexed `@include snippet(name)`
    # sub-bundle. See `SbAppData.record_gallery` for the recording driver,
    # which hardcodes the public URL list.
    @include sandbox = begin
        path = joinpath(dirname(dirname(@__DIR__)), "web", "sandbox")

        snippets() = begin
            isdir(path) || mkpath(path)
            local out = Pair{String,String}[]
            for f in sort(readdir(path))
                endswith(f, ".jl") || continue
                local name = f[1:end-3]
                local code = read(joinpath(path, f), String)
                push!(out, name => code)
            end
            out
        end

        # Per-`code` derivations: transpile, render the result, render the
        # editor form. `result` evaluates user-typed SLIC source (throws on
        # parse / transpile failure — `snippet(name).card` wraps this in
        # `safely(; obj=__self__)` to contain errors).
        @struct draft(code) = begin
            result = begin
                exprs = Meta.parseall(code).args
                rv = nothing
                for expr in exprs
                    expr isa LineNumberNode && continue
                    rv = Base.eval(@__MODULE__, expr)
                end
                m = rv::StanBlocks.stan.SlicModel
                (code=Base.invokelatest(stan_code, m),)
            end

            output = h.div(; id="sandbox-output")(
                h.div(
                    h.p(h.strong("Transpilation: "), h.span("PASS"; data_status="success")),
                    h.pre(h.code(result.code; class="language-stan"); class="pdb-code-large"),
                ),
            )

            editor(; name="", target="#sandbox-output", standalone=false) = h.form(;
                hx_post=__parent__/"run", hx_target=target, hx_swap="outerHTML",
            )(
                name == "" ? "" : h.input(; name="name", type="hidden", value=name),
                standalone ? h.input(; name="standalone", type="hidden", value="1") : "",
                h.textarea(code; name="code", rows="15",
                    class="pdb-snippet-input",
                    _="on keydown[(shiftKey or ctrlKey) and key is 'Enter'] halt the event send submit to closest <form/> on keydown[key is 'Escape'] halt the event set editor to closest .pdb-snippet-editor add @hidden to editor remove @hidden from previous <pre/> from editor"),
            )
        end

        default_slic_code = """@slic (;y=randn(10)) begin
        mu ~ normal(0, 1)
        sigma ~ gamma(1, 1)
        y ~ normal(mu, sigma)
    end"""

        # Per-snippet bundle. Owns ALL per-name derivations and per-snippet
        # routes (mounted at `/sandbox/snippet/<name>/...`).
        @include snippet(name::String) = begin
            file_path = joinpath(__parent__.path, name * ".jl")
            code = read(file_path, String)

            read_sidecar(suffix; default=nothing) = let p = "$file_path.$suffix"
                isfile(p) ? read(p, String) : default
            end
            status = read_sidecar("status")
            stan_sidecar = read_sidecar("stan")
            expect = read_sidecar("expect"; default="pass")
            should_fail = expect == "fail"

            # Delegate draft(code) to sandbox so sub-structs (card, stanc) can
            # reach it via a single __parent__ instead of __parent__.__parent__.
            draft(code) = __parent__.draft(code)

            # Stanc verification: cached sidecar (read at construction), the
            # render fragments that depend on it, and the live re-run route.
            # Mounted at `/sandbox/snippet/<name>/stanc`.
            @include stanc = begin
                sidecar = read_sidecar("stanc")

                badge       = stanc_badge(sidecar)
                error_block = stanc_error_block(sidecar)

                @get index() = begin
                    r = __parent__.draft(code).result
                    sc = stanc_check(r.code)
                    if sc.ok
                        h.div(h.span("stanc: OK"; data_status="success"),
                            isempty(sc.output) ? "" : h.pre(sc.output; class="pdb-stanc-output"))
                    else
                        h.div(h.span("stanc: FAIL"; data_status="error"),
                            h.pre(sc.output; class="pdb-stanc-output"))
                    end
                end
            end

            # Sort key: 0 unexpected fail, 1 untried, 2 stanc fails, 3 pass, 4 xfail.
            sort_key = begin
                ok = isnothing(status) ? nothing : status == "PASS"
                stanc_ok = isnothing(stanc.sidecar) ? nothing : stanc.sidecar == "OK"
                if !isnothing(ok) && !ok && !should_fail
                    0
                elseif isnothing(ok)
                    1
                elseif ok && !isnothing(stanc_ok) && !stanc_ok
                    2
                elseif ok
                    3
                else
                    4
                end
            end

            # Card rendering: id (DOM id), the cheap `lazy` variant (cached
            # sidecar state, no re-eval), the heavy `full` variant
            # (re-evaluates, writes status/stan/stanc sidecars). Both
            # variants share `id`, `header_buttons`, and the editor block.
            @struct card = begin
                id = "snippet-$name"
                snippet_url = __parent__   # the snippet's URL (parent-prefix)

                code_block(; standalone=false) = h.div(
                    h.pre(h.code(code; class="language-julia");
                        class="pdb-snippet-output",
                        title="Ctrl+click to edit",
                        _="on click[ctrlKey or shiftKey] halt the event set editor to the next .pdb-snippet-editor add @hidden to me remove @hidden from editor focus() the first <textarea/> in editor"),
                    h.div(; class="pdb-snippet-editor", hidden="")(
                        __parent__.draft(code).editor(; name, target="#$id", standalone),
                    ),
                )

                # Heavy: re-evaluates code, writes status/stan/stanc sidecars,
                # renders the full card. Wraps eval in `safely` so transpile
                # errors render as a contained error article.
                index(; standalone=false) = safely(; obj=__parent__) do
                    d = __parent__.draft(code)
                    write("$file_path.status", "PASS")
                    write("$file_path.stan", d.result.code)
                    sc = stanc_check(d.result.code)
                    write("$file_path.stanc", sc.ok ? "OK" : sc.output)
                    refresh_url = standalone ? snippet_url : snippet_url/"refresh"
                    badge = should_fail ?
                        h.span("XPASS"; data_status="warning",
                            title="Expected failure but transpiled — toggle 'should pass' if intentional") :
                        h.span("PASS"; data_status="success")
                    expect_label = should_fail ? "xfail" : "should pass"
                    h.div(; id, class="htmxo-status-banner", data_status="success")(
                        h.div(; class="pdb-snippet-header")(
                            h.strong(name; contenteditable="true", class="pdb-name-input",
                                _="on blur if my textContent.trim() is not '$name' fetch $(snippet_url/"rename")?to=\${my.textContent.trim()} then put the result into #$id.outerHTML"),
                            badge, stanc_badge(sc.ok),
                            h.button("↻"; type="button", class="pdb-icon-btn",
                                hx_get=refresh_url, hx_target="#$id", hx_swap="outerHTML"),
                            h.button(expect_label; type="button", class="pdb-icon-btn", data_variant="text",
                                hx_get=snippet_url/"toggle_expect", hx_target="#$id", hx_swap="outerHTML"),
                            h.button("✕"; type="button", class="pdb-icon-btn", data_variant="del",
                                hx_delete=snippet_url/"delete", hx_target="#$id", hx_swap="outerHTML",
                                hx_confirm="Delete snippet '$name'?"),
                            standalone ? "" : h.a("⧉"; href=snippet_url, target="_blank", class="pdb-icon-link"),
                            h.a("macro"; href=snippet_url/"macroexpand", target="_blank", class="pdb-icon-link", title="Show macroexpanded code"),
                        ),
                        code_block(; standalone),
                        stanc_error_block(sc.ok ? "" : sc.output),
                        standalone ? d.output :
                        h.details(
                            h.summary("Stan code"),
                            d.output,
                        ),
                    )
                end

                # Cheap: renders cached sidecar state, no re-eval. Used by
                # the gallery grid and the editor's snippet list.
                lazy = begin
                    banner_status = isnothing(status) ? "accent" :
                        status == "PASS" ? "success" :
                        should_fail ? "warning" : "error"
                    badge = isnothing(status) ? "" :
                        status == "PASS" ? h.span("PASS"; data_status="success") :
                        should_fail ? h.span("XFAIL"; data_status="warning") :
                        h.span("FAIL"; data_status="error", title=status)
                    expect_label = should_fail ? "xfail" : "should pass"
                    h.div(; id, class="htmxo-status-banner", data_status=banner_status)(
                        h.div(; class="pdb-snippet-header")(
                            h.strong(name),
                            badge, stanc.badge,
                            h.button("↻"; type="button", class="pdb-icon-btn",
                                hx_get=snippet_url/"refresh", hx_target="#$id", hx_swap="outerHTML"),
                            h.button(expect_label; type="button", class="pdb-icon-btn", data_variant="text",
                                hx_get=snippet_url/"toggle_expect", hx_target="#$id", hx_swap="outerHTML"),
                            h.button("✕"; type="button", class="pdb-icon-btn", data_variant="del",
                                hx_delete=snippet_url/"delete", hx_target="#$id", hx_swap="outerHTML",
                                hx_confirm="Delete snippet '$name'?"),
                            h.a("⧉"; href=snippet_url, target="_blank", class="pdb-icon-link"),
                        ),
                        code_block(),
                        stanc.error_block,
                    )
                end
            end

            @get index() = card.index(standalone=true)
            @get refresh() = card.index()

            @get macroexpand() = begin
                exprs = Meta.parseall(code).args
                mod = @__MODULE__
                expanded = String[]
                for expr in exprs
                    expr isa LineNumberNode && continue
                    ex = macroexpand(mod, expr)
                    push!(expanded, sprint(Base.show_unquoted, ex))
                end
                h.pre(h.code(join(expanded, "\n\n"); class="language-julia"))
            end

            @get rename(; to="") = begin
                local target = strip(to)
                isempty(target) && return card.index()
                local new_path = joinpath(__parent__.path, target * ".jl")
                isfile(file_path) && mv(file_path, new_path; force=true)
                __parent__.snippet(String(target)).card.index()
            end

            @get toggle_expect() = begin
                ep = "$file_path.expect"
                write(ep, expect == "fail" ? "pass" : "fail")
                __parent__.snippet(name).card.lazy
            end

            @delete delete() = begin
                for suffix in ("", ".status", ".expect", ".stanc")
                    fp = file_path * suffix
                    isfile(fp) && rm(fp)
                end
                ""
            end
        end

        # Read-only gallery card renderer — reads sidecar state via the
        # per-snippet bundle. Stan source is shown only when cached
        # (`.jl.stan` written by `snippet(name).card` on editor view, or by
        # the recording flow). Eval-into-`Main` is too expensive +
        # state-polluting to run 132× on every gallery render.
        gallery_card(item) = safely(; obj=__self__, req=__req__) do
            s = snippet(item.id)
            ok_pass = s.status == "PASS"
            status_data = isnothing(s.status) ? "muted" :
                          ok_pass ? "success" : "error"
            status_text = isnothing(s.status) ? "?" :
                          ok_pass ? "PASS" : "FAIL"
            h.article(; id=item.id)(
                h.h4(
                    h.a(item.title; href=__self__/"snippet/$(item.id)", target="_blank"),
                    h.span(status_text; data_status=status_data),
                    s.stanc.badge,
                ),
                isempty(item.description) ? h.span() : h.p(item.description),
                h.h5("SLIC source"),
                h.pre(h.code(s.code; class="language-julia")),
                isnothing(s.stan_sidecar) ? h.span() : h.div(
                    h.h5("Generated Stan"),
                    h.pre(h.code(s.stan_sidecar; class="language-stan")),
                ),
                (!isnothing(s.status) && !ok_pass) ?
                    h.p(; class="htmxo-card-error")(h.code(strip(s.status))) : h.span(),
                s.stanc.error_block,
            )
        end

        list() = h.div(; id="snippet-list", class="pdb-snippet-grid")(
            [snippet(name).card.lazy for (name, _) in sort(snippets(), by=p -> (snippet(first(p)).sort_key, first(p)))]...
        )

        # Batch compile bundle. Mounted at `/sandbox/compile/...`.
        # `one(name, code)` writes the PASS sidecar on success or the
        # one-line error message in the FAIL sidecar; `summary(items)`
        # renders the batch summary (the route safety wrapper isn't a fit
        # for individual entries — we want the summary to render even if
        # some entries fail).
        @include compile = begin
            one(name, code) = begin
                t0 = time()
                ok, msg = try
                    r = __parent__.draft(code).result
                    write(joinpath(__parent__.path, name * ".jl.stan"), r.code)
                    true, ""
                catch e
                    false, String(first(split(sprint(showerror, e), "\n")))
                end
                write(joinpath(__parent__.path, name * ".jl.status"), ok ? "PASS" : msg)
                (name=name, ok=ok, msg=msg, dt=time() - t0)
            end

            summary(items) = begin
                results = [one(n, c) for (n, c) in items]
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

            @get all() = summary(__parent__.snippets())

            @get failures() = begin
                to_compile = Base.filter(__parent__.snippets()) do (n, _)
                    s = __parent__.snippet(n)
                    isnothing(s.status) || (s.status != "PASS" && !s.should_fail)
                end
                isempty(to_compile) ? h.p("No unexpected failures or untried snippets") : summary(to_compile)
            end
        end

        @get index() = h.div(; data_layout="wide")(
            h.h2("SLIC Sandbox"),
            h.div(; class="pdb-sandbox-toolbar")(
                h.h3("Saved Snippets"),
                h.button("Compile Failures"; type="button",
                    hx_get=__self__/"compile/failures", hx_target="#compile-all-output", hx_swap="innerHTML"),
                h.button("Spot Check (5)"; type="button",
                    hx_get=__self__/"spot_check", hx_target="#compile-all-output", hx_swap="innerHTML"),
                h.button("Compile All"; type="button",
                    hx_get=__self__/"compile/all", hx_target="#compile-all-output", hx_swap="innerHTML"),
                h.button("Verify stanc"; type="button",
                    hx_get=__self__/"stanc_all", hx_target="#compile-all-output", hx_swap="innerHTML"),
            ),
            list(),
            h.div(; id="compile-all-output")(""),
            h.h3("New"),
            draft(default_slic_code).editor(),
            h.div(; id="sandbox-output")(""),
        )

        @post run(; code="", name="", standalone="") = begin
            local from_card = !isempty(strip(name))
            local snippet_name = String(strip(name))
            if isempty(snippet_name)
                local m = match(r"@slic\s+(?:\([^)]*\)\s+)?begin\s*\n\s*(\w+)", code)
                local base = isnothing(m) ? "snippet" : m[1]
                local existing = first.(snippets())
                snippet_name = base
                local i = 1
                while snippet_name in existing
                    i += 1
                    snippet_name = "$(base)_$i"
                end
            end
            isdir(path) || mkpath(path)
            write(joinpath(path, snippet_name * ".jl"), code)
            if from_card
                snippet(snippet_name).card.index(standalone=standalone=="1")
            else
                [draft(code).output, h.template(list())]
            end
        end

        @get refresh_all() = list()

        @get spot_check(; n="5") = begin
            num = parse(Int, n)
            passes = Base.filter(snippets()) do (nm, _)
                s = snippet(nm).status
                !isnothing(s) && s == "PASS"
            end
            sample = passes[Random.randperm(length(passes))[1:min(num, length(passes))]]
            isempty(sample) ? h.p("No passing snippets to spot check") : compile.summary(sample)
        end

        @get stanc_all() = begin
            items = snippets()
            results = map(items) do (nm, _)
                s = snippet(nm)
                if isnothing(s.stan_sidecar)
                    (name=nm, ok=nothing, output="no cached .stan — compile first")
                else
                    sc = stanc_check(s.stan_sidecar)
                    (name=nm, ok=sc.ok, output=sc.output)
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

        # Read-only view of `web/sandbox/` rendered via
        # `HTMXObjects.gallery_grid`. Same files as `/sandbox` but no editor
        # / CRUD; this is what the docs `record_gallery` flow ships as
        # static recordings. `__page__` provides the chrome and gallery
        # styles; this route returns bare content with `data-layout="wide"`.
        @get gallery() = h.div(; data_layout="wide")(
            gallery_grid(_sandbox_gallery.items;
                section_titles=_sandbox_gallery.section_titles,
                card_renderer=gallery_card),
        )
    end

    # `/record_gallery` — drive the AppData IP `record_gallery` to dump
    # `/sandbox/gallery` + every `/sandbox/snippet/<id>` URL into
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

end

function __init__()
    route!(AppContext())
end

end # module PosteriorDBWeb
