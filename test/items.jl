using TestItemRunner

@testsnippet StanBlocksImports begin
    using Random, Statistics
    using StanBlocks
    using LogDensityProblems
    # BridgeStan directly (not just through `instantiate`) for gq draw assertions:
    # `param_names(...; include_gq=true)` + `param_constrain(...; rng=…)`.
    import BridgeStan
    import StanBlocks.stan: @deffun, full_cqual_eq, transpiles, stan_model, stan_code, stanc_check, instantiate
    # View usertypes are not exported; alias so @deffun signatures can name them.
    const RaggedVector = StanBlocks.RaggedVector
    using PosteriorDB
end

@testmodule StanBlocksTestSetup begin
    using Random, Statistics
    using StanBlocks
    using LogDensityProblems
    # BridgeStan directly (not just through `instantiate`) for gq draw assertions:
    # `param_names(...; include_gq=true)` + `param_constrain(...; rng=…)`.
    import BridgeStan
    import StanBlocks.stan: @deffun, full_cqual_eq, transpiles, stan_model, stan_code, stanc_check, instantiate
    # View usertypes are not exported; alias so @deffun signatures can name them.
    const RaggedVector = StanBlocks.RaggedVector
    using PosteriorDB

    function stanc_compiles(model; re::Bool=true)
        result = stanc_check(stan_code(model); warn_pedantic=false)
        result.ok && return true
        re && error("stanc rejected generated Stan:\n$(result.output)")
        return false
    end


    # ──────────────────────────────────────────────────────────────────────────────
    # Module-level definitions needed by tests (structs, @deffun, sub-models, etc.)
    # ──────────────────────────────────────────────────────────────────────────────

    # --- slic.jl helpers ---

    msg(e::ErrorException) = e.msg
    msg(e::AssertionError) = e.msg
    msg(e::MethodError) = e.msg

    # `@lpxf` opts a `_lpdf`-named @deffun into the base-callable + lpxf/rng/likelihood
    # triad registration: it binds the base fn (`simple`, `vararg`, `fof`, `srs2`) and
    # wires `lpxf_expr`/`rng_expr`/`likelihood_expr` so the name is usable both as a
    # `y ~ simple(...)` distribution AND as a `::typeof(simple)` dispatch tag below.
    # This registration was NAME-driven until 31772b4 (2026-04-24, "Add @lhs opt-in")
    # made it an explicit `@lpxf`/`@lhs` opt-in; a plain `@deffun simple_lpdf` no longer
    # binds `simple`, so the `::typeof(simple)` methods here threw `UndefVarError: simple`
    # and blocked the whole include. Annotate only the heads actually used as base
    # distributions; `my_lpdf` stays plain (`my` is only ever called, never `~ my(...)`).
    @deffun begin
        @lpxf simple_lpdf(y, x) = 0.
        simple_lpdfs(y, x) = 0.
        simple_rng(x) = 0.
        @lpxf vararg_lpdf(y, args...) = 0.
        vararg_lpdfs(y, args...) = 0.
        vararg_rng(args...) = 0.

        my_lpdf(y, fargs...) = reject(1)
        my_lpdfs(args...) = reject(1)
        my_rng(args...) = reject(1)
        my_lpdf(y, ::typeof(simple), args...) = simple_lpdf(y, args...)
        my_lpdfs(y, ::typeof(simple), args...) = simple_lpdfs(y, args...)
        my_rng(::typeof(simple), args...) = simple_rng(args...)
        my_lpdf(y, ::typeof(vararg), args...) = vararg_lpdf(y, args...)

        @lpxf fof_lpdf(y, f, args...) = my_lpdf(y, f, args...)
        fof_lpdfs(y, f, args...) = my_lpdfs(y, f, args...)
        fof_rng(f, args...) = my_rng(f, args...)
        @lpxf srs2_lpdf(y, f, args...) = simple_reduce_sum(srs2_helper, rep_array(y, 1), f, args...)
        srs2_helper(y, f, args...) = my_lpdf(y, f, args...)
        srs2_lpdfs(y, f, args...) = 0.
        srs2_rng(f, args...) = 0.
    end

    # --- issue @deffun definitions (hoisted to module level) ---

    @deffun begin
        issue10a(::vector[n]) = 0.
        issue10b(_::vector[n]) = 0.
    end

    # UDF signature dimensions are bound in the emitted Stan body only when
    # this definition uses them. Repeated dimensions remain runtime-checked.
    @deffun begin
        udf_dead_size(x::vector[dead_n])::real = sum(x)
        udf_body_size(x::vector[body_n])::real = sum(x) / body_n
        udf_return_size(x::vector[return_n])::vector[return_n] = x
        udf_checked_size(x::vector[checked_n], y::vector[checked_n])::real = dot_product(x, y)
    end

    # --- computed type annotations (@deffun container inference; todo 1v94weu) ---
    # `rv::typeof(f(x[1]))[dims]` infers the output container from `f`'s per-element
    # return type: a `real` element → native `vector[n]`, an `int` element →
    # `array[] int` (stays int, usable as an index), `real` + 2 dims → `matrix[n,n]`.
    # So ONE definition covers both int- and real-valued element functions. Covers
    # BOTH the body-decl form (`umap`) and the `::ret` form (`umap_r`); HOF-aware —
    # `f` is a closure arg, lifted per call. Exercised by the "computed type
    # annotations" testset (commits `03aef9c` body decls + `a11934a` `::ret`).
    @deffun umap(f, x::anything[n]) = begin
        rv::typeof(f(x[1]))[n]
        for i in 1:n
            rv[i] = f(x[i])
        end
        rv
    end
    @deffun umap_r(f, x::anything[n])::typeof(f(x[1]))[n] = begin
        rv::typeof(f(x[1]))[n]
        for i in 1:n
            rv[i] = f(x[i])
        end
        rv
    end
    @deffun umap_mat(f, x::anything[n]) = begin
        rv::typeof(f(x[1]))[n,n]
        for i in 1:n
            for j in 1:n
                rv[i,j] = f(x[i])
            end
        end
        rv
    end
    # Error-path fixture: the annotation base is a VALUE (`x[1]::real`), not a type
    # token → `_decl_computed_type` must error loudly at trace time (reject sub-test).
    @deffun bad_computed_ann(x::anything[n]) = begin
        rv::(x[1])[n]
        rv[1] = x[1]
        rv
    end

    # --- typed assignment compatibility ---
    @deffun typed_assign_symbolic(x::vector[n]) = begin
        rv::vector[n] = x * 2.0
        rv
    end
    @deffun typed_assign_computed(x::anything[n]) = begin
        rv::typeof(x) = x
        rv
    end
    @deffun typed_assign_bad_center(x::vector[n]) = begin
        rv::matrix[n,n] = x
        rv
    end
    @deffun typed_assign_bad_dim(x::vector[n]) = begin
        rv::vector[n+1] = x
        rv
    end
    # --- BARE whole-vector local (no size annotation) with a broadcast RHS ---
    # The intermediate local's inferred size is the arg-placeholder fragment
    # `dims(a)[1]` (a `StanExpr{String}` of center `int`). It must emit UNQUOTED
    # (`vector[dims(a)[1]]`), not the invalid quoted `vector["dims(a)[1]"]`.
    # Single-arg (unambiguous dim) and multi-arg (repeated dim, size sourced
    # from a non-first arg) both exercise the render path.
    @deffun bare_vec_local_single(a::vector[n]) = begin
        xi = exp(a) .* a
        xi
    end
    @deffun bare_vec_local_multi(a::vector[n], b::vector[n], c::vector[n]) = begin
        xi = (a - b) .* exp(c)
        xi
    end
    # --- bounded array comprehensions in @deffun ---
    # A scalar expression over one or more explicit `lo:hi` ranges. Fixtures cover
    # real/int result inference, non-1 lower bounds, N-D (multi-generator) results,
    # and the still-rejected filtered / comprehension-valued shapes.
    @deffun comprehension_square(x::vector[n]) = [x[i] * x[i] for i in 1:n]
    @deffun comprehension_int_shift(x::int[n]) = [x[i] + 1 for i in 1:n]
    @deffun comprehension_slice(x::vector[n], lo::int, hi::int) = [x[i] for i in lo:hi]
    @deffun comprehension_filtered(x::vector[n]) = [x[i] for i in 1:n if x[i] > 0]
    # Multi-generator (`for i …, j …`) lowers to a 2-D result (matrix[n,m] here).
    @deffun comprehension_multiple(x::vector[n], m::int) = [x[i] + j for i in 1:n, j in 1:m]
    @deffun comprehension_nested(x::vector[n]) = [[x[i] + j for j in 1:2] for i in 1:n]
    # Chained/flattened generators (`for i … for j …`) build a ragged result — a
    # deferred follow-up; rejected loudly for now.
    @deffun comprehension_flatten(x::vector[n], m::int) = [x[i] + j for i in 1:n for j in 1:m]
    # enumerate / zip destructuring + N-D / nested iteration (devibe 2eb42c2).
    @deffun iter_enum(x::vector[n]) = [i * xi for (i, xi) in enumerate(x)]                 # → vector[n]
    @deffun iter_enum_col(X::matrix[m, k]) = [j * sum(c) for (j, c) in enumerate(EachCol(X))]
    @deffun iter_enum_acc(x::vector[n])::real = begin
        acc = 0.0
        for (i, xi) in enumerate(x); acc .= acc + i * xi; end
        acc
    end
    @deffun iter_zip(a::vector[n], b::vector[n]) = [ai * bi for (ai, bi) in zip(a, b)]      # → vector[n]
    @deffun iter_zip3(a::vector[n], b::vector[n], c::vector[n]) =
        [ai + bi + ci for (ai, bi, ci) in zip(a, b, c)]
    @deffun iter_nd(x::vector[n], y::vector[m]) = [x[i] * y[j] for i in 1:n, j in 1:m]      # → matrix[n,m]
    @deffun iter_nd_int(x::int[n], y::int[m]) = [x[i] + y[j] for i in 1:n, j in 1:m]        # → array[n,m] int
    @deffun iter_nested(X::matrix[n, m])::real = begin
        acc = 0.0
        for i in 1:n, j in 1:m; acc .= acc + X[i, j]; end
        acc
    end
    @deffun iter_nd3(x::int[n]) = [i + j + k for i in 1:n, j in 1:n, k in 1:n]              # 3-D → reject
    # value-iteration generators `[expr for xi in <container>]` desugar to index
    # iteration `[expr for _vi in 1:length(container)]` with `xi = container[_vi]`.
    @deffun comprehension_iterable(x::vector[n]) = [xi for xi in x]
    @deffun comprehension_iter_square(x::vector[n]) = [xi * xi for xi in x]
    @deffun comprehension_iter_eachcol(X::matrix[m, k]) = [sum(c) for c in EachCol(X)]
    @deffun comprehension_iter_ragged(g::RaggedVector) = [sum(gi) for gi in g]
    @deffun comprehension_stepped(x::vector[n]) = [x[i] for i in 1:2:n]
    # value-iteration FOR loop with `.=` accumulation (the `=` form hits the SSA
    # single-assignment gate — a pre-existing @deffun limitation, index loops too).
    @deffun for_iter_sum(x::vector[n])::real = begin
        acc = 0.0
        for xi in x
            acc .= acc + xi
        end
        acc
    end
    # --- public transpile-time return-type query ---
    @deffun return_type_scalar(x::real)::real = square(x)
    @deffun return_type_vector(x::vector[n])::vector[n] = x
    @deffun copy_via_return_type(x::vector[n]) = begin
        rv::return_type_of(return_type_scalar, x[1])[n]
        for i in 1:n
            rv[i] = x[i]
        end
        rv
    end
    ordinary_return_type_probe(x) = x

    # Determinism regression: an inline UDF whose locals are renamed `<name>__il_<id>`.
    @deffun @inline det_polish(x::vector[n])::vector[n] = begin
        z = x * 2
        return z + 1
    end

    # Determinism regression model — exercises BOTH a session-counter-dependent
    # Stan-output site: an inlined UDF (`__il_<id>` local rename) AND a lifted
    # closure (`// lifted closure (id <id>)` + `closure_<id>` fn name). Transpiled
    # twice in one session it must be byte-identical (per-trace counters, not
    # session-global ones).
    det_model = @slic (;n=5, ts=collect(1.0:5.0)) begin
        lambda ~ std_normal(;lower=0.)
        mu     ~ std_normal(;n=n)
        obs    = det_polish(mu)
        y      = ode_rk45((t, y_state) -> -lambda * y_state, [1.0], 0.0, to_array_1d(ts))
    end

    # Snag `ode-closure-capt` regression model — the Torsten/Stan idiom of a
    # parameter CAPTURED by an ODE-RHS closure whose solve feeds a DOWNSTREAM
    # likelihood. `lambda` must stay a sampled parameter (in `parameters`, `~`
    # in `model`), NOT be optimised into `generated quantities` as a
    # `std_normal_rng()` draw while still emitted as a trailing solver arg in
    # `transformed parameters` (out-of-scope Stan). `det_model` above only
    # covers the PRIOR-PREDICTIVE path (ODE output unused → `lambda` correctly
    # lands in GQ); this model forces the ODE output into transformed
    # parameters and so exercises the `backward!(closure)` capture-following fix.
    det_lik_model = @slic (;ts=[1.0], yobs=0.37) begin
        lambda ~ std_normal(;lower=0.)
        y      = ode_rk45((t, y_state) -> -lambda * y_state, [1.0], 0.0, to_array_1d(ts))
        pred   = y[1][1]
        yobs   ~ normal(pred, 0.05)
    end

    # Snag `ode-closure-capt`, DATA-side twin — a DATA value (`k`) captured ONLY
    # inside a lifted ODE-RHS closure. The emitter threads `k` as a trailing
    # solver arg, so `fetch_data!(closure)` must follow captures and emit `k`'s
    # `data` declaration; otherwise dead-data elimination drops it and the solve
    # references an out-of-scope `k`. `k` appears NOWHERE outside the closure, so
    # only the capture-following path can declare it.
    det_datacap_model = @slic (;ts=[1.0], yobs=0.37, k=2.0) begin
        lambda ~ std_normal(;lower=0.)
        y      = ode_rk45((t, y_state) -> -lambda * k * y_state, [1.0], 0.0, to_array_1d(ts))
        pred   = y[1][1]
        yobs   ~ normal(pred, 0.05)
    end

    # Built-in math constants (`pi`, `ℯ`) resolve to their Float64 value in a model
    # body (user decision 3bbtrv); arbitrary module-level numbers must NOT.
    @deffun @inline _scale_by_consts(x::vector[n])::vector[n] = (4. / pi) * x .+ ℯ
    consts_model = @slic (;n=5) begin
        x ~ std_normal(;n)
        s = _scale_by_consts(x)
    end
    CONSTS_TEST_NUM = 2.5
    consts_bad_model = @slic (;n=5) begin
        x ~ std_normal(;n)
        s = CONSTS_TEST_NUM .* x
    end

    # --- required @deffun kwargs (no default) ---
    # A kwarg declared without a default is *required*: omitting it at the
    # call site errors at trace time (the SLIC analogue of Julia's
    # `UndefKeywordError`). `kw_req_opt` mixes a required kwarg (`a`) with an
    # optional one (`b`), so the two paths coexist in one shim.
    @deffun kw_scale(x::vector[n]; scale)::vector[n] = scale * x
    @deffun kw_req_opt(x::vector[n]; a, b=2.0)::vector[n] = a * x .+ b
    # required kwarg provided → transpiles
    kw_required_model = @slic (;n=3) begin
        x ~ std_normal(;n)
        s = kw_scale(x; scale=2.0)
    end
    # mixed required+optional, both provided
    kw_mixed_model = @slic (;n=3) begin
        x ~ std_normal(;n)
        s = kw_req_opt(x; a=1.5, b=0.5)
    end
    # mixed, optional omitted → falls back to the registered default `b=2.0`
    kw_opt_default_model = @slic (;n=3) begin
        x ~ std_normal(;n)
        s = kw_req_opt(x; a=1.5)
    end

    # issue 12 sub-models
    sm12a = @slic begin
        x ~ std_normal(;n)
        return x .* x
    end
    sm12b = @slic begin
        x ~ std_normal(;n)
        return x
    end

    # issue 15 sub-models
    sm15a = @slic begin
        x = rep_vector(0., n)
        return x
    end
    sm15b = @slic begin
        x = rep_vector(0., n)
        xx = append_row(x, x)
        return xx
    end

    # chained + variable-bound splice (the crowdsource.qmd composition pattern).
    # The 916f100 migration converted the inline splice sites but overlooked
    # crowdsource.qmd, whose transforms are SEPARATELY-BOUND `quote` blocks doing
    # ASSIGNMENT (`=`) overrides, CHAINED through `Base.merge`. issue12/issue15
    # cover only a single INLINE `Base.merge(base, quote…end)`; this fixture drives
    # the chained + variable-bound + assignment-override shape.
    sm_splice_base = @slic begin
        lambda = rep_vector(1, n)
        delta  = rep_vector(2, n)
        x ~ std_normal(;n)
        return x .* lambda .* delta
    end
    splice_t1 = quote
        lambda = rep_vector(0, n)
    end
    splice_t2 = quote
        delta = rep_vector(1, n)
    end

    @deffun begin
        @lhs @lpxf issue17_lpdf(y::vector[n]) = begin
            rv = 0.
            for i in 1:n
                rv += y[i]
            end
            for i in 1:n-1
                rv += y[i]
            end
            for i in 1:dims(y)[1]
                rv += y[i]
            end
            rv
        end
    end

    @deffun begin
        @lpxf issue18_lpdf(y) = normal_cdf(y, 0, 1) + normal_lcdf(y, 0, 1) + normal_lccdf(y, 0, 1)
    end

    # issue 19 sub-models
    sm19a = @slic begin
        x = rep_vector(0., n)
        xx = x .* x
        return xx
    end
    sm19b = @slic begin
        x = rep_vector(0., n)
        xx = append_row(x,x)
        return xx
    end

    # issue 20 module
    module m20
        using StanBlocks
        @deffun begin
            f(x) = begin
                return x
            end
        end
        model = @slic begin
            x ~ std_normal(;n)
            return x
        end
        modela = @slic begin
            x ~ model(;n)
        end
        modelb = @slic begin
            x ~ model(;n)
            y = f(x)
        end
    end

    @deffun begin
        @lhs @lpxf issue9_lpdf(x::vector[n], n) = 0.
        issue9_rng(n) = rep_vector(0., n)
        issue9_rng(vector[n], n)::vector[n] = issue9_rng(n)
    end

    # --- logdensity.jl helpers ---

    _stan_std_normal(x) = -0.5*x^2
    _stan_normal(x, mu, sigma) = -log(sigma) - 0.5*((x-mu)/sigma)^2
    _stan_binomial(k, n, p) = k*log(p) + (n-k)*log(1.0-p)

    # --- builtin_shapes.jl helpers ---

    function check_type(model, var::Symbol, expected_sigtype::AbstractString)
        code = stan_code(model)
        pat = Regex("\\b$(expected_sigtype)(?:\\[[^\\]]*\\])+ (?:\\w+ )?$var\\b")
        occursin(pat, code)
    end

    # ──────────────────────────────────────────────────────────────────────────────
    # Tests
    # ──────────────────────────────────────────────────────────────────────────────

    # === slic.jl tests ===








    # --- `@slic f(…)=…` named sub-model functions (SubmodelFn) ---
    # The @deffun-analogue for models: a named callable whose POSITIONAL args bind
    # by name into the sub-model's data, embedded via `~`-rhs-is-SlicModel. Typed
    # args drive native @deffun-style multiple-dispatch on the StanExpr center-type.
    # Mirrors the `web/sandbox/submodelfn_{positional,2arg,dispatch}.jl` snippets so
    # the feature has a CI regression as well as the dashboard smoke surface (§R14).
    @slic linpred(slope) = begin
        intercept ~ normal(0, 1)
        return intercept + slope
    end
    @slic affine(a, b) = begin
        z ~ normal(0, 1)
        return a + b * z
    end
    @slic dcase(s::real) = begin
        z ~ normal(0, 1)
        return z + s
    end
    @slic dcase(s::int) = begin
        z ~ normal(0, 5)
        return z + s
    end


    @deffun begin
        preconditioned_normal_lpdf(xi::matrix[m, n], loc::vector[m], scale::vector[m], prescale::matrix[m,m], n) = begin
            multi_normal_cholesky_lpdf(eachcol(xi), mdivide_left_tri_low(prescale, loc), mdivide_left_tri_low(prescale, diag_matrix(scale)))
        end
    end







    # An in-body `@doc` lowers to a DocumentExpr; `forward!` wraps the docstring
    # String into a `StanExpr{String}`, so render must dispatch
    # `commentstring(::StanExpr)` (not just `::String`).
    doc_model = @slic (;n=5) begin
        y ~ std_normal(;n)
        @doc "documented local declaration" z = y .* 2
    end







    # === partly-missing-vector imputation tests ===








    # === logdensity.jl tests ===






    # A bare `::` declaration has two deliberately scope-sensitive meanings:
    # model/sub-model scope declares an improper-uniform parameter, while a non-inline
    # `@deffun` body keeps the established Stan function-local declaration semantics.
    @deffun bare_parameter_local_copy(x::vector[n])::vector[n] = begin
        out::vector[n]
        for i in 1:n
            out[i] = x[i]
        end
        return out
    end
    bare_parameter_submodel = @slic begin
        offset::real
        return offset
    end




    # === builtin_shapes.jl tests ===














    # --- regression: @deffun symbolic-size operand-type propagation through HOFs ---
    # A symbolic-size UDF return type (here `gslice`'s `vector[max(0, ends-start+1)]`)
    # threaded through a higher-order `size(f(...))` layer used to corrupt the size
    # expression's operands: every anonymization level reused `_arg1`, `_arg2`, … so
    # an inner `deanon_size` re-substituted an enclosing UDF param's placeholder with
    # the wrong arg, aliasing a scalar index to a 2-D value and surfacing as
    # `tracetype not defined for (anything - anything)` (and an infinite-loop hang on
    # the direct path). Per-call placeholder namespacing (`_arg<tok>_<i>`) fixes it.
    # Reproduces with ZERO `@inline` — it is a `@deffun` size-propagation defect.
    @deffun begin
        hd_rstart(x, i) = if i == 1; 1; else 1 + x.ends[i-1] end
        hd_rend(x, i) = x.ends[i]
        hd_gslice(x::vector[_], start, ends)::vector[max(0, ends-start+1)] = if ends < start
            rep_vector(0., 0)
        else
            x[start:ends]
        end
        hd_rvec(x, i) = hd_gslice(x.mem, hd_rstart(x, i), hd_rend(x, i))
        # symbolic-size 2-D core (sizes from arg shapes) — the necessary ingredient
        hd_core_sym(a::vector[m], b::vector[n])::int[m, n] = rep_array(0, m, n)
        # concrete-size 2-D core — control: must stay clean (and did pre-fix)
        hd_core_concrete(a::vector[m], b::vector[n])::int[2, 3] = rep_array(0, 2, 3)
        hd_wrap_sym(subject, ut, dt) = to_array_1d(hd_core_sym(hd_rvec(ut, subject), hd_rvec(dt, subject)))
        hd_wrap_concrete(subject, ut, dt) = to_array_1d(hd_core_concrete(hd_rvec(ut, subject), hd_rvec(dt, subject)))
        # double-HOF: f threaded through `size(f(...))` (sub_lens -> sub_len)
        hd_sublen(f, xi, a, b) = size(f(xi, a, b))
        hd_sublens(f, x::anything[n], a, b) = begin
            rv::int[n]
            for i in 1:n
                rv[i] = hd_sublen(f, x[i], a, b)
            end
            rv
        end
    end


    # --- regression: compiler-injected fresh-result slice-fills (case-3 layers 2+3) ---
    # An `@inline` UDF that declares its own result vector and fills it element/slice-wise
    # (`out::vector[n]; out[i]=…; return out`) injects a bare declaration + those fills into
    # the *model* body. They reach `distribute!` wrapped as NON-void StanExprs, which the
    # generic method rejected. `forward!` now PROMOTES the base var's qual across the fills
    # (`_promote_qual`); `backward!(::StanExpr{<:DeclExpr})` stays INERT so the promoted qual
    # survives; `distribute!` routes the decl + every fill to ONE block by that finalized qual
    # (`_qual_blocks`). The read-only gate still rejects filling a SAMPLED parameter. Commit
    # e52344e (todos 168mbms + 1fncg6j). GATE ON stanc_compiles() — transpiles()/stan_code alone
    # accept invalid Stan (primer §@deffun-gotchas); the original piece was only verified with
    # stan_code (codegen), never stanc.
    @deffun @inline c3_freshfill(x::vector[n])::vector[n] = begin
        out::vector[n]
        out[1] = x[1] * x[1]
        out[2] = x[2] + x[3]
        return out
    end
    @deffun @inline c3_nestedfill(x::vector[n])::vector[n] = begin
        return c3_freshfill(x)
    end
    @deffun @inline c3_mapinto!(out::vector[n], f, x::vector[n])::vector[n] = begin
        for i in 1:n
            out[i] = f(x[i])
        end
        return out
    end
    @deffun c3_softplus(x::real)::real = log1p(exp(x))
    @deffun c3_build_eta(alpha::real, beta::vector[k], X::matrix[n, k])::vector[n] = begin
        eta = rep_vector(alpha, n)
        smooth = rep_vector(0., n)
        c3_mapinto!(smooth, c3_softplus, eta)
        return eta + smooth
    end
    @deffun @inline c3_set_first!(buf::vector[n])::vector[n] = begin
        buf[1] = 42.
        return buf
    end



    # --- regression: cert-marker read-only gate for compiler-injected slice-fills (case-3) ---
    # The gate in `backward!(::AssignmentExpr)` (passes.jl) routes a compiler-injected
    # slice-fill IFF its base was declared FRESH via `::` (`fresh_decl`, stamped by
    # `forward!(::DeclExpr)`); EVERY other base is read-only for element assignment in a
    # model block and must reject. Commit a6bf5a3 (todo 4dhw3f) GENERALISED the earlier
    # parameter-only gate to also catch two cases the testset above never exercised (it
    # predates the cert marker): a DATA input, and a PRE-COMMITTED/derived local. Assert the
    # error *message* too, so a regression that rejects for the WRONG reason — or silently
    # mis-routes/accepts — surfaces here, not just a bare `!transpiles`. All four models use
    # the `c3_set_first!` mutating helper defined above (`buf[1]=42.; return buf`).
    _cert_reject_msg(m) = try; stan_code(m); nothing; catch e; sprint(showerror, e); end

    # --- regression: compiler-injected for-loop + range-indexed fresh-result fills ---
    # (case-3 deltas C.1 / C.2; commit 29c3b59 + todos 17cynmi / 1u392rh).
    # C.1 (ForExpr routing): an `@inline` UDF whose body is a `for`-loop fresh-result
    # fill (`out::vector[n]; for i in 1:n; out[i]=…; end; return out`), called DIRECTLY
    # from a model body, injects a REAL Stan `for` into the *model* body — where `info`
    # is the top-level StanModel. That path was never exercised before 29c3b59 (existing
    # injected loops only inline into forward-only @deffun bodies) and needed five gaps
    # fixed: `pop!(::StanModel)`, a `:data` qual on the loop index (else `:undefined`
    # poisoned `maximum(qual, args)` and mis-routed the loop to generated quantities),
    # `backward!`/`distribution_blocks`/`fetch_data!(::StanExpr{<:ForExpr})`. The loop
    # routes by the coarse (max) qual over the base vars its body fills and emits verbatim
    # via `show(::ForExpr)`, co-located with its decl + result-bind in ONE block.
    # C.2 (range-indexed LHS): a range fresh-result fill (`out[1:2]=x[1:2]`) coarse-grains
    # its range-getindex LHS to the base var (`_base_lhs_symbol(::CanonicalExpr)`), routing
    # exactly like a scalar fill (no code change — verify-only, todo 1u392rh).
    # GATE ON stanc_compiles() (stanc): the qual-poisoning bug 29c3b59 fixed
    # mis-routed the loop while STILL passing stan_code/transpiles (primer §compiles-not-
    # codegen); the `!occursin("for(", generated quantities)` check guards that regression.

    # Extract a top-level Stan block's body (braces included) by brace-matching from its
    # header — robust to the nested braces of a `for(...){...}` loop body, which a
    # `…\{[^}]*…` regex (as used by the case-3 testset above) would truncate at the loop's
    # first inner `}`.
    function stan_block(code::AbstractString, name::AbstractString)
        r = findfirst(name * " {", code)
        isnothing(r) && return ""
        i = last(r); depth = 0; start = i
        while i <= lastindex(code)
            c = code[i]
            c == '{' && (depth += 1)
            c == '}' && (depth -= 1; depth == 0 && return code[start:i])
            i = nextind(code, i)
        end
        code[start:end]
    end

    @deffun @inline c3_forfill(x::vector[n])::vector[n] = begin
        out::vector[n]
        for i in 1:n
            out[i] = x[i] * x[i]
        end
        return out
    end
    @deffun @inline c3_rangefill(x::vector[n])::vector[n] = begin
        out::vector[n]
        out[1:2] = x[1:2]
        return out
    end

    # Test-only producer for the shared do-block plate ROUTING contract. The public
    # `rv ~ plate(...) do ... end` emitter is a separate feature; this hook injects
    # exactly its agreed lower-level shape: outer declarations followed by one
    # symbolic loop containing fresh indexed samples, a transformed result fill, a
    # data fill, and an indexed observation. Keeping the producer synthetic lets
    # this regression isolate the shared forward/backward/distribution machinery.
    function c3_plate_router_probe end
    function StanBlocks.stan.expand_inline_or_trace(
            call::StanBlocks.stan.CanonicalExpr{typeof(c3_plate_router_probe)};
            info)
        n, obs = call.args
        injected = quote
            plate_x::vector[$n]
            plate_y::vector[$n]
            plate_prior::vector[$n]
            plate_flat::real
            plate_rv::vector[$n]
            plate_data_copy::vector[$n]
            for plate_i in 1:$n
                plate_x[plate_i] ~ normal(0.0, 1.0)
                plate_y[plate_i] ~ normal(plate_x[plate_i] + plate_flat, 1.0)
                plate_prior[plate_i] ~ normal(0.0, 1.0)
                plate_rv[plate_i] = plate_x[plate_i] + plate_y[plate_i]
                plate_data_copy[plate_i] = $obs[plate_i]
                $obs[plate_i] ~ normal(plate_rv[plate_i], 1.0)
            end
        end
        StanBlocks.stan.forward!(StanBlocks.stan.canonical(injected); info)
    end

    c3_plate_router_submodel = @slic (;obs=randn(4), n=4) begin
        c3_plate_router_probe(n, obs)
        return plate_rv
    end
    c3_plate_router_model = @slic (;) begin
        routed ~ c3_plate_router_submodel
    end

    # Public plate regression: the do-block calls a named @slic submodel whose own
    # sample and derived local are not visible in the plate's raw AST.  The plate
    # discovery trace must find all three flattened bindings (`cell_z`,
    # `cell_shifted`, `cell`), and the emit trace must promote every definition and
    # reference to the current outer index.
    @slic c3_plate_ncp(mu::real, sigma::real) = begin
        z ~ std_normal()
        shifted = mu + sigma * z
        return shifted
    end
    c3_plate_submodel_model = @slic (;y=randn(6), mu0=0.5) begin
        sigma ~ normal(0.0, 1.0; lower=0.0)
        theta ~ plate(y; outer=(6,)) do yi
            cell ~ c3_plate_ncp(mu0, sigma)
            yi ~ normal(cell, sigma)
            cell
        end
    end

    # The same promotion must preserve a vector-valued submodel cell: the internal
    # vector parameter and returned value collect as matrices and are indexed by
    # column in the emitted loop.
    @slic c3_plate_vector_ncp(k::int) = begin
        z::vector[k] ~ std_normal()
        return z
    end
    c3_plate_vector_submodel_model = @slic (;n=3, k=2) begin
        theta ~ plate(; outer=(n,)) do _i
            cell ~ c3_plate_vector_ncp(k)
            cell
        end
    end

    # Heterogeneous vector cells use one data-sized flat-memory carrier per
    # discovered binding. Discovery must retain the actual plate index inside
    # `K[g]`, including across a called submodel boundary.
    @slic c3_plate_ragged_ncp(k::int) = begin
        z::vector[k] ~ std_normal()
        return z
    end
    @slic c3_plate_ragged_correlated_cell(k::int, L::matrix[k, k]) = begin
        z::vector[k] ~ std_normal()
        return L * z
    end
    c3_plate_ragged_model = @slic (;K=[2, 3, 4], y=0.3) begin
        b ~ plate(; outer=length(K)) do g
            z::vector[K[g]] ~ std_normal()
            z
        end
        y ~ normal(sum(b[1]), 1.0)
    end
    c3_plate_ragged_submodel_model = @slic (;K=[2, 3, 4], y=0.3) begin
        b ~ plate(; outer=length(K)) do g
            cell ~ c3_plate_ragged_ncp(K[g])
            cell
        end
        y ~ normal(sum(b[2]), 1.0)
    end
    # Snag regression (ragged-dist-arg-dcffbc1b): a plate collects per-cell VECTORS
    # of differing lengths into a ragged result `mu`, then a TOP-LEVEL obs model is
    # applied to the WHOLE ragged result (obs-outside). It BROADCASTS across groups.
    c3_plate_ragged_obs_outside_model = @slic (;
        ts=[[0.5, 0.7, 0.9], [0.4, 0.6], [0.3, 0.8, 1.0, 1.2]],
        ys=[[0.6, 0.8, 1.0], [0.5, 0.7], [0.4, 0.9, 1.1, 1.3]],
        nsub=3,
    ) begin
        sigma ~ exponential(1)
        mu ~ plate(ts; outer=(nsub,)) do t
            m::typeof(t) = 2.0 .* t
            m
        end
        ys ~ normal(mu, sigma)
    end
    # Obs-in-cell twin — the reference the broadcast form must be NUMERICALLY equal to.
    c3_plate_ragged_obs_incell_model = @slic (;
        ts=[[0.5, 0.7, 0.9], [0.4, 0.6], [0.3, 0.8, 1.0, 1.2]],
        ys=[[0.6, 0.8, 1.0], [0.5, 0.7], [0.4, 0.9, 1.1, 1.3]],
        nsub=3,
    ) begin
        sigma ~ exponential(1)
        mu ~ plate(ts, ys; outer=(nsub,)) do t, y
            m::typeof(t) = 2.0 .* t
            y ~ normal(m, sigma)
            m
        end
    end
    # Vector-valued (multi_normal) ragged obs: the ragged structure MUST be preserved
    # — each group is ONE multivariate observation, never a flattened backing.
    # Equal-length groups so a shared covariance applies.
    c3_ragged_obs_broadcast_mvn_model = @slic (;
        ym=[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]],
        mm=[[0.0, 0.1, 0.2], [0.3, 0.4, 0.5], [0.6, 0.7, 0.8]],
        Sig=[1.0 0.0 0.0; 0.0 1.0 0.0; 0.0 0.0 1.0],
    ) begin
        scale ~ exponential(1)
        ym ~ multi_normal(mm, Sig)
    end
    c3_plate_ragged_brm_model = @slic (;K=[1, 2, 3], y=0.3) begin
        L::cholesky_factor_corr[K] ~ lkj_corr_cholesky(2.0)
        b ~ plate(; outer=length(K)) do g
            cell ~ c3_plate_ragged_correlated_cell(K[g], L[g])
            cell
        end
        y ~ normal(sum(b[3]), 1.0)
    end

    # Snag regression (plate-called-cel-c38a50e8): a called cell declaring a plain
    # `matrix[k, k]` argument must accept a FIXED-width top-level Stan constrained
    # square-matrix value (`cholesky_factor_corr[k]`).  Unlike
    # `c3_plate_ragged_brm_model`, whose ragged `L[g]` reconstructs a plain `matrix`
    # (ndim 2), here `L` stays `cholesky_factor_corr` (ndim 1, `r_ndim(square_matrix)`)
    # and previously failed plate-discovery dispatch with a `SubmodelFn` MethodError
    # on the `ndims` type parameter (value 1 ≠ decl 2), even though
    # `cholesky_factor_corr <: matrix`.  The value flows into the cell verbatim, so
    # its constrained declaration is preserved at the caller.
    @slic c3_plate_fixed_correlated_cell(k::int, L::matrix[k, k], tau::vector[k]) = begin
        z::vector[k] ~ std_normal()
        return diag_pre_multiply(tau, L) * z
    end
    c3_plate_fixed_correlated_model = @slic (;n_groups=5, k=3, y=[0.2, -0.1, 0.3, 0.0, 0.4]) begin
        L::cholesky_factor_corr[k] ~ lkj_corr_cholesky(2.0)
        tau::vector[k] ~ normal(0.0, 1.0; lower=0.0)
        b::vector[k] ~ plate(y; outer=(n_groups,)) do yi
            cell ~ c3_plate_fixed_correlated_cell(k, L, tau)
            yi ~ normal(cell[1], 0.5)
            cell
        end
    end

    # Snag regression (plate-inside-cal-69cec8bc): a fixed-vector plate INSIDE a
    # called `@slic` submodel body. Previously threw
    # `TypeError: ... expected StanModel, got SubModel` at `_plate_discover`,
    # because discovery + outer-decl emission assumed a top-level `StanModel` info.
    # Now the probe MIRRORS the SubModel structure, fresh cell collections discover
    # and emit under their flattened (global) names at the root, and the `rv` result
    # keeps its local alias for the submodel's `return`. This is BRM's
    # `ranef_correlated_draws`-as-a-plate migration shape: submodel-scope params
    # (`L`, `tau`) feed a per-group correlated draw.
    @slic c3_plate_in_sub_draws(n_groups::int, k::int) = begin
        L::cholesky_factor_corr[k] ~ lkj_corr_cholesky(2.0)
        tau::vector[k] ~ std_normal(; lower=0.0)
        b::vector[k] ~ plate(; outer=(n_groups,)) do g
            z::vector[k] ~ std_normal()
            diag_pre_multiply(tau, L) * z
        end
        return b
    end
    c3_plate_in_submodel_model = @slic (;n_groups=4, k=3, y=[0.2, -0.1, 0.3, 0.0]) begin
        b ~ c3_plate_in_sub_draws(n_groups, k)
        y ~ normal(to_vector(b'[:, 1]), 0.5)
    end
    # Top-level equivalent: identical parameters + target, so the two must yield an
    # identical log density and gradient. Guards the SubModel promotion path against
    # ever diverging from the already-verified top-level plate path.
    c3_plate_in_submodel_reference = @slic (;n_groups=4, k=3, y=[0.2, -0.1, 0.3, 0.0]) begin
        L::cholesky_factor_corr[k] ~ lkj_corr_cholesky(2.0)
        tau::vector[k] ~ std_normal(; lower=0.0)
        b::vector[k] ~ plate(; outer=(n_groups,)) do g
            z::vector[k] ~ std_normal()
            diag_pre_multiply(tau, L) * z
        end
        y ~ normal(to_vector(b'[:, 1]), 0.5)
    end
    # Snag regression (plate-submodel-c-f792c57a): the SAME submodel, but the CALLER
    # names the group count `G` and term count `P` — DIFFERENT from the submodel's
    # `n_groups`/`k`. A submodel data arg ALIASES the caller's binding (`n_groups` ⇒
    # `G`, `k` ⇒ `P`) rather than gaining a `<sub>_` prefix, so the raw local names
    # do NOT exist at the root scope. The fresh cell's root decl AND the `rv` decl
    # must therefore be sized by `P`/`G`. The matching-names model above only passed
    # because the caller happened to reuse `n_groups`/`k`, masking this.
    c3_plate_in_submodel_renamed = @slic (;G=4, P=3, y=[0.2, -0.1, 0.3, 0.0]) begin
        b ~ c3_plate_in_sub_draws(G, P)
        y ~ normal(to_vector(b'[:, 1]), 0.5)
    end
    # Top-level equivalent with the caller-scope names, for log-density/gradient
    # parity against the renamed submodel path.
    c3_plate_in_submodel_renamed_reference = @slic (;G=4, P=3, y=[0.2, -0.1, 0.3, 0.0]) begin
        L::cholesky_factor_corr[P] ~ lkj_corr_cholesky(2.0)
        tau::vector[P] ~ std_normal(; lower=0.0)
        b::vector[P] ~ plate(; outer=(G,)) do g
            z::vector[P] ~ std_normal()
            diag_pre_multiply(tau, L) * z
        end
        y ~ normal(to_vector(b'[:, 1]), 0.5)
    end

    # N-dimensional plate regressions: preserve the shipped dense 1-D shapes,
    # route one compiler-owned loop per outer axis, and keep logical cell dimensions
    # first while Stan array prefixes carry outer axes beyond the vector/matrix core.
    c3_plate_outer_int_model = @slic (;y=randn(4)) begin
        theta ~ plate(y; outer=4) do yi
            z ~ normal(0.0, 1.0)
            yi ~ normal(z, 1.0)
            z
        end
    end
    c3_plate_outer_2d_scalar_model = @slic (;y=randn(2, 3)) begin
        theta ~ plate(y; outer=(2, 3)) do yi
            z ~ normal(0.0, 1.0)
            yi ~ normal(z, 1.0)
            z
        end
    end
    c3_plate_outer_2d_vector_model = @slic (;y=randn(2, 3), k=4) begin
        theta ~ plate(y; outer=(2, 3)) do yi
            z::vector[k] ~ std_normal()
            yi ~ normal(z[1], 1.0)
            z
        end
    end
    c3_plate_outer_3d_scalar_model = @slic (;y=randn(2, 3)) begin
        theta ~ plate(; outer=(2, 3, 4)) do i, j, k
            z ~ normal(0.0, 1.0)
            y[i, j] ~ normal(z, 1.0)
            z + 0.0 * k
        end
    end
    c3_plate_outer_4d_scalar_model = @slic (;y=randn(2, 3)) begin
        theta ~ plate(; outer=(2, 3, 4, 5)) do i, j, k, l
            z ~ normal(0.0, 1.0)
            y[i, j] ~ normal(z, 1.0)
            z + 0.0 * k + 0.0 * l
        end
    end

    # Folded `lwchee` regression: one public model exercises the registered
    # transform builtin, compiler-certified range fill, symbolic TP loop, and the
    # compile-time RaggedVector representation together.
    c3_ragged_simplex_model = @slic (;K=[2, 3, 4], y=0.3) begin
        p::simplex[K] ~ flat()
        y ~ normal(sum(p[1]), 0.1)
    end
    c3_ragged_simplex_informative_model = @slic (;
        K=[2, 3, 4],
        alpha=[[1.5, 2.0], [1.2, 1.4, 1.6], [1.1, 1.3, 1.5, 1.7]],
        y=0.3,
    ) begin
        p::simplex[K] ~ dirichlet(alpha)
        y ~ normal(sum(p[1]), 0.1)
    end
    c3_ragged_ordered_informative_model = @slic (;K=[2, 3, 4], y=0.3) begin
        p::ordered[K] ~ normal(0.0, 1.0)
        y ~ normal(sum(p[1]), 0.1)
    end

    # Matrix-valued ragged constraints use a flat RaggedMatrix carrier.  Start at
    # K=1 so the correlation family also exercises its zero-coordinate first group.
    c3_ragged_cholesky_corr_model = @slic (;K=[1, 2, 3], y=0.3) begin
        L::cholesky_factor_corr[K] ~ flat()
        y ~ normal(sum(to_vector(L[2])), 0.1)
    end
    c3_ragged_cholesky_cov_model = @slic (;K=[1, 2, 3], y=0.3) begin
        L::cholesky_factor_cov[K] ~ flat()
        y ~ normal(sum(to_vector(L[2])), 0.1)
    end
    c3_ragged_cholesky_corr_informative_model = @slic (;K=[1, 2, 3], y=0.3) begin
        L::cholesky_factor_corr[K] ~ lkj_corr_cholesky(2.0)
        y ~ normal(sum(to_vector(L[2])), 0.1)
    end



    # Public `rv ~ plate(...) do ... end` emitter (Feature 2, surface n35u3c).
    # The routing testset above exercises the shared lower-level contract via a
    # synthetic producer; THIS testset drives the real public surface end-to-end,
    # promoting the `web/sandbox/plate_*.jl` regression snippets into the suite so
    # the trace-then-promote rework (1vujeta) cannot silently regress the MVP.






    # === posteriordb.jl tests ===

    # Generate per-model test functions for PosteriorDB
  for _name in names(@__MODULE__; all=true, imported=false)
      _text = String(_name)
      Base.isidentifier(_text) || continue
      startswith(_text, "#") && continue
      _name in (:StanBlocksTestSetup, :eval, :include, :_name, :_text) && continue
      @eval export $_name
  end
end

"""
Verify `slic: normal(loc,scale)` in an isolated test item.
"""
@testitem "slic: normal(loc,scale)" tags=[:slic] setup=[StanBlocksImports, StanBlocksTestSetup] begin
    @test stanc_compiles(@slic (;obs=0.) begin
        loc ~ std_normal()
        scale ~ std_normal(;lower=0.)
        obs ~ normal(loc, scale)
    end)
end

@testitem "slic: required @deffun kwarg" tags=[:slic] setup=[StanBlocksImports, StanBlocksTestSetup] begin
    # Required kwarg provided at the call site → transpiles.
    @test transpiles(kw_required_model)
    # Omitting a required kwarg (no default) errors at trace time — the SLIC
    # analogue of Julia's `UndefKeywordError`. The `@slic` form constructs
    # fine; the error surfaces when the model is traced (`stan_code`).
    @test_throws Exception stan_code(@slic (;n=3) begin
        x ~ std_normal(;n)
        s = kw_scale(x)
    end)
end

@testitem "slic: required + optional @deffun kwargs" tags=[:slic] setup=[StanBlocksImports, StanBlocksTestSetup] begin
    # Required (`a`) + optional (`b`) kwargs coexist in one shim: both provided,
    # and the optional omitted (falls back to the registered default) both work.
    @test transpiles(kw_mixed_model)
    @test transpiles(kw_opt_default_model)
    # Omitting the *required* kwarg still errors even when the optional is given.
    @test_throws Exception stan_code(@slic (;n=3) begin
        x ~ std_normal(;n)
        s = kw_req_opt(x; b=1.0)
    end)
end

"""
Verify `slic: simple` in an isolated test item.
"""
@testitem "slic: simple" tags=[:slic] setup=[StanBlocksImports, StanBlocksTestSetup] begin
    @test stanc_compiles(@slic (;obs=0.) begin
        loc ~ std_normal()
        obs ~ simple(loc)
    end)
end

"""
Verify `slic: vararg` in an isolated test item.
"""
@testitem "slic: vararg" tags=[:slic] setup=[StanBlocksImports, StanBlocksTestSetup] begin
    @test stanc_compiles(@slic (;obs=0.) begin
        loc ~ std_normal()
        obs ~ vararg(loc)
    end)
end

"""
Verify `slic: fof(simple)` in an isolated test item.
"""
@testitem "slic: fof(simple)" tags=[:slic] setup=[StanBlocksImports, StanBlocksTestSetup] begin
    @test stanc_compiles(@slic (;obs=0.) begin
        loc ~ std_normal()
        obs ~ fof(simple, loc)
    end)
end

"""
Verify `slic: srs2(vararg)` in an isolated test item.
"""
@testitem "slic: srs2(vararg)" tags=[:slic] setup=[StanBlocksImports, StanBlocksTestSetup] begin
    @test stanc_compiles(@slic (;obs=0.) begin
        loc ~ std_normal()
        obs ~ srs2(vararg, loc)
    end)
end

"""
Verify `slic: srs2(vararg, extra)` in an isolated test item.
"""
@testitem "slic: srs2(vararg, extra)" tags=[:slic] setup=[StanBlocksImports, StanBlocksTestSetup] begin
    @test stanc_compiles(@slic (;obs=0.) begin
        loc ~ std_normal()
        obs ~ srs2(vararg, loc, (1, 2, 3))
    end)
end

"""
Verify `slic: stan_model re-data` in an isolated test item.
"""
@testitem "slic: stan_model re-data" tags=[:slic] setup=[StanBlocksImports, StanBlocksTestSetup] begin
    @test stanc_compiles(stan_model(@slic (;obs=randn(5)) begin
        loc ~ std_normal()
        scale ~ std_normal(;lower=0.)
        obs ~ normal(loc, scale)
    end)(;obs=randn(10)))
end

"""
Verify `slic: @slic f(…)=… sub-model functions` in an isolated test item.
"""
@testitem "slic: @slic f(…)=… sub-model functions" tags=[:slic] setup=[StanBlocksImports, StanBlocksTestSetup] begin
    # positional 1-arg: `mu ~ linpred(x)` binds x into the sub-model's data.
    @test stanc_compiles(@slic (; x=0.7, y=[1.0, 2.0, 3.0]) begin
        mu ~ linpred(x)
        y ~ normal(mu, 1)
    end)
    # positional 2-arg.
    @test stanc_compiles(@slic (; p=1.0, q=2.0, y=[1.0, 2.0]) begin
        w ~ affine(p, q)
        y ~ normal(w, 1)
    end)
    # typed-arg native dispatch: a `::real` arg and an `::int` arg select
    # DISTINCT methods of the SAME `dcase`, so BOTH priors must appear in the
    # emitted Stan — the load-bearing proof that @deffun-style dispatch resolved.
    disp = @slic (; sc=2.0, si=3, y=[1.0, 2.0]) begin
        a ~ dcase(sc)   # sc::real -> normal(0, 1) method
        b ~ dcase(si)   # si::int  -> normal(0, 5) method
        y ~ normal(a + b, 1)
    end
    @test transpiles(disp)
    disp_code = stan_code(disp)
    @test occursin("normal(0, 1)", disp_code)   # ::real method body
    @test occursin("normal(0, 5)", disp_code)   # ::int  method body
end

"""
Verify `issue9` in an isolated test item.
"""
@testitem "issue9" tags=[:slic, :regression] setup=[StanBlocksImports, StanBlocksTestSetup] begin
    @test stanc_compiles(@slic (;n=10) begin
        x ~ issue9(n)
    end)
    @test stanc_compiles(@slic (;n=10, y=1.) begin
        x ~ issue9(n)
        y ~ vararg(x)
    end)
end

"""
Verify `issue10` in an isolated test item.
"""
@testitem "issue10" tags=[:slic, :regression] setup=[StanBlocksImports, StanBlocksTestSetup] begin
    @test stanc_compiles(@slic (;n=10) begin
        y ~ std_normal(;n)
        x = issue10a(y)
    end)
    @test stanc_compiles(@slic (;n=10) begin
        y ~ std_normal(;n)
        x = issue10b(y)
    end)
end

"""
Verify dead UDF signature dimensions are not emitted while required bindings remain.
"""
@testitem "slic: UDF signature dimension use analysis" tags=[:slic, :regression, :shapes, :stanc] setup=[StanBlocksImports, StanBlocksTestSetup] begin
    model = @slic (;x=randn(3), y=randn(3)) begin
        dead = udf_dead_size(x)
        body_used = udf_body_size(x)
        return_used = udf_return_size(x)
        checked = udf_checked_size(x, y)
        p ~ normal(dead + body_used + sum(return_used) + checked, 1.0)
    end

    @test stanc_compiles(model)
    functions = stan_block(stan_code(model), "functions")
    @test occursin("udf_dead_size", functions)
    @test !occursin("int dead_n = dims(x)[1];", functions)
    @test occursin("int body_n = dims(x)[1];", functions)
    @test occursin("int return_n = dims(x)[1];", functions)
    @test occursin("int checked_n = dims(x)[1];", functions)
    @test occursin("if (dims(y)[1] != checked_n) reject", functions)
end

"""
Verify `issue12` in an isolated test item.
"""
@testitem "issue12" tags=[:slic, :regression] setup=[StanBlocksImports, StanBlocksTestSetup] begin
    @test stan_code(Base.merge(sm12a, quote
        return x
    end)(; n=10, y=1.)) == stan_code(sm12b(; n=10, y=1.))
    @test stanc_compiles(@slic (;n=10, y=1.) begin
        x ~ sm12a(;n)
        y ~ simple(x)
    end)
    @test stanc_compiles(@slic (;n=10, y=1.) begin
        x ~ sm12b(;n)
        y ~ simple(x)
    end)
end

"""
Verify `issue15` in an isolated test item.
"""
@testitem "issue15" tags=[:slic, :regression] setup=[StanBlocksImports, StanBlocksTestSetup] begin
    @test stan_code(Base.merge(sm15a, quote
        xx = append_row(x, x)
        return xx
    end)(; n=10, y=1.)) == stan_code(sm15b(; n=10, y=1.))
    @test stanc_compiles(sm15a(;n=10, y=1.))
    @test stanc_compiles(sm15b(;n=10, y=1.))
end

"""
Verify `slic: chained + variable-bound Base.merge splice (crowdsource pattern)` in an isolated test item.
"""
@testitem "slic: chained + variable-bound Base.merge splice (crowdsource pattern)" tags=[:slic] setup=[StanBlocksImports, StanBlocksTestSetup] begin
    # single variable-bound splice: the `lambda` assignment is overridden to
    # rep_vector(0, n); the original rep_vector(1, n) must be gone. (Match the
    # assignment `lambda = …`, not a bare `rep_vector(0, n)` — the latter also
    # appears in the std_normal RNG helper body, which would be vacuous.)
    single = Base.merge(sm_splice_base, splice_t1)
    @test transpiles(single(; n=3))
    single_code = stan_code(single(; n=3))
    @test occursin("lambda = rep_vector(0, n)", single_code)    # override applied
    @test !occursin("lambda = rep_vector(1, n)", single_code)   # original gone
    @test occursin("delta = rep_vector(2, n)", single_code)     # untouched statement kept
    # chained: BOTH overrides applied — lambda→0 AND delta→1 — original delta gone.
    chained = Base.merge(Base.merge(sm_splice_base, splice_t1), splice_t2)
    @test transpiles(chained(; n=3))
    chained_code = stan_code(chained(; n=3))
    @test occursin("lambda = rep_vector(0, n)", chained_code)   # lambda→0
    @test occursin("delta = rep_vector(1, n)", chained_code)    # delta→1
    @test !occursin("delta = rep_vector(2, n)", chained_code)   # original delta gone
    # each Base.merge returns a NEW model — the base is unchanged by the merges.
    # (crowdsource.qmd consumes each merged model directly via `stan_code(posterior)`,
    # which is exactly what the transpiles/stan_code checks above exercise.)
    base_code = stan_code(sm_splice_base(; n=3))
    @test occursin("lambda = rep_vector(1, n)", base_code)
    @test occursin("delta = rep_vector(2, n)", base_code)
end

"""
Verify `determinism: inline UDF + lifted closure` in an isolated test item.
"""
@testitem "determinism: inline UDF + lifted closure" tags=[:slic, :regression] setup=[StanBlocksImports, StanBlocksTestSetup] begin
    a = stan_code(stan_model(det_model))
    b = stan_code(stan_model(det_model))
    # non-vacuous: the model must actually exercise both counter-dependent sites
    @test occursin("__il_", a)
    @test occursin("// lifted closure", a)
    # transpiling twice in one session must be byte-identical
    @test a == b
end

"""
Verify `slic: ODE closure-captured param feeding a likelihood stays a parameter`
in an isolated test item.
"""
@testitem "slic: ODE closure-captured param feeding a likelihood stays a parameter" tags=[:slic, :regression, :bridgestan] setup=[StanBlocksImports, StanBlocksTestSetup] begin
    # Snag `ode-closure-capt`: a parameter captured by an ODE-RHS closure whose
    # solve feeds a downstream likelihood must remain a SAMPLED parameter. The
    # §9 prior-but-no-downstream-use → GQ `_rng` analysis must follow the closure
    # capture (the emitter threads it as a trailing solver arg), else `lambda` is
    # GQ-lowered yet used in `transformed parameters` → stanc "not in scope".
    @test stanc_compiles(det_lik_model)
    sc = stan_code(det_lik_model)
    # `lambda` stayed a sampled parameter: `~` in the model block, no GQ rng draw.
    @test occursin("lambda ~ std_normal()", sc)
    @test !occursin("std_normal_rng", sc)
    # Instantiating (not just transpiling) confirms it: `lambda` is the sole
    # sampler dimension and the density evaluates finitely.
    p = instantiate(stan_model(det_lik_model))
    @test LogDensityProblems.dimension(p) == 1
    @test isfinite(LogDensityProblems.logdensity(p, [0.5]))
end

"""
Verify `slic: ODE closure-captured DATA value is declared in the data block`
in an isolated test item.
"""
@testitem "slic: ODE closure-captured DATA value is declared in the data block" tags=[:slic, :regression, :bridgestan] setup=[StanBlocksImports, StanBlocksTestSetup] begin
    # Snag `ode-closure-capt` DATA-side twin: a data value (`k`) captured only
    # inside a lifted ODE-RHS closure is threaded as a trailing solver arg, so
    # `fetch_data!(closure)` must follow captures and declare it — else dead-data
    # elimination drops `real k;` and the solve references an out-of-scope `k`.
    @test stanc_compiles(det_datacap_model)
    sc = stan_code(det_datacap_model)
    @test occursin("real k;", sc)                     # k declared in the data block
    @test occursin("to_array_1d(ts), k", sc)          # and threaded into the solve
    # Instantiating confirms the data binding round-trips.
    p = instantiate(stan_model(det_datacap_model))
    @test LogDensityProblems.dimension(p) == 1
    @test isfinite(LogDensityProblems.logdensity(p, [0.5]))
end

"""
Verify `in-body @doc docstring renders (StanExpr unwrap)` in an isolated test item.
"""
@testitem "in-body @doc docstring renders (StanExpr unwrap)" tags=[:slic] setup=[StanBlocksImports, StanBlocksTestSetup] begin
    @test transpiles(doc_model)
    @test occursin("// documented local declaration", stan_code(stan_model(doc_model)))
end

"""
Verify `built-in constants resolve (pi, ℯ); arbitrary const errors` in an isolated test item.
"""
@testitem "built-in constants resolve (pi, ℯ); arbitrary const errors" tags=[:slic] setup=[StanBlocksImports, StanBlocksTestSetup] begin
    @test transpiles(consts_model)
    code = stan_code(stan_model(consts_model))
    @test occursin("3.14159", code)   # pi
    @test occursin("2.71828", code)   # ℯ
    # arbitrary module-level number is NOT a built-in constant → loud failure
    @test !transpiles(consts_bad_model; re=false)
end

"""
Verify scalar-array elementwise arithmetic against the shipped generalized lowering.
"""
@testitem "slic: scalar-array elementwise arithmetic" tags=[:slic, :regression, :shapes] setup=[StanBlocksImports, StanBlocksTestSetup] begin
    # Array-operand elementwise arithmetic on scalar arrays (`array[] int` /
    # `array[] real`) lowers to the `jbroadcasted` element loop and preserves
    # the element kind (int → array[] int, real → vector). Guarded by stanc,
    # NOT just `transpiles`:
    # the pre-fix bug transpiled PASS but stanc REJECTed (StanBlocks fcb0f64 / dfd588a).
    @test stanc_compiles(@slic (;Ks=[2,3,1]) begin d = Ks .+ 1; x ~ normal(1.0*sum(d), 1.) end)       # array .+ scalar
    @test stanc_compiles(@slic (;Ks=[2,3,1]) begin d = 1 .+ Ks; x ~ normal(1.0*sum(d), 1.) end)       # commutative scalar-first reorder
    @test stanc_compiles(@slic (;Ks=[2,3,1]) begin d = Ks .- 1; x ~ normal(1.0*sum(d), 1.) end)       # array .- scalar
    @test stanc_compiles(@slic (;Ks=[2,3,1]) begin d = Ks .* Ks; x ~ normal(1.0*sum(d), 1.) end)      # array .* array
    @test stanc_compiles(@slic (;rs=[1.0,2.0,3.0]) begin g = rs + rs; x ~ normal(1.0*sum(g), 1.) end) # real array + real array
    # reporter's flagship `fe - Ks + 1` on int-array operands (closes the int-array snag)
    @test stanc_compiles(@slic (;Ks=[2,3,1]) begin fe = cumulative_sum(Ks); fs = fe - Ks + 1; x ~ normal(1.0*sum(fs), 1.) end)
    # Integer elementwise arithmetic stays integer and remains index-usable.
    int_code = stan_code(@slic (;Ks=[2,3,1]) begin d = Ks .+ 1; x ~ normal(1.0*sum(d), 1.) end)
    @test occursin(r"array\[[^]]+\] int d", int_code)

    # Generalized lowering handles scalar-first dotted operators directly.
    @test stanc_compiles(@slic (;Ks=[2,3,1]) begin d = 1 ./ Ks; x ~ normal(sum(d), 1.) end)

    # Matmul/dimension operators do NOT lower and must reject loudly.
    @test !transpiles(@slic (;Ks=[2,3,1]) begin d = Ks * Ks; x ~ normal(1.0*sum(d), 1.) end; re=false)  # plain `*` = matmul, not elementwise
    @test !transpiles(@slic (;Ks=[2,3,1]) begin d = Ks / 2; x ~ normal(1.0*sum(d), 1.) end; re=false)   # plain `/`
    @test !transpiles(@slic (;Ks=[2,3,1]) begin d = Ks ^ 2; x ~ normal(1.0*sum(d), 1.) end; re=false)   # plain `^`

    # Vector arithmetic (scalar⊗vector, vector⊗vector) is unaffected by the new dispatch:
    @test transpiles(@slic (;) begin a ~ std_normal(; n=3); c = a .* a; y ~ normal(sum(c), 1.) end)
    @test transpiles(@slic (;) begin a ~ std_normal(; n=3); c = 3*a; y ~ normal(sum(c), 1.) end)
end

"""
Verify `issue17` in an isolated test item.
"""
@testitem "issue17" tags=[:slic, :regression] setup=[StanBlocksImports, StanBlocksTestSetup] begin
    @test stanc_compiles(@slic (;n=10, y=1.) begin
        x ~ issue17(;n)
        y ~ simple(x)
    end)
end

"""
Verify `issue18` in an isolated test item.
"""
@testitem "issue18" tags=[:slic, :regression] setup=[StanBlocksImports, StanBlocksTestSetup] begin
    @test stanc_compiles(@slic (;n=10, y=1.) begin
        x ~ issue18(;n)
        y ~ simple(x)
    end)
end

"""
Verify `issue19` in an isolated test item.
"""
@testitem "issue19" tags=[:slic, :regression] setup=[StanBlocksImports, StanBlocksTestSetup] begin
    @test stanc_compiles(@slic (;n=10, y=1.) begin
        x ~ sm19a(;n)
        y ~ simple(x)
    end)
    @test stanc_compiles(@slic (;n=10, y=1.) begin
        x ~ sm19b(;n)
        y ~ simple(x)
    end)
end

"""
Verify `issue20` in an isolated test item.
"""
@testitem "issue20" tags=[:slic, :regression] setup=[StanBlocksImports, StanBlocksTestSetup] begin
    @test stanc_compiles(m20.model(;n=10))
    @test stanc_compiles(m20.modela(;n=10))
    @test stanc_compiles(m20.modelb(;n=10))
end

"""
Verify `partly-missing: basic scalar-dist transpiles` in an isolated test item.
"""
@testitem "partly-missing: basic scalar-dist transpiles" tags=[:slic, :missing] setup=[StanBlocksImports, StanBlocksTestSetup] begin
    @test transpiles(@slic (;y=[1.0, missing, 3.0, missing, 5.0]) begin
        mu    ~ normal(0., 10.)
        sigma ~ gamma(2., 1.)
        y     ~ normal(mu, sigma)
    end)
end

"""
Verify the partly-missing scalar-distribution model is accepted by stanc.
"""
@testitem "partly-missing: basic scalar-dist passes stanc" tags=[:slic, :missing, :stanc] setup=[StanBlocksImports, StanBlocksTestSetup] begin
    @test stanc_compiles(@slic (;y=[1.0, missing, 3.0, missing, 5.0]) begin
        mu    ~ normal(0., 10.)
        sigma ~ gamma(2., 1.)
        y     ~ normal(mu, sigma)
    end)
end

"""
Verify `partly-missing: logdensity equals obs-only model` in an isolated test item.
"""
@testitem "partly-missing: logdensity equals obs-only model" tags=[:slic, :missing, :bridgestan] setup=[StanBlocksImports, StanBlocksTestSetup] begin
    # y_mis goes to generated_quantities (GQ rng draw), not sampler params.
    # So the sampler dimension is 2: mu (unconstrained) + log_sigma (lower=0).
    # Rigorous check: logdensity must equal an explicit obs-only model at the
    # same unconstrained params — proves missing entries are excluded correctly.
    m_miss = @slic (;y=[1., missing, 3., missing, 5.]) begin
        mu    ~ normal(0., 10.)
        sigma ~ gamma(2., 1.)
        y     ~ normal(mu, sigma)
    end
    m_obs = @slic (;y=[1., 3., 5.]) begin
        mu    ~ normal(0., 10.)
        sigma ~ gamma(2., 1.)
        y     ~ normal(mu, sigma)
    end
    p_miss = instantiate(stan_model(m_miss))
    p_obs  = instantiate(stan_model(m_obs))
    @test LogDensityProblems.dimension(p_miss) == 2
    @test LogDensityProblems.dimension(p_obs)  == 2
    for v in [[-1.0, 0.5], [0.0, 0.0], [1.5, -0.3]]
        @test LogDensityProblems.logdensity(p_miss, v) ≈
              LogDensityProblems.logdensity(p_obs,  v) atol=1e-6
    end
    # y_mis appears in generated quantities (GQ rng draw), not parameters
    sc = stan_code(m_miss)
    @test occursin("generated quantities", sc) && occursin("y_mis", sc)
    @test occursin("merge_missing", sc)
end

"""
Verify `partly-missing: vector dist arg (regression check)` in an isolated test item.
"""
@testitem "partly-missing: vector dist arg (regression check)" tags=[:slic, :regression, :missing, :bridgestan] setup=[StanBlocksImports, StanBlocksTestSetup] begin
    # This exercises the getindex branch of maybe_index — previously broken
    # because the node was built with Symbol :getindex instead of Function.
    m = @slic (;x=collect(1.:6.), y=[1., missing, 3., missing, 5., missing]) begin
        a     ~ normal(0., 1.)
        b     ~ normal(0., 1.)
        sigma ~ gamma(2., 1.)
        y     ~ normal(a .+ b .* x, sigma)
    end
    @test stanc_compiles(m)
    @test LogDensityProblems.dimension(instantiate(stan_model(m))) == 3
    sc = stan_code(m)
    @test occursin("y_ii_obs", sc) && occursin("y_ii_mis", sc)
end

"""
Verify `partly-missing: error on joint dist` in an isolated test item.
"""
@testitem "partly-missing: error on joint dist" tags=[:slic, :missing] setup=[StanBlocksImports, StanBlocksTestSetup] begin
    @test_throws Exception stan_model(@slic (;y=[1.0, missing, 3.0]) begin
        mu  ~ std_normal(;n=3)
        cov = diag_matrix(rep_vector(1., 3))
        y   ~ multi_normal(mu, cov)
    end)
end

"""
Verify `partly-missing: error on missing not used as LHS` in an isolated test item.
"""
@testitem "partly-missing: error on missing not used as LHS" tags=[:slic, :missing] setup=[StanBlocksImports, StanBlocksTestSetup] begin
    @test_throws Exception stan_model(@slic (;y=[1.0, missing, 3.0]) begin
        mu ~ normal(0., 10.)
    end)
end

"""
Verify `partly-missing: regression — all-observed vector unaffected` in an isolated test item.
"""
@testitem "partly-missing: regression — all-observed vector unaffected" tags=[:slic, :regression, :missing] setup=[StanBlocksImports, StanBlocksTestSetup] begin
    @test stanc_compiles(@slic (;y=[1.0, 2.0, 3.0]) begin
        mu    ~ normal(0., 10.)
        sigma ~ gamma(2., 1.)
        y     ~ normal(mu, sigma)
    end)
end

"""
Verify `logdensity: unconstrained scalar` in an isolated test item.
"""
@testitem "logdensity: unconstrained scalar" tags=[:slic, :logdensity, :bridgestan] setup=[StanBlocksImports, StanBlocksTestSetup] begin
    obs_val = 0.5
    problem = instantiate(stan_model(@slic (;obs=obs_val) begin
        mu ~ std_normal()
        obs ~ normal(mu, 1.0)
    end))
    @test LogDensityProblems.dimension(problem) == 1
    for mu_val in [-2.0, 0.0, 1.5]
        expected = _stan_std_normal(mu_val) + _stan_normal(obs_val, mu_val, 1.0)
        @test LogDensityProblems.logdensity(problem, [mu_val]) ≈ expected atol=1e-6
    end
end

"""
Verify `logdensity: lower-bounded scalar` in an isolated test item.
"""
@testitem "logdensity: lower-bounded scalar" tags=[:slic, :logdensity, :bridgestan] setup=[StanBlocksImports, StanBlocksTestSetup] begin
    obs_val = 1.0
    problem = instantiate(stan_model(@slic (;obs=obs_val) begin
        sigma ~ std_normal(;lower=0.)
        obs ~ normal(0., sigma)
    end))
    @test LogDensityProblems.dimension(problem) == 1
    for θ_val in [-1.0, 0.0, 1.0]
        sigma_val = exp(θ_val)
        expected = (
            _stan_std_normal(sigma_val) + θ_val
            + _stan_normal(obs_val, 0.0, sigma_val)
        )
        @test LogDensityProblems.logdensity(problem, [θ_val]) ≈ expected atol=1e-6
    end
end

"""
Verify `logdensity: interval-bounded scalar (beta prior)` in an isolated test item.
"""
@testitem "logdensity: interval-bounded scalar (beta prior)" tags=[:slic, :logdensity, :bridgestan] setup=[StanBlocksImports, StanBlocksTestSetup] begin
    k_val, n_val = 3, 10
    problem = instantiate(stan_model(@slic (;k=k_val, n=n_val) begin
        theta ~ beta(1., 1.)
        k ~ binomial(n, theta)
    end))
    @test LogDensityProblems.dimension(problem) == 1
    for θ_val in [-2.0, 0.0, 1.5]
        p = 1.0 / (1.0 + exp(-θ_val))
        log_jac = log(p) + log(1.0 - p)
        expected = (
            0.0 + log_jac
            + _stan_binomial(k_val, n_val, p)
        )
        @test LogDensityProblems.logdensity(problem, [θ_val]) ≈ expected atol=1e-6
    end
end

"""
Verify `logdensity: unconstrained vector` in an isolated test item.
"""
@testitem "logdensity: unconstrained vector" tags=[:slic, :logdensity, :bridgestan] setup=[StanBlocksImports, StanBlocksTestSetup] begin
    obs_val = [0.5, -0.3, 1.2]
    n = length(obs_val)
    problem = instantiate(stan_model(@slic (;obs=obs_val) begin
        mu ~ std_normal(;n=length(obs))
        obs ~ normal(mu, 1.0)
    end))
    @test LogDensityProblems.dimension(problem) == n
    mu_val = [0.1, -0.5, 0.8]
    expected = (
        sum(_stan_std_normal.(mu_val))
        + sum(_stan_normal.(obs_val, mu_val, 1.0))
    )
    @test LogDensityProblems.logdensity(problem, mu_val) ≈ expected atol=1e-6
end

"""
Verify `logdensity: linear regression, flat prior` in an isolated test item.
"""
@testitem "logdensity: linear regression, flat prior" tags=[:slic, :regression, :logdensity, :bridgestan] setup=[StanBlocksImports, StanBlocksTestSetup] begin
    x_val = [1.0, 2.0, 3.0]
    y_val = [2.1, 3.9, 6.2]
    problem = instantiate(stan_model(@slic (;y=y_val, x=x_val) begin
        alpha ~ flat()
        beta  ~ flat()
        y ~ normal(alpha + beta * to_vector(x), 1.0)
    end))
    @test LogDensityProblems.dimension(problem) == 2
    for (alpha_val, beta_val) in [(0.0, 2.0), (1.0, 1.5), (-0.5, 2.1)]
        predicted = alpha_val .+ beta_val .* x_val
        expected  = sum(_stan_normal.(y_val, predicted, 1.0))
        @test LogDensityProblems.logdensity(problem, [alpha_val, beta_val]) ≈ expected atol=1e-6
    end
end

"""
Verify `slic: standalone bare typed model parameters` in an isolated test item.
"""
@testitem "slic: standalone bare typed model parameters" tags=[:slic, :bridgestan] setup=[StanBlocksImports, StanBlocksTestSetup] begin
    y = [0.2, -0.1, 0.4]
    model = @slic (;y) begin
        alpha::real
        beta::vector[3]
        unused::real
        mu = bare_parameter_local_copy(alpha + beta)
        y ~ normal(mu, 1.0)
    end

    @test transpiles(model)
    @test stanc_compiles(model)
    code = stan_code(model)
    let parameters = stan_block(code, "parameters")
        @test occursin(r"\breal alpha;", parameters)
        @test occursin(r"\bvector\[3\] beta;", parameters)
        @test occursin(r"\breal unused;", parameters)
        @test !occursin("out", parameters)
    end
    let functions = stan_block(code, "functions")
        @test occursin("bare_parameter_local_copy", functions)
        @test occursin(r"\bvector\[n\] out;", functions)
    end
    @test !occursin("flat", code)
    @test !occursin(r"\b(alpha|beta|unused)\s*~", stan_block(code, "model"))

    problem = instantiate(stan_model(model))
    @test LogDensityProblems.dimension(problem) == 5
    theta = [0.25, -0.1, 0.2, 0.4, 17.0]
    expected = sum(_stan_normal.(y, theta[1] .+ theta[2:4], 1.0))
    @test LogDensityProblems.logdensity(problem, theta) ≈ expected atol=1e-6

    embedded = @slic (;y=0.3) begin
        mu ~ bare_parameter_submodel
        y ~ normal(mu, 1.0)
    end
    @test transpiles(embedded)
    @test stanc_compiles(embedded)
    embedded_code = stan_code(embedded)
    @test occursin(r"\breal mu_offset;", stan_block(embedded_code, "parameters"))
    @test !occursin(r"\bmu_offset\s*~", stan_block(embedded_code, "model"))

    # User-authored model-body declare-then-fill stays rejected before `forward!`;
    # only compiler-injected indexed fills carry the reclassification certificate.
    bad = @slic (;) begin
        target::vector[2]
        target[1] = 0.0
    end
    err = try
        stan_code(bad)
        nothing
    catch e
        sprint(showerror, e)
    end
    @test err !== nothing
    @test occursin("Indexed/slice assignment", err)
end

"""
Verify `logdensity: hierarchical (normal-normal)` in an isolated test item.
"""
@testitem "logdensity: hierarchical (normal-normal)" tags=[:slic, :logdensity, :bridgestan] setup=[StanBlocksImports, StanBlocksTestSetup] begin
    n = 3
    y_val = [0.2, -0.1, 0.8]
    problem = instantiate(stan_model(@slic (;y=y_val) begin
        mu    ~ std_normal()
        sigma ~ std_normal(;lower=0.)
        theta ~ normal(mu, sigma; n=length(y))
        y     ~ normal(theta, 1.)
    end))
    @test LogDensityProblems.dimension(problem) == n + 2
    mu_unc    = 0.3
    log_sigma = 0.1
    theta_unc = [0.5, -0.2, 0.6]
    θ = [mu_unc; log_sigma; theta_unc]
    sigma_val = exp(log_sigma)
    expected = (
        _stan_std_normal(mu_unc)
        + _stan_std_normal(sigma_val) + log_sigma
        + sum(_stan_normal.(theta_unc, mu_unc, sigma_val))
        + sum(_stan_normal.(y_val, theta_unc, 1.0))
    )
    @test LogDensityProblems.logdensity(problem, θ) ≈ expected atol=1e-6
end

"""
Verify `logdensity: dimension from data` in an isolated test item.
"""
@testitem "logdensity: dimension from data" tags=[:slic, :logdensity, :bridgestan] setup=[StanBlocksImports, StanBlocksTestSetup] begin
    for n in [1, 5, 20]
        obs = randn(n)
        problem = instantiate(stan_model(@slic (;obs=obs) begin
            mu ~ std_normal(;n=length(obs))
            obs ~ normal(mu, 1.0)
        end))
        @test LogDensityProblems.dimension(problem) == n
    end
end

"""
Verify `shapes: vector creation` in an isolated test item.
"""
@testitem "shapes: vector creation" tags=[:slic, :shapes, :stanc] setup=[StanBlocksImports, StanBlocksTestSetup] begin
    @testset "rep_vector(v, n) :: vector[n]" begin
        model = @slic (;n=5, obs=randn(5)) begin
            v   = rep_vector(0., n)
            obs ~ normal(v, 1.)
        end
        @test check_type(model, :v, "vector")
        @test stanc_compiles(model)
    end
    @testset "linspaced_vector(n, lo, hi) :: vector[n]" begin
        model = @slic (;n=5, obs=randn(5)) begin
            v   = linspaced_vector(n, 0., 1.)
            obs ~ normal(v, 1.)
        end
        @test check_type(model, :v, "vector")
        @test stanc_compiles(model)
    end
end

"""
Verify `shapes: matrix creation` in an isolated test item.
"""
@testitem "shapes: matrix creation" tags=[:slic, :shapes, :stanc] setup=[StanBlocksImports, StanBlocksTestSetup] begin
    @testset "rep_matrix(x::real, m, n) :: matrix[m,n]" begin
        model = @slic (;m=3, n=4, obs=randn(3)) begin
            A   = rep_matrix(0., m, n)
            obs ~ normal(A * rep_vector(0., n), 1.)
        end
        @test check_type(model, :A, "matrix")
        @test stanc_compiles(model)
    end
    @testset "rep_matrix(v::vector[m], n) :: matrix[m,n]" begin
        model = @slic (;m=3, n=4, obs=randn(3)) begin
            v   = rep_vector(1., m)
            A   = rep_matrix(v, n)
            obs ~ normal(A * rep_vector(0., n), 1.)
        end
        @test check_type(model, :A, "matrix")
        @test stanc_compiles(model)
    end
    @testset "diag_matrix(v::vector[n]) :: matrix[n,n]" begin
        model = @slic (;n=4, obs=randn(4)) begin
            v   = rep_vector(2., n)
            D   = diag_matrix(v)
            obs ~ normal(D * rep_vector(1., n), 1.)
        end
        @test check_type(model, :D, "matrix")
        @test stanc_compiles(model)
    end
    @testset "to_matrix(v, m, n) :: matrix[m,n]" begin
        model = @slic (;m=2, n=3, obs=randn(2)) begin
            A   = to_matrix(rep_vector(0., m*n), m, n)
            obs ~ normal(A * rep_vector(0., n), 1.)
        end
        @test check_type(model, :A, "matrix")
        @test stanc_compiles(model)
    end
end

"""
Verify `shapes: conversions` in an isolated test item.
"""
@testitem "shapes: conversions" tags=[:slic, :shapes, :stanc] setup=[StanBlocksImports, StanBlocksTestSetup] begin
    @testset "to_vector(x) :: vector[n]" begin
        model = @slic (;x=randn(5), obs=randn(5)) begin
            v   = to_vector(x)
            obs ~ normal(v, 1.)
        end
        @test check_type(model, :v, "vector")
        @test stanc_compiles(model)
    end
    @testset "to_row_vector(x) :: row_vector[n]" begin
        model = @slic (;x=randn(5)) begin rv = to_row_vector(x) end
        @test transpiles(model)
        @test check_type(model, :rv, "row_vector")
    end
end

"""
Verify `shapes: append functions` in an isolated test item.
"""
@testitem "shapes: append functions" tags=[:slic, :shapes, :stanc] setup=[StanBlocksImports, StanBlocksTestSetup] begin
    @testset "append_col(v::vector[n], w::vector[n]) :: matrix[n,2]" begin
        model = @slic (;n=4, obs=randn(4)) begin
            v   = rep_vector(0., n)
            w   = rep_vector(1., n)
            A   = append_col(v, w)
            obs ~ normal(A * rep_vector(0., 2), 1.)
        end
        @test check_type(model, :A, "matrix")
        @test stanc_compiles(model)
    end
    @testset "append_col(A::matrix[m,n1], B::matrix[m,n2]) :: matrix[m,n1+n2]" begin
        model = @slic (;m=3, n1=2, n2=3, obs=randn(3)) begin
            A1  = rep_matrix(0., m, n1)
            A2  = rep_matrix(0., m, n2)
            C   = append_col(A1, A2)
            obs ~ normal(C * rep_vector(0., n1+n2), 1.)
        end
        @test stanc_compiles(model)
    end
    @testset "append_row(v::vector[m], x::real) :: vector[m+1]" begin
        model = @slic (;n=4, obs=randn(5)) begin
            v   = rep_vector(0., n)
            v2  = append_row(v, 1.)
            obs ~ normal(v2, 1.)
        end
        @test check_type(model, :v2, "vector")
        @test stanc_compiles(model)
    end
    @testset "append_row(v1::vector[m], v2::vector[n]) :: vector[m+n]" begin
        model = @slic (;m=3, n=2, obs=randn(5)) begin
            v1  = rep_vector(0., m)
            v2  = rep_vector(1., n)
            v3  = append_row(v1, v2)
            obs ~ normal(v3, 1.)
        end
        @test check_type(model, :v3, "vector")
        @test stanc_compiles(model)
    end
end

"""
Verify `shapes: linear algebra` in an isolated test item.
"""
@testitem "shapes: linear algebra" tags=[:slic, :shapes, :stanc] setup=[StanBlocksImports, StanBlocksTestSetup] begin
    @testset "dot_product(v, w) :: real" begin
        model = @slic (;n=4, obs=0.) begin
            v   = rep_vector(1., n)
            obs ~ normal(dot_product(v, v), 1.)
        end
        @test stanc_compiles(model)
    end
    @testset "rows_dot_product(A, B) :: vector[m]" begin
        let model = @slic (;m=3, n=4) begin
                A = rep_matrix(1., m, n)
                d = rows_dot_product(A, A)
            end
        @test check_type(model, :d, "vector")
        end
        @test stanc_compiles(@slic (;m=3, n=4, obs=randn(3)) begin
            A   = rep_matrix(1., m, n)
            obs ~ normal(rows_dot_product(A, A), 1.)
        end)
    end
    @testset "cumulative_sum(v) :: vector[n]" begin
        let model = @slic (;n=5) begin v = rep_vector(1., n); cs = cumulative_sum(v) end
        @test check_type(model, :cs, "vector")
        end
        @test stanc_compiles(@slic (;n=5, obs=randn(5)) begin
            v   = rep_vector(1., n)
            obs ~ normal(cumulative_sum(v), 1.)
        end)
    end
    @testset "mdivide_left_tri_low(L, b) :: vector[n]" begin
        model = @slic (;n=3, obs=randn(3)) begin
            L   = diag_matrix(rep_vector(1., n))
            b   = rep_vector(1., n)
            obs ~ normal(mdivide_left_tri_low(L, b), 1.)
        end
        let m2 = @slic (;n=3) begin L = diag_matrix(rep_vector(1., n)); x = mdivide_left_tri_low(L, rep_vector(1., n)) end
        @test check_type(m2, :x, "vector")
        end
        @test stanc_compiles(model)
    end
    @testset "diag_pre/post_multiply" begin
        @test stanc_compiles(@slic (;m=3, n=4, obs=randn(3)) begin
            v   = rep_vector(1., m)
            A   = rep_matrix(1., m, n)
            obs ~ normal(diag_pre_multiply(v, A) * rep_vector(0., n), 1.)
        end)
        @test stanc_compiles(@slic (;m=3, n=4, obs=randn(3)) begin
            v   = rep_vector(1., n)
            A   = rep_matrix(1., m, n)
            obs ~ normal(diag_post_multiply(A, v) * rep_vector(0., n), 1.)
        end)
    end
end

"""
Verify `shapes: introspection` in an isolated test item.
"""
@testitem "shapes: introspection" tags=[:slic, :shapes, :stanc] setup=[StanBlocksImports, StanBlocksTestSetup] begin
    @testset "dims(v) :: int[1]" begin
        model = @slic (;n=4, obs=randn(4)) begin
            v   = rep_vector(0., n)
            w   = rep_vector(1., dims(v)[1])
            obs ~ normal(w, 1.)
        end
        @test stanc_compiles(model)
    end
    @testset "rows / cols" begin
        @test stanc_compiles(@slic (;m=3, n=4, obs=0.) begin
            A   = rep_matrix(0., m, n)
            obs ~ normal(rows(A) + cols(A), 1.)
        end)
    end
end

"""
Verify `shapes: scalar math` in an isolated test item.
"""
@testitem "shapes: scalar math" tags=[:slic, :shapes, :stanc] setup=[StanBlocksImports, StanBlocksTestSetup] begin
    @testset "inv_logit / logit" begin
        @test stanc_compiles(@slic (;obs=0.) begin
            x   = inv_logit(0.)
            obs ~ normal(x, 1.)
        end)
        @test stanc_compiles(@slic (;obs=0.) begin
            x   = logit(0.5)
            obs ~ normal(x, 1.)
        end)
    end
    @testset "log1m" begin
        @test stanc_compiles(@slic (;obs=0.) begin
            x   = log1m(0.5)
            obs ~ normal(x, 1.)
        end)
    end
    @testset "log_inv_logit" begin
        @test stanc_compiles(@slic (;obs=0.) begin
            x   = log_inv_logit(0.)
            obs ~ normal(x, 1.)
        end)
    end
    @testset "log1m_exp" begin
        @test stanc_compiles(@slic (;obs=0.) begin
            x   = log1m_exp(-1.)
            obs ~ normal(x, 1.)
        end)
    end
end

"""
Verify `shapes: arrays` in an isolated test item.
"""
@testitem "shapes: arrays" tags=[:slic, :shapes, :stanc] setup=[StanBlocksImports, StanBlocksTestSetup] begin
    @testset "rep_array(x::real, n) :: real[n]" begin
        let model = @slic (;n=5) begin a = rep_array(1., n) end
        @test transpiles(model)
        @test check_type(model, :a, "array")
        end
    end
    @testset "to_array_1d" begin
        @test transpiles(@slic (;x=randn(3)) begin
            a = to_array_1d(to_vector(x))
        end)
    end
    @testset "append_array(a, b) :: real[m+n]" begin
        @test transpiles(@slic (;n=3, m=2) begin
            a = rep_array(1., n)
            b = rep_array(2., m)
            c = append_array(a, b)
        end)
    end
end

"""
Verify `slic: scalar-array elementwise broadcasting (jbroadcasted)` in an isolated test item.
"""
@testitem "slic: scalar-array elementwise broadcasting (jbroadcasted)" tags=[:slic] setup=[StanBlocksImports, StanBlocksTestSetup] begin
    # Generalised trace-level `jbroadcasted` (f4b601f): elementwise arithmetic on
    # scalar arrays (`array[] int` / `array[] real` — what a `Vector{Int}` /
    # `Vector{Float64}` data input becomes; NOT a native `vector`) has no Stan
    # operator, so it lowers to an element loop whose output CONTAINER is inferred
    # from `f`'s per-element return type — `int` stays `array[] int` (usable as an
    # index), `real` → `vector[n]`. Native `vector` operands keep Stan's built-in
    # scalar-vector ops (no lowering). GATE ON `compiles`, NOT `transpiles`: the
    # pre-generalisation bug transpiled PASS but stanc REJECTed (StanBlocks primer,
    # `4b45a5ac` scalar-array section).
    @testset "int array stays int (reporter: fs = fe - Ks + 1)" begin
        model = @slic (;fe=[3,4,5], Ks=[1,1,1]) begin
            fs = fe - Ks + 1
            mu ~ std_normal()
            mu
        end
        code = stan_code(model)
        # int container preserved — NOT silently coerced to a real `vector`
        @test occursin(r"array\[\w*\] int (?:\w+ )?fs\b", code)
        @test !occursin(r"vector\[\w*\] (?:\w+ )?fs\b", code)
        @test occursin("jbroadcasted_add", code)
        @test occursin("jbroadcasted_sub", code)
        @test stanc_compiles(model)
    end
    @testset "real scalar-array → vector" begin
        model = @slic (;n=4) begin
            ra = rep_array(1.5, n)   # array[n] real
            z  = 2.0 .- ra           # scalar-array real → jbroadcasted → vector
            mu ~ std_normal()
            mu
        end
        @test occursin("jbroadcasted_sub", stan_code(model))
        @test check_type(model, :z, "vector")
        @test stanc_compiles(model)
    end
    @testset "scalar-first ./ and .^ on a real array lower (any arg position)" begin
        # Previously fell through to the loud reject (no commute/negate identity);
        # the generalised any-position jbroadcasted lowers them directly.
        model = @slic (;n=4) begin
            ra = rep_array(2.0, n)
            q  = 3.0 ./ ra
            p  = 3.0 .^ ra
            mu ~ std_normal()
            mu
        end
        @test occursin("jbroadcasted", stan_code(model))
        @test stanc_compiles(model)
    end
    @testset "native vector operand keeps built-in scalar-vector op (no lowering)" begin
        model = @slic (;w=[1.0,2.0,3.0]) begin   # Vector{Float64} → Stan vector
            z = 2.0 .- w
            mu ~ std_normal()
            mu
        end
        @test !occursin("jbroadcasted", stan_code(model))
        @test stanc_compiles(model)
    end
    @testset "regression: binomial_lpmfs caller keeps real vector[n]" begin
        # jbroadcasted backs every `*_lpmfs` distribution — the container inference
        # must keep the real-element path a `vector[n]`, not flip it to `array[] int`.
        model = @slic (;y=[0,1,1,0,1], N=[2,2,2,2,2]) begin
            p ~ std_normal(;lower=0., upper=1.)
            y ~ binomial(N, p)
            p
        end
        @test occursin(r"vector\[\w*\] (?:\w+ )?y_likelihood\b", stan_code(model))
        @test stanc_compiles(model)
    end
    @testset "reject floor: plain * / ^ on scalar arrays still loudly rejected" begin
        # Matmul/dim-shaped `*` on arrays is NOT elementwise — must stay rejected,
        # never silently miscompiled.
        @test_throws Exception stan_code(@slic (;a=[1,2,3], b=[1,2,3]) begin
            c = a * b
            mu ~ std_normal()
            mu
        end)
    end
end

@testitem "slic: boolean-mask indexing via comparison broadcast + findall" tags=[:slic, :bridgestan] setup=[StanBlocksImports, StanBlocksTestSetup] begin
    # Stan has no boolean-mask indexing (`v[mask]`). Instead, element-wise
    # comparison on a DATA scalar array (`cmt .== 1`, cmt an `array[] int`) lowers
    # via `jbroadcasted` to a 0/1 `array[] int` mask, and `findall(mask)`
    # materialises the true-positions as a data-sized `int[sum(mask)]` array in
    # transformed data. `y[findall(cmt .== 1)]` is then ordinary integer-array
    # indexing — the mask precomputes for free (cmt is data) and one index array is
    # shared by every use at zero runtime/gradient cost. GATE ON `compiles`.
    @testset "BRM-shaped: y[findall(cmt .== 1)] ~ normal(mu[...], sd)" begin
        # `sd` is lower-bounded so any unconstrained test point maps to a valid
        # (positive) normal scale — the gradient check below must be deterministic.
        model = @slic (;n=8, cmt=[1,2,1,2,1,2,1,2], y=randn(8), mu=randn(8)) begin
            idx = findall(cmt .== 1)
            sd ~ std_normal(;lower=0.)
            y[idx] ~ normal(mu[idx], sd)
        end
        code = stan_code(model)
        # mask lowers to an int-container jbroadcasted; idx is data-sized in tdata.
        @test occursin("jbroadcasted_eq", code)
        @test occursin(r"array\[[^\]]*\] int idx = findall\(", code)
        @test occursin("y[idx] ~ normal(mu[idx]", code)
        @test stanc_compiles(model)
        # gradient-correct end to end (BridgeStan), at a fixed valid point.
        prob = instantiate(model)
        lp, g = LogDensityProblems.logdensity_and_gradient(prob, [0.4])
        @test isfinite(lp) && all(isfinite, g)
    end
    @testset "all six comparison operators broadcast to an int mask" begin
        for form in (
            :(findall(cmt .== 2)), :(findall(cmt .!= 2)), :(findall(cmt .< 2)),
            :(findall(cmt .<= 2)), :(findall(cmt .> 2)), :(findall(cmt .>= 2)),
        )
            body = Expr(:block, Expr(:(=), :idx, form), :(z::real ~ normal(0,1)))
            model = StanBlocks.stan.SlicModel(body, (;n=6, cmt=[1,2,3,1,2,3]), @__MODULE__)
            @test stanc_compiles(model)
        end
    end
    @testset "mask usable as a plain count: sum(cmt .== 1)" begin
        model = @slic (;n=6, cmt=[1,2,1,2,1,2]) begin
            k = sum(cmt .== 1)
            z::real ~ normal(0,1)
            z
        end
        @test stanc_compiles(model)
    end
    @testset "element-wise comparison on a native vector fails loudly (no silent miscompile)" begin
        # `vector .> scalar` has no int-mask lowering (only scalar arrays do); it
        # must error, never emit invalid Stan silently.
        @test_throws Exception stan_code(@slic (;n=6, x=randn(6)) begin
            idx = findall(x .> 0.5)
            z::real ~ normal(0,1)
            z
        end)
    end
    @testset "bool-type auto-sugar: y[cmt .== 1] indexes directly with the mask" begin
        # A comparison broadcast yields a `bool[n]` (a subtype of `int`), and
        # `getindex` on a `bool[n]` mask lowers to `v[findall(mask)]` — so the
        # full `y[cmt .== 1] ~ normal(mu[cmt .== 1], sd)` sugar works with no
        # explicit `findall`. The findall is HOISTED to a transformed-data binding
        # (zero per-iteration cost), and the model block is plain integer indexing.
        model = @slic (;n=8, cmt=[1,2,1,2,1,2,1,2], y=randn(8), mu=randn(8)) begin
            sd ~ std_normal(;lower=0.)
            y[cmt .== 1] ~ normal(mu[cmt .== 1], sd)
        end
        code = stan_code(model)
        @test occursin(r"array\[[^\]]*\] int boolmask_idx_\d+ = findall\(", code)   # hoisted to tdata
        @test occursin(r"y\[boolmask_idx_\d+\] ~ normal\(mu\[boolmask_idx_\d+\]", code)
        @test !occursin("findall", split(code, "model {")[2])   # NOT recomputed in the model block
        @test stanc_compiles(model)
        # Auto-sugar is numerically identical to the explicit findall form.
        explicit = @slic (;n=8, cmt=[1,2,1,2,1,2,1,2], y=randn(8), mu=randn(8)) begin
            idx = findall(cmt .== 1)
            sd ~ std_normal(;lower=0.)
            y[idx] ~ normal(mu[idx], sd)
        end
        # same data seed so the two log-densities must coincide
        Random.seed!(7); cmt=[1,2,1,2,1,2,1,2]; y=randn(8); mu=randn(8)
        ms = @slic (;n=8, cmt=cmt, y=y, mu=mu) begin
            sd ~ std_normal(;lower=0.); y[cmt .== 1] ~ normal(mu[cmt .== 1], sd)
        end
        me = @slic (;n=8, cmt=cmt, y=y, mu=mu) begin
            idx = findall(cmt .== 1); sd ~ std_normal(;lower=0.); y[idx] ~ normal(mu[idx], sd)
        end
        ps = instantiate(ms); pe = instantiate(me)
        xx = [0.3]
        @test LogDensityProblems.logdensity(ps, xx) ≈ LogDensityProblems.logdensity(pe, xx)
        g = LogDensityProblems.logdensity_and_gradient(ps, xx)[2]
        @test all(isfinite, g)
    end
    @testset "bool is an int everywhere except getindex" begin
        # sum(mask) counts, mask arithmetic stays int — `bool <: int`.
        model = @slic (;n=6, cmt=[1,2,1,2,1,2]) begin
            k = sum(cmt .== 1)          # bool[] sums as int
            m2 = (cmt .== 1) + (cmt .== 2)   # int arithmetic on masks
            z::real ~ normal(0,1)
            z
        end
        @test stanc_compiles(model)
    end
end

"""
Verify `slic: bounded one-dimensional @deffun comprehensions` in an isolated test item.
"""
@testitem "slic: bounded one-dimensional @deffun comprehensions" tags=[:slic, :shapes] setup=[StanBlocksImports, StanBlocksTestSetup] begin
    @testset "canonicalization preserves the comprehension/generator structure" begin
        c = StanBlocks.stan.canonical(Meta.parse("[x[i] * x[i] for i in 1:n]"))
        @test c isa StanBlocks.stan.ComprehensionExpr
        @test length(c.args) == 1
        @test c.args[1] isa StanBlocks.stan.GeneratorExpr
        @test length(c.args[1].args) == 2
        @test c.args[1].args[2] isa StanBlocks.stan.AssignmentExpr
    end

    @testset "real and int comprehensions lower to typed locals + symbolic loops" begin
        real_model = @slic (;x=[1.0, 2.0, 3.0], y=[1.0, 4.0, 9.0]) begin
            mu = comprehension_square(x)
            y ~ normal(mu, 1.0)
        end
        real_code = stan_code(real_model)
        @test occursin(r"vector\[[^]]+\] comprehension_result__lc_\d+;", real_code)
        @test occursin(r"for\(i in 1:n\)", real_code)
        @test occursin(r"comprehension_result__lc_\d+\[i\] = \(x\[i\] \* x\[i\]\)", real_code)
        @test stanc_compiles(real_model)

        int_model = @slic (;x=[1, 2, 3]) begin
            shifted = comprehension_int_shift(x)
            mu ~ normal(shifted[1], 1.0)
        end
        int_code = stan_code(int_model)
        @test occursin(r"array\[[^]]+\] int comprehension_result__lc_\d+;", int_code)
        @test stanc_compiles(int_model)
    end

    @testset "non-1 lower bound uses a dense output index" begin
        model = @slic (;x=[1.0, 2.0, 3.0, 4.0], y=[2.0, 3.0, 4.0]) begin
            mu = comprehension_slice(x, 2, 4)
            y ~ normal(mu, 1.0)
        end
        code = stan_code(model)
        @test occursin(r"for\(i in lo:hi\)", code)
        @test occursin(r"comprehension_result__lc_\d+\[\(\(i - lo\) \+ 1\)\] = x\[i\]", code)
        @test stanc_compiles(model)
    end

    comprehension_error(f) = try
        stan_code(f())
        nothing
    catch e
        sprint(showerror, e)
    end

    @testset "unsupported generator forms reject explicitly" begin
        filtered = comprehension_error() do
            @slic (;x=[-1.0, 2.0]) begin
                z = comprehension_filtered(x)
                mu ~ normal(z[1], 1.0)
            end
        end
        @test filtered !== nothing
        @test occursin("filtered generators are not supported", filtered)

        nested = comprehension_error() do
            @slic (;x=[1.0, 2.0]) begin
                z = comprehension_nested(x)
                mu ~ normal(z[1], 1.0)
            end
        end
        @test nested !== nothing
        @test occursin("nested comprehensions are not supported", nested)

        # stepped ranges stay rejected via the same (non-bounded-`lo:hi`) gate.
        stepped = comprehension_error() do
            @slic (;x=[1.0, 2.0, 3.0, 4.0]) begin
                z = comprehension_stepped(x)
                mu ~ normal(z[1], 1.0)
            end
        end
        @test stepped !== nothing
        @test occursin("bounded `lo:hi` range", stepped)

        # chained/flattened `for i … for j …` (ragged result) rejects specifically.
        flattened = comprehension_error() do
            @slic (;x=[1.0, 2.0]) begin
                z = comprehension_flatten(x, 2)
                mu ~ normal(z[1], 1.0)
            end
        end
        @test flattened !== nothing
        @test occursin("chained/flattened generators", flattened)
    end

    @testset "multi-generator comprehensions lower to N-D results" begin
        # `[x[i] + j for i in 1:n, j in 1:m]` now lowers to nested fill loops over
        # an N-D result (was rejected as "multiple generators"); real → matrix.
        real_nd = @slic (;x=[1.0, 2.0, 3.0]) begin
            M = comprehension_multiple(x, 2)
            mu ~ normal(M[1, 1], 1.0)
        end
        nd_code = stan_code(real_nd)
        @test occursin(r"matrix\[[^]]+, ?[^]]+\] comprehension_result__lc_\d+;", nd_code)
        @test occursin(r"for\(i in 1:n\)", nd_code)
        @test occursin(r"comprehension_result__lc_\d+\[i, ?j\] =", nd_code)
        @test stanc_compiles(real_nd)
    end

    @testset "value-iteration generators lower to index loops" begin
        # `[xi for xi in x]` desugars to `[x[_vi] for _vi in 1:length(x)]`.
        vec_model = @slic (;x=[1.0, 2.0, 3.0], y=6.0) begin
            w ~ std_normal(; n=2)
            z = comprehension_iter_square(x)
            y ~ normal(sum(z) + w[1], 0.5)
        end
        vcode = stan_code(vec_model)
        @test occursin(r"for\(value_index__vi_\d+ in 1:", vcode)
        @test stanc_compiles(vec_model)
        vprob = instantiate(vec_model)
        vlp, vgrad = LogDensityProblems.logdensity_and_gradient(
            vprob, fill(0.1, LogDensityProblems.dimension(vprob)))
        @test isfinite(vlp)
        @test all(isfinite, vgrad)

        # value-iteration over the EachCol view usertype: `[sum(c) for c in EachCol(X)]`.
        col_model = @slic (;X=[1.0 2.0; 3.0 4.0; 5.0 6.0], y=6.0) begin
            w ~ std_normal(; n=3)
            cs = comprehension_iter_eachcol(X)
            y ~ normal(sum(cs) + w[1], 0.5)
        end
        @test occursin("col(X", stan_code(col_model))
        @test stanc_compiles(col_model)
        cprob = instantiate(col_model)
        clp, cgrad = LogDensityProblems.logdensity_and_gradient(
            cprob, fill(0.1, LogDensityProblems.dimension(cprob)))
        @test isfinite(clp)
        @test all(isfinite, cgrad)

        # value-iteration over a ragged-data RaggedVector: `[sum(gi) for gi in g]`.
        rag_model = @slic (;g=[[1.0, 2.0, 3.0], [4.0, 5.0]], y=6.0) begin
            w ~ std_normal(; n=2)
            gs = comprehension_iter_ragged(g)
            y ~ normal(sum(gs) + w[1], 0.5)
        end
        @test stanc_compiles(rag_model)
        rprob = instantiate(rag_model)
        rlp, rgrad = LogDensityProblems.logdensity_and_gradient(
            rprob, fill(0.1, LogDensityProblems.dimension(rprob)))
        @test isfinite(rlp)
        @test all(isfinite, rgrad)
    end

    @testset "value-iteration for-loop with `.=` accumulation" begin
        # `for xi in x; acc .= acc + xi` desugars to `for _vi in 1:length(x)` with
        # `xi = x[_vi]`; `.=` (not `=`) is required for the accumulation itself.
        model = @slic (;y=6.0) begin
            x ~ std_normal(; n=3)
            s = for_iter_sum(x)
            y ~ normal(s, 0.5)
        end
        code = stan_code(model)
        @test occursin(r"for\(value_index__vi_\d+ in 1:", code)
        @test stanc_compiles(model)
        prob = instantiate(model)
        lp, grad = LogDensityProblems.logdensity_and_gradient(
            prob, fill(0.1, LogDensityProblems.dimension(prob)))
        @test isfinite(lp)
        @test all(isfinite, grad)
    end

    @testset "model-level comprehension remains rejected" begin
        err = comprehension_error() do
            @slic (;x=[1.0, 2.0]) begin
                z = [x[i] for i in 1:2]
                mu ~ normal(z[1], 1.0)
            end
        end
        @test err !== nothing
        @test occursin("`comprehension` control flow is not supported in @slic model bodies", err)
    end
end

"""
Verify the enumerate/zip/N-D/nested iteration-protocol extensions (devibe
`2eb42c2`): tuple-destructuring sources and multi-axis comprehensions/loops in
`@deffun` bodies, gated on `stanc_compiles` with BridgeStan lp/gradient spot
checks, plus the 3-D and flattened-generator rejections.
"""
@testitem "slic: enumerate / zip / N-D comprehension iteration" tags=[:slic, :shapes] setup=[StanBlocksImports, StanBlocksTestSetup] begin
    iter_error(f) = try (f(); nothing) catch e sprint(showerror, e) end

    @testset "enumerate binds (index, element)" begin
        # comprehension: index is the 1-based position (data int), element is x[i].
        m = @slic (;x=[1.0, 2.0, 3.0], y=6.0) begin
            w ~ std_normal(; n=2)
            z = iter_enum(x)
            y ~ normal(sum(z) + w[1], 0.5)
        end
        code = stan_code(m)
        @test occursin("iter_enum", code)
        @test stanc_compiles(m)
        prob = instantiate(m)
        lp, g = LogDensityProblems.logdensity_and_gradient(prob, fill(0.1, LogDensityProblems.dimension(prob)))
        @test isfinite(lp) && all(isfinite, g)

        # enumerate over the EachCol view + a for-loop accumulation form.
        col = @slic (;X=[1.0 2.0; 3.0 4.0; 5.0 6.0], y=6.0) begin
            w ~ std_normal(; n=2)
            cs = iter_enum_col(X)
            acc = iter_enum_acc(cs)
            y ~ normal(acc + w[1], 0.5)
        end
        @test occursin("col(X", stan_code(col))
        @test stanc_compiles(col)
        cprob = instantiate(col)
        clp, cg = LogDensityProblems.logdensity_and_gradient(cprob, fill(0.1, LogDensityProblems.dimension(cprob)))
        @test isfinite(clp) && all(isfinite, cg)
    end

    @testset "zip iterates parallel containers" begin
        m = @slic (;a=[1.0, 2.0, 3.0], b=[4.0, 5.0, 6.0], y=6.0) begin
            w ~ std_normal(; n=2)
            z = iter_zip(a, b)
            y ~ normal(sum(z) + w[1], 0.5)
        end
        @test occursin("iter_zip", stan_code(m))
        @test stanc_compiles(m)
        prob = instantiate(m)
        lp, g = LogDensityProblems.logdensity_and_gradient(prob, fill(0.1, LogDensityProblems.dimension(prob)))
        @test isfinite(lp) && all(isfinite, g)

        m3 = @slic (;a=[1.0, 2.0], b=[3.0, 4.0], c=[5.0, 6.0], y=6.0) begin
            w ~ std_normal(; n=2)
            z = iter_zip3(a, b, c)
            y ~ normal(sum(z) + w[1], 0.5)
        end
        @test stanc_compiles(m3)
    end

    @testset "N-D comprehensions and one-line nested loops" begin
        # real → matrix[n,m], with nested fill loops.
        nd = @slic (;x=[1.0, 2.0, 3.0], y=[4.0, 5.0], obs=6.0) begin
            w ~ std_normal(; n=2)
            M = iter_nd(x, y)
            obs ~ normal(M[1, 1] + w[1], 0.5)
        end
        ndc = stan_code(nd)
        @test occursin(r"matrix\[[^]]+, ?[^]]+\] comprehension_result__lc_\d+;", ndc)
        @test occursin(r"for\(i in 1:n\)", ndc)
        @test occursin(r"for\(j in 1:m\)", ndc)
        @test stanc_compiles(nd)
        ndprob = instantiate(nd)
        ndlp, ndg = LogDensityProblems.logdensity_and_gradient(ndprob, fill(0.1, LogDensityProblems.dimension(ndprob)))
        @test isfinite(ndlp) && all(isfinite, ndg)

        # int → array[n,m] int.
        ndi = @slic (;x=[1, 2, 3], y=[4, 5]) begin
            M = iter_nd_int(x, y)
            mu ~ normal(M[1, 1], 1.0)
        end
        @test occursin(r"array\[[^]]+, ?[^]]+\] int comprehension_result__lc_\d+;", stan_code(ndi))
        @test stanc_compiles(ndi)

        # one-line nested for-loop (statement) accumulating a matrix sum.
        nested = @slic (;X=[1.0 2.0; 3.0 4.0], obs=6.0) begin
            w ~ std_normal(; n=2)
            s = iter_nested(X)
            obs ~ normal(s + w[1], 0.5)
        end
        @test stanc_compiles(nested)
        nprob = instantiate(nested)
        nlp, ng = LogDensityProblems.logdensity_and_gradient(nprob, fill(0.1, LogDensityProblems.dimension(nprob)))
        @test isfinite(nlp) && all(isfinite, ng)
    end

    @testset "unsupported iteration shapes reject loudly" begin
        # 3-D comprehension (SLIC containers are ≤ 2-D).
        d3 = iter_error() do
            stan_code(@slic (;x=[1, 2]) begin
                M = iter_nd3(x); mu ~ normal(M[1, 1, 1], 1.0)
            end)
        end
        @test d3 !== nothing
        @test occursin("dimensional comprehensions are unsupported", d3)

        # chained/flattened `for i … for j …` (ragged result — deferred follow-up).
        flat = iter_error() do
            stan_code(@slic (;x=[1.0, 2.0]) begin
                z = comprehension_flatten(x, 2); mu ~ normal(z[1], 1.0)
            end)
        end
        @test flat !== nothing
        @test occursin("chained/flattened generators", flat)
    end
end

"""
Verify `return_type_of(f, args...)` exposes the existing SLIC inference table
both as a direct public query and as a computed `@deffun` type annotation.
"""
@testitem "slic: public return_type_of transpile-time query" tags=[:slic, :shapes] setup=[StanBlocksImports, StanBlocksTestSetup] begin
    scalar_rt = return_type_of(return_type_scalar, 1.0)
    vector_rt = return_type_of(return_type_vector, [1.0, 2.0, 3.0])

    @test scalar_rt isa StanBlocks.StanType
    @test sprint(show, scalar_rt) == "real"
    @test sprint(show, vector_rt) == "vector[3]"

    model = @slic (;x=[1.0, 2.0, 3.0]) begin
        y = copy_via_return_type(x)
        mu ~ std_normal()
        mu
    end
    @test check_type(model, :y, "vector")
    @test stanc_compiles(model)

    err = try
        return_type_of(ordinary_return_type_probe, 1.0)
        nothing
    catch e
        e
    end
    @test err isa ArgumentError
    @test occursin("Only non-inline @deffun functions and @defsig-registered SLIC callables are supported", sprint(showerror, err))
end

"""
Verify `slic: computed type annotations (@deffun container inference)` in an isolated test item.
"""
@testitem "slic: computed type annotations (@deffun container inference)" tags=[:slic, :shapes] setup=[StanBlocksImports, StanBlocksTestSetup] begin
    # `@deffun` accepts a COMPUTED type annotation — `typeof(f(x[1]))[dims]` — in
    # both a body decl (`umap`) and the `::ret` position (`umap_r`). The output
    # container is INFERRED from `f`'s per-element return type: a `real` element →
    # native `vector[n]`, an `int` element → `array[] int` (stays int, usable as an
    # index), `real` + 2 dims → `matrix[n,n]`. So ONE definition covers both int-
    # and real-valued element functions, subsuming the separate `map`/`imap` (j*/i*)
    # variants (todo `1v94weu`, commits `03aef9c` body decls + `a11934a` `::ret`).
    # GATE ON `compiles`, NOT `transpiles` — an @deffun body can transpile yet be
    # rejected by stanc (StanBlocks primer §5 of stanblocks-use; the umap fixtures
    # are defined at module level above).
    @testset "body decl: real element → vector[n]" begin
        model = @slic (;v=[1.0,2.0,3.0]) begin
            w = umap(a -> a*2.0, v)
            mu ~ std_normal()
            mu
        end
        @test check_type(model, :w, "vector")
        @test stanc_compiles(model)
    end
    @testset "body decl: int element → array[] int (stays int, index-usable)" begin
        model = @slic (;a=[1,2,3]) begin
            b = umap(x -> x + 1, a)
            mu ~ std_normal()
            mu
        end
        code = stan_code(model)
        # int container preserved — NOT silently coerced to a real `vector`
        @test occursin(r"array\[\w*\] int (?:\w+ )?b\b", code)
        @test !occursin(r"vector\[\w*\] (?:\w+ )?b\b", code)
        @test stanc_compiles(model)
    end
    @testset "::ret form: real → vector[n], int → array[] int" begin
        mr = @slic (;v=[1.0,2.0,3.0]) begin
            w = umap_r(a -> a*2.0, v)
            mu ~ std_normal()
            mu
        end
        mi = @slic (;a=[1,2,3]) begin
            b = umap_r(x -> x + 1, a)
            mu ~ std_normal()
            mu
        end
        @test check_type(mr, :w, "vector")
        @test occursin(r"array\[\w*\] int (?:\w+ )?b\b", stan_code(mi))
        @test stanc_compiles(mr)
        @test stanc_compiles(mi)
    end
    @testset "2 dims: real element → matrix[n,n]" begin
        model = @slic (;v=[1.0,2.0,3.0]) begin
            w = umap_mat(a -> a*2.0, v)
            mu ~ std_normal()
            mu
        end
        @test check_type(model, :w, "matrix")
        @test stanc_compiles(model)
    end
    @testset "reject: annotation base must be a type token, not a value" begin
        # `rv::(x[1])[n]` — `x[1]` is a real VALUE, not a `typeof(...)` token — so
        # `_decl_computed_type` errors loudly at trace time instead of miscompiling.
        @test_throws Exception stan_code(@slic (;v=[1.0,2.0,3.0]) begin
            w = bad_computed_ann(v)
            mu ~ std_normal()
            mu
        end)
    end
end

"""
Verify `slic: typed assignment compatibility` in an isolated test item.
"""
@testitem "slic: typed assignment compatibility" tags=[:slic, :shapes] setup=[StanBlocksImports, StanBlocksTestSetup] begin
    good = @slic (;x=[1.0, 2.0, 3.0], ii=[1, 2, 3]) begin
        # Known-value equality: the literal 3 matches the data-size symbol x_n.
        direct::vector[3] = x
        symbolic = typed_assign_symbolic(x)
        computed_real = typed_assign_computed(x)
        computed_int = typed_assign_computed(ii)
        filled = bare_parameter_local_copy(x)
        widened::real = ii[1]
        mu ~ normal(sum(direct + symbolic + computed_real + filled), 1.0)
        idx = computed_int[1]
        result = mu + idx + widened
    end
    @test transpiles(good)
    @test stanc_compiles(good)
    code = stan_code(good)
    @test occursin(r"array\[\w*\] int (?:\w+ )?computed_int\b", code)

    center_bad = @slic (;x=[1.0, 2.0, 3.0]) begin
        rv = typed_assign_bad_center(x)
        mu ~ std_normal()
        mu
    end
    center_err = try
        stan_code(center_bad)
        nothing
    catch e
        sprint(showerror, e)
    end
    @test center_err !== nothing
    @test occursin("Typed assignment", center_err)
    @test occursin("RHS center", center_err)

    dim_bad = @slic (;x=[1.0, 2.0, 3.0]) begin
        rv = typed_assign_bad_dim(x)
        mu ~ std_normal()
        mu
    end
    dim_err = try
        stan_code(dim_bad)
        nothing
    catch e
        sprint(showerror, e)
    end
    @test dim_err !== nothing
    @test occursin("Typed assignment", dim_err)
    @test occursin("dimension 1", dim_err)
end

@testitem "slic: bare whole-vector local (broadcast RHS) emits unquoted size" tags=[:slic, :shapes] setup=[StanBlocksImports, StanBlocksTestSetup] begin
    # Regression for the inferred whole-vector-local snag: a bare (un-annotated)
    # intermediate local whose RHS is a broadcast must emit a VALID unquoted size
    # `vector[dims(a)[1]]`, not the invalid quoted `vector["dims(a)[1]"]` that
    # transpiles() accepts but stanc rejects.
    single = @slic (;yy=randn(4)) begin
        a ~ std_normal(;n=4)
        out = bare_vec_local_single(a)
        yy ~ normal(out, 1.0)
    end
    @test transpiles(single)
    @test stanc_compiles(single)
    single_code = stan_code(single)
    @test occursin("vector[dims(a)[1]]", single_code)
    @test !occursin("\"dims(", single_code)

    # Repeated dim, size sourced from a non-first arg (`c`): even here the bare
    # form works because it declares with the RHS type directly (no assertion).
    multi = @slic (;a=randn(4), b=randn(4), yy=randn(4)) begin
        c ~ std_normal(;n=4)
        out = bare_vec_local_multi(a, b, c)
        yy ~ normal(out, 1.0)
    end
    @test transpiles(multi)
    @test stanc_compiles(multi)
    @test !occursin("\"dims(", stan_code(multi))
end

"""
Verify `slic: jmap (inference-driven element-wise map)` in an isolated test item.
"""
@testitem "slic: jmap (inference-driven element-wise map)" tags=[:slic, :shapes] setup=[StanBlocksImports, StanBlocksTestSetup] begin
    # `jmap(f, x::anything[n])` maps `f` over `x`, inferring the output CONTAINER
    # from `f`'s per-element return type (`typeof(f(x[1]))`): a `real`-returning
    # `f` → `vector[n]`, an `int`-returning `f` → `array[] int`. ONE definition
    # covers both element kinds — no separate real `map` / int `imap` (prong 3,
    # commit `7279601`, decision `1mk4gqh`; enabler `03aef9c`/`a11934a`). GATE ON
    # `compiles`, NOT `transpiles` — a `@deffun` body can transpile yet stanc-REJECT.
    @testset "real f over a vector → vector[n]" begin
        model = @slic (;w=[1.0,2.0,3.0]) begin   # Vector{Float64} → Stan vector
            z = jmap(a -> a * 2.0, w)
            mu ~ std_normal()
            mu
        end
        @test check_type(model, :z, "vector")
        @test stanc_compiles(model)
    end
    @testset "int f over an int array → array[] int (container preserved)" begin
        model = @slic (;k=[3,4,5]) begin        # Vector{Int} → array[] int
            r = jmap(a -> a + 1, k)
            mu ~ std_normal()
            mu
        end
        code = stan_code(model)
        # int container preserved — NOT silently coerced to a real `vector`
        @test occursin(r"array\[\w*\] int (?:\w+ )?r\b", code)
        @test !occursin(r"vector\[\w*\] (?:\w+ )?r\b", code)
        @test stanc_compiles(model)
    end
    @testset "real f over an int array → vector[n] (element type from f, not x)" begin
        # The OUTPUT container follows `f`'s return type, not the input element
        # type: a real-returning `f` over an int array still yields a `vector`.
        model = @slic (;k=[3,4,5]) begin
            r = jmap(a -> a * 1.5, k)
            mu ~ std_normal()
            mu
        end
        @test check_type(model, :r, "vector")
        @test stanc_compiles(model)
    end
    @testset "reuse: two identical closure literals don't collide" begin
        # Regression for a closure-lift dedup/naming mismatch (fixed `harden.43kmku8o`,
        # functions.jl `sig_expr(::StanExpr2{<:types.closure})`): two DISTINCT
        # `a->a*2.0` literals get distinct closure ids → distinct Stan fn names
        # `jmap_closure_1`/`_2`, but the function-dedup key had dropped the id and
        # collapsed both to ONE definition, leaving the 2nd call site referencing
        # an undeclared `jmap_closure_2` (stanc reject). Binding the closure to a
        # name (one `:->` site → one id) was the only working form before the fix.
        model = @slic (;w=[1.0,2.0,3.0], v=[4.0,5.0,6.0]) begin
            z1 = jmap(a -> a * 2.0, w)
            z2 = jmap(a -> a * 2.0, v)
            mu ~ std_normal()
            mu
        end
        @test stanc_compiles(model)
    end
end

"""
Verify `shapes: RNG` in an isolated test item.
"""
@testitem "shapes: RNG" tags=[:slic, :shapes, :stanc] setup=[StanBlocksImports, StanBlocksTestSetup] begin
    @testset "vector_std_normal_rng(n) :: vector[n]" begin
        @test stanc_compiles(@slic (;n=5, obs=randn(5)) begin
            mu     ~ std_normal(;n=n)
            obs    ~ normal(mu, 1.)
            y_rep  = vector_std_normal_rng(n)
        end)
    end
end

"""
Verify `shapes: distributions` in an isolated test item.
"""
@testitem "shapes: distributions" tags=[:slic, :shapes, :stanc] setup=[StanBlocksImports, StanBlocksTestSetup] begin
    @testset "normal" begin
        @test stanc_compiles(@slic (;obs=1.5) begin
            mu    ~ std_normal()
            sigma ~ std_normal(;lower=0.)
            obs   ~ normal(mu, sigma)
        end)
        @test stanc_compiles(@slic (;obs=randn(10)) begin
            mu    ~ std_normal()
            sigma ~ std_normal(;lower=0.)
            obs   ~ normal(mu, sigma)
        end)
    end
    @testset "student_t" begin
        @test stanc_compiles(@slic (;obs=0.) begin
            mu  ~ std_normal()
            obs ~ student_t(3., mu, 1.)
        end)
    end
    @testset "cauchy" begin
        @test stanc_compiles(@slic (;obs=0.) begin
            mu  ~ std_normal()
            obs ~ cauchy(mu, 1.)
        end)
    end
    @testset "lognormal" begin
        @test stanc_compiles(@slic (;obs=1.) begin
            mu    ~ std_normal()
            sigma ~ std_normal(;lower=0.)
            obs   ~ lognormal(mu, sigma)
        end)
    end
    @testset "gamma" begin
        @test stanc_compiles(@slic (;obs=1.) begin
            alpha ~ std_normal(;lower=0.)
            beta  ~ std_normal(;lower=0.)
            obs   ~ gamma(alpha, beta)
        end)
    end
    @testset "beta" begin
        @test stanc_compiles(@slic (;obs=0.5) begin
            alpha  ~ std_normal(;lower=0.)
            beta_p ~ std_normal(;lower=0.)
            obs    ~ beta(alpha, beta_p)
        end)
    end
    @testset "exponential" begin
        @test stanc_compiles(@slic (;obs=1.) begin
            lambda ~ std_normal(;lower=0.)
            obs    ~ exponential(lambda)
        end)
    end
    @testset "uniform" begin
        @test stanc_compiles(@slic (;obs=0.5) begin
            obs ~ uniform(0., 1.)
        end)
    end
    @testset "bernoulli / bernoulli_logit" begin
        @test stanc_compiles(@slic (;obs=1) begin
            theta ~ beta(1., 1.)
            obs   ~ bernoulli(theta)
        end)
        @test stanc_compiles(@slic (;obs=1) begin
            logit_theta ~ std_normal()
            obs         ~ bernoulli_logit(logit_theta)
        end)
    end
    @testset "binomial" begin
        @test stanc_compiles(@slic (;k=3, n=10) begin
            theta ~ beta(1., 1.)
            k     ~ binomial(n, theta)
        end)
    end
    @testset "neg_binomial_2" begin
        @test stanc_compiles(@slic (;obs=5) begin
            mu  ~ std_normal(;lower=0.)
            phi ~ std_normal(;lower=0.)
            obs ~ neg_binomial_2(mu, phi)
        end)
    end
    @testset "multi_normal" begin
        @test stanc_compiles(@slic (;obs=randn(3)) begin
            mu  ~ std_normal(;n=3)
            cov = diag_matrix(rep_vector(1., 3))
            obs ~ multi_normal(mu, cov)
        end)
    end
end

"""
Verify `regression: @deffun symbolic-size operand propagation through HOFs` in an isolated test item.
"""
@testitem "regression: @deffun symbolic-size operand propagation through HOFs" tags=[:slic, :regression] setup=[StanBlocksImports, StanBlocksTestSetup] begin
    D = (; us=[1, 2], ut=[[0., 1.], [0., 1., 2.]], dt=[[0.], [0., 5.]])
    # double-HOF + symbolic-size core: was `tracetype not defined for (anything - anything)`
    m_hof = @slic D begin
        r = hd_sublens(hd_wrap_sym, us, ut, dt)
        p ~ normal(1.0 * sum(r), 1.0)
    end
    @test transpiles(m_hof)
    # direct (0-HOF) symbolic-size call: was an infinite-loop hang
    m_direct = @slic D begin
        idx = hd_wrap_sym(1, ut, dt)
        p ~ normal(1.0 * sum(idx), 1.0)
    end
    @test transpiles(m_direct)
    # control: concrete-size core stays clean (no regression)
    m_concrete = @slic D begin
        r = hd_sublens(hd_wrap_concrete, us, ut, dt)
        p ~ normal(1.0 * sum(r), 1.0)
    end
    @test transpiles(m_concrete)
end

"""
Verify `slic: compiler-injected fresh-result slice-fills (case-3)` in an isolated test item.
"""
@testitem "slic: compiler-injected fresh-result slice-fills (case-3)" tags=[:slic] setup=[StanBlocksImports, StanBlocksTestSetup] begin
    # PARAM input: fresh result filled from a parameter → transformed parameters.
    param_model = @slic (;y=randn(3)) begin
        x ~ std_normal(;n=3)
        r = c3_freshfill(x)
        y ~ normal(r, 1.0)
    end
    @test transpiles(param_model)
    @test stanc_compiles(param_model)
    @test occursin(r"transformed parameters\s*\{[^}]*\bout__il_\d+", stan_code(param_model))

    # DATA input: fresh result filled from data → transformed data.
    data_model = @slic (;xd=randn(3)) begin
        r = c3_freshfill(xd)
        p ~ std_normal()
    end
    @test transpiles(data_model)
    @test stanc_compiles(data_model)
    @test occursin(r"transformed data\s*\{[^}]*\bout__il_\d+", stan_code(data_model))

    # NESTED INLINE: the fresh declaration is revealed only during the nested
    # re-trace, so sibling-statement lookahead cannot classify it. Its first
    # certified data fill must reset the provisional parameter qual to data.
    nested_data_model = @slic (;xd=randn(3)) begin
        r = c3_nestedfill(xd)
        p ~ std_normal()
    end
    @test transpiles(nested_data_model)
    @test stanc_compiles(nested_data_model)
    @test occursin(r"transformed data\s*\{[^}]*\bout__il_\d+", stan_code(nested_data_model))

    # QUANTITIES input: a cv-tainted parameter is predictive-only from the
    # forward pass onward, so its first fill reclassifies the fresh declaration
    # directly into generated quantities (not parameters/TP).
    cv_n = StanBlocks.stan.maybecv(:n, 3)
    quantities_model = @slic (;n=cv_n) begin
        x ~ std_normal(;n=n)
        r = c3_freshfill(x)
    end
    @test transpiles(quantities_model)
    @test stanc_compiles(quantities_model)
    quantities_code = stan_code(quantities_model)
    @test occursin(r"generated quantities\s*\{[^}]*\bout__il_\d+", quantities_code)
    @test !occursin(r"parameters\s*\{[^}]*\bout__il_\d+", quantities_code)

    # regression: inline UDF with a for-loop fill that USES the size param `n`
    # (layer-0 inline size resolution) still transpiles AND compiles.
    build_eta_model = @slic (;y=randn(20), X=randn(20,3)) begin
        alpha ~ std_normal()
        beta  ~ std_normal(;n=3)
        sigma ~ std_normal(;lower=0.)
        mu    = c3_build_eta(alpha, beta, X)
        y     ~ normal(mu, sigma)
    end
    @test transpiles(build_eta_model)
    @test stanc_compiles(build_eta_model)

    # gate: inlining a mutating helper ONTO a sampled parameter must still be rejected
    # (Stan parameters are read-only in the model block).
    reject_model = @slic (;y=randn(4)) begin
        theta ~ std_normal(;n=4)
        s = c3_set_first!(theta)
        y ~ normal(s, 1.0)
    end
    @test !transpiles(reject_model; re=false)
end

"""
Verify `slic: ragged lowering adopts a bare free parameter` in an isolated test item.
"""
@testitem "slic: ragged lowering adopts a bare free parameter" tags=[:slic, :ragged, :bridgestan] setup=[StanBlocksImports, StanBlocksTestSetup] begin
    ragged_model = @slic (;Ks=[2, 3], y=0.2) begin
        p::simplex[Ks] ~ flat()
        y ~ normal(sum(p[1]), 1.0)
    end
    @test transpiles(ragged_model)
    @test stanc_compiles(ragged_model)
    code = stan_code(ragged_model)
    @test LogDensityProblems.dimension(instantiate(stan_model(ragged_model))) == 3

    let parameters = stan_block(code, "parameters")
        @test occursin(r"\bvector\[[^]]+\] p_free__rc_\d+;", parameters)
        @test !occursin("p_mem__rc_", parameters)
    end
    let tp = stan_block(code, "transformed parameters")
        @test occursin(r"\bvector\[[^]]+\] p_mem__rc_\d+;", tp)
        @test occursin(r"p_mem__rc_\d+\[", tp)
    end
    let model_block = stan_block(code, "model")
        @test !occursin(r"p_free__rc_\d+\s*~", model_block)
        @test occursin("y ~ normal", model_block)
    end
end

"""
Verify `slic: cert-marker read-only gate (case-3)` in an isolated test item.
"""
@testitem "slic: cert-marker read-only gate (case-3)" tags=[:slic] setup=[StanBlocksImports, StanBlocksTestSetup] begin
    # SAMPLED parameter (untyped `~`): read-only ⇒ the parameter message.
    m_param = @slic (;y=randn(4)) begin
        theta ~ std_normal(;n=4)
        s = c3_set_first!(theta)
        y ~ normal(s, 1.0)
    end
    e_param = _cert_reject_msg(m_param)
    @test e_param !== nothing
    @test occursin("parameter `theta`", e_param) && occursin("read-only", e_param)

    # SAMPLED parameter with a TYPED LHS (`theta::vector[n] ~ ...`): same reject. The
    # typed-LHS trace path builds the type differently — it must NOT leak `fresh_decl`
    # onto a sampled parameter (which would wrongly *accept* the fill).
    m_typed = @slic (;y=randn(4)) begin
        theta::vector[4] ~ std_normal()
        s = c3_set_first!(theta)
        y ~ normal(s, 1.0)
    end
    @test !transpiles(m_typed; re=false)

    # DATA input: Stan data is read-only; filling it via an inlined mutating helper
    # previously emitted invalid `xd[i] = ...` — now a clear reject (the DATA case,
    # NEW in a6bf5a3). `xd` (a signature input) has no `fresh_decl`, so the gate fires.
    m_data = @slic (;xd=randn(4)) begin
        s = c3_set_first!(xd)
        p ~ normal(s[1], 1.0)
    end
    e_data = _cert_reject_msg(m_data)
    @test e_data !== nothing
    @test occursin("`xd`", e_data) && occursin("not a fresh declaration", e_data) && occursin("data", e_data)

    # PRE-COMMITTED / derived local (`w = 2 .* theta`): its qual is committed at its
    # assignment — one-pass tracing can't un-commit it, so a later element-fill rejects
    # (the settled SSA bar, NEW in a6bf5a3). Derived-from-parameter ⇒ `:parameter` qual,
    # so the base is not `fresh_decl` and the gate fires.
    m_derived = @slic (;y=randn(4)) begin
        theta ~ std_normal(;n=4)
        w = 2.0 .* theta
        s = c3_set_first!(w)
        y ~ normal(s, 1.0)
    end
    @test !transpiles(m_derived; re=false)
end

"""
Verify `slic: compiler-injected for-loop + range fresh-result fills (case-3 C.1/C.2)` in an isolated test item.
"""
@testitem "slic: compiler-injected for-loop + range fresh-result fills (case-3 C.1/C.2)" tags=[:slic] setup=[StanBlocksImports, StanBlocksTestSetup] begin
    # C.1 PARAM: for-loop fresh fill from a parameter → loop+decl+bind in transformed parameters.
    forfill_param = @slic (;y=randn(3)) begin
        x ~ std_normal(;n=3)
        r = c3_forfill(x)
        y ~ normal(r, 1.0)
    end
    @test transpiles(forfill_param)
    @test stanc_compiles(forfill_param)
    let tp = stan_block(stan_code(forfill_param), "transformed parameters")
        @test occursin("for(", tp)             # loop emitted symbolically (not unrolled)
        @test occursin(r"\bout__il_\d+", tp)   # fresh-result decl co-located in the same block
    end
    # qual-routing guard (29c3b59): the loop must NOT land in generated quantities.
    @test !occursin("for(", stan_block(stan_code(forfill_param), "generated quantities"))

    # C.1 DATA: for-loop fresh fill from data → transformed data.
    forfill_data = @slic (;xd=randn(3)) begin
        r = c3_forfill(xd)
        p ~ std_normal()
    end
    @test transpiles(forfill_data)
    @test stanc_compiles(forfill_data)
    let td = stan_block(stan_code(forfill_data), "transformed data")
        @test occursin("for(", td)
        @test occursin(r"\bout__il_\d+", td)
    end

    # C.2 PARAM: range-indexed fresh fill from a parameter → transformed parameters,
    # base-coarse-grained (out[1:2] routes by base `out`, like a scalar fill).
    rangefill_param = @slic (;y=randn(3)) begin
        x ~ std_normal(;n=3)
        r = c3_rangefill(x)
        y ~ normal(r[1], 1.0)
    end
    @test transpiles(rangefill_param)
    @test stanc_compiles(rangefill_param)
    @test occursin(r"out__il_\d+\[1:2\]\s*=", stan_block(stan_code(rangefill_param), "transformed parameters"))

    # C.2 DATA: range-indexed fresh fill from data → transformed data.
    rangefill_data = @slic (;xd=randn(3)) begin
        r = c3_rangefill(xd)
        p ~ std_normal()
    end
    @test transpiles(rangefill_data)
    @test stanc_compiles(rangefill_data)
    @test occursin(r"out__il_\d+\[1:2\]\s*=", stan_block(stan_code(rangefill_data), "transformed data"))
end

"""
Verify `slic: mixed sampling/fill routing in compiler-owned plate loop` in an isolated test item.
"""
@testitem "slic: mixed sampling/fill routing in compiler-owned plate loop" tags=[:slic, :plate] setup=[StanBlocksImports, StanBlocksTestSetup] begin
    @test transpiles(c3_plate_router_model)
    @test stanc_compiles(c3_plate_router_model)
    code = stan_code(c3_plate_router_model)

    let parameters = stan_block(code, "parameters")
        @test occursin("vector[n] routed_plate_x", parameters)
        @test occursin("vector[n] routed_plate_y", parameters)
        @test occursin("real routed_plate_flat", parameters)
        @test !occursin("routed_plate_rv", parameters)
    end
    let td = stan_block(code, "transformed data")
        @test occursin("for(routed_plate_i in 1:n)", td)
        @test occursin("routed_plate_data_copy[routed_plate_i] = obs[routed_plate_i]", td)
    end
    let tp = stan_block(code, "transformed parameters")
        @test occursin("for(routed_plate_i in 1:n)", tp)
        @test occursin("routed_plate_rv[routed_plate_i] =", tp)
    end
    let model_block = stan_block(code, "model")
        @test occursin("for(routed_plate_i in 1:n)", model_block)
        @test occursin("routed_plate_x[routed_plate_i] ~ normal", model_block)
        @test occursin("routed_plate_y[routed_plate_i] ~ normal", model_block)
        @test occursin("obs[routed_plate_i] ~ normal", model_block)
        @test !occursin("routed_plate_rv[routed_plate_i] =", model_block)
    end
    let gq = stan_block(code, "generated quantities")
        @test occursin("vector[n] routed_plate_prior", gq)
        @test occursin("for(routed_plate_i in 1:n)", gq)
        @test occursin("routed_plate_prior[routed_plate_i] = normal_rng", gq)
    end
end

"""
Verify `slic: public plate() do-block emitter` in an isolated test item.
"""
@testitem "slic: public plate() do-block emitter" tags=[:slic, :plate, :bridgestan] setup=[StanBlocksImports, StanBlocksTestSetup] begin
    # 1. Scalar-per-cell plate with a live per-cell likelihood → vector result.
    doblock = @slic (; y = randn(6), mu0 = 0.5) begin
        sigma ~ normal(0.0, 1.0; lower = 0.0)
        theta ~ plate(y; outer = (6,)) do yi
            t ~ normal(mu0, 1.0)              # fresh per-cell param (mu0 captured)
            yi ~ normal(t, sigma)             # observation on the sliced input
            t                                 # cell output → theta[i]
        end
    end
    @test transpiles(doblock)
    @test stanc_compiles(doblock)
    let code = stan_code(doblock)
        params = stan_block(code, "parameters")
        @test occursin("vector[6] theta_t", params)    # fresh cell-local, namespaced under `theta`
        @test occursin(r"real<lower=0\.0> sigma", params)
        @test !occursin("theta;", params)              # theta is a TP fill, not a param (theta_t is the cell)
        tp = stan_block(code, "transformed parameters")
        @test occursin("vector[6] theta", tp)
        @test occursin(r"theta\[plate_i\w*\] = theta_t\[plate_i", tp)
        mb = stan_block(code, "model")
        @test occursin(r"theta_t\[plate_i\w*\] ~ normal\(mu0", mb)
        @test occursin(r"y\[plate_i\w*\] ~ normal\(theta_t\[plate_i", mb)
    end
    # sampler dims = sigma + t[6]; theta is a deterministic TP fill, not a dim.
    @test LogDensityProblems.dimension(instantiate(doblock)) == 7

    # 2. Vector-per-cell plate (BRM #1): typed fresh `vector[K]` param + typed
    #    vector cell output, each collected as `matrix[K, N]` (cells in columns).
    vec_cell = @slic (; n_series = 8) begin
        L::cholesky_factor_corr[6] ~ lkj_corr_cholesky(2.0)   # shared, captured
        tau::vector[6] ~ normal(0.0, 1.0; lower = 0.0)        # shared, captured
        b::vector[6] ~ plate(; outer = (n_series,)) do s
            z::vector[6] ~ std_normal()                       # fresh per-cell vector
            diag_pre_multiply(tau, L) * z                     # vector[6] cell output
        end
    end
    @test transpiles(vec_cell)
    @test stanc_compiles(vec_cell)
    let code = stan_code(vec_cell)
        params = stan_block(code, "parameters")
        @test occursin("matrix[6, n_series] b_z", params)     # per-cell vector → matrix column
        tp = stan_block(code, "transformed parameters")
        @test occursin("matrix[6, n_series] b", tp)
        @test occursin(r"b\[:, plate_i\w*\] = \(diag_pre_multiply\(tau, L\) \* b_z\[:, plate_i", tp)
        mb = stan_block(code, "model")
        @test occursin(r"b_z\[:, plate_i\w*\] ~ std_normal\(\)", mb)  # sampled per column
        @test !occursin("b[:", mb)                            # the b fill is NOT in model
    end

    # 3. fspmjv regression: a FULLY prior-only dead plate (no likelihood anywhere
    #    AND its output unused) must keep its fresh per-cell samples as PARAMETERS
    #    (not prior-only-lowered to generated quantities), so the transformed-
    #    parameters return-fill stays in scope. Before aef6a42 this emitted
    #    invalid Stan — stanc: "Identifier w not in scope". `stanc_compiles`
    #    is the guard that actually catches that regression.
    priordead = @slic (;) begin
        tau ~ normal(0.0, 1.0; lower = 0.0)
        z ~ plate(; outer = (4,)) do i
            w ~ normal(0.0, tau)              # fresh per-cell param — MUST stay a parameter
            w                                 # cell output → z[i] (transformed parameters)
        end
    end
    @test transpiles(priordead)
    @test stanc_compiles(priordead)
    let code = stan_code(priordead)
        params = stan_block(code, "parameters")
        @test occursin("vector[4] z_w", params)             # w stays a PARAMETER (the fix), namespaced under `z`
        @test occursin(r"real<lower=0\.0> tau", params)
        tp = stan_block(code, "transformed parameters")
        @test occursin("vector[4] z", tp)
        @test occursin(r"z\[plate_i\w*\] = z_w\[plate_i", tp)
        # The bug: w was RNG-lowered to generated quantities while z (TP) still
        # referenced it. Guard both the decl and the RNG draw are absent from GQ.
        gq = stan_block(code, "generated quantities")
        @test !occursin("w", gq)
    end
    # dim = tau + w[4] = 5 (w is a real sampler dim; z is a deterministic TP fill).
    @test LogDensityProblems.dimension(instantiate(priordead)) == 5

    # 4. Control for fspmjv: the moment the plate output feeds a likelihood, the
    #    fresh sample is naturally a parameter and TP is valid — confirms the
    #    fspmjv fix did not disturb the common (non-dead) path.
    priorlive = @slic (; obs = randn(4)) begin
        tau ~ normal(0.0, 1.0; lower = 0.0)
        z ~ plate(; outer = (4,)) do i
            w ~ normal(0.0, tau)
            w
        end
        obs ~ normal(z, 1.0)
    end
    @test transpiles(priorlive)
    @test stanc_compiles(priorlive)
    let code = stan_code(priorlive)
        @test occursin("vector[4] z_w", stan_block(code, "parameters"))
        @test occursin("vector[4] z", stan_block(code, "transformed parameters"))
        @test occursin(r"obs ~ normal\(z", stan_block(code, "model"))
    end
    @test LogDensityProblems.dimension(instantiate(priorlive)) == 5
end

"""
Verify `slic: multiple plates reuse cell-local names hygienically` in an isolated
test item. Two sequential independent plates that each use the SAME fresh cell-local
name (`z ~ std_normal()`) must scope hygienically into two INDEPENDENT parameter
carriers, not collide. Before the multiple-plate-l snag fix the second plate's
discovery saw the first plate's leaked `z` and rejected `z ~ …` as "LHS bound to a
parameter-qualified value". Regression for the BRM crossed-effects use case.
"""
@testitem "slic: multiple plates reuse cell-local names hygienically" tags=[:slic, :plate, :bridgestan] setup=[StanBlocksImports, StanBlocksTestSetup] begin
    # BRM crossed-effects shape: two grouping factors, each its own plate, both
    # using the natural fresh cell-local name `z`, then a JOINT likelihood.
    crossed = @slic (; y = [0.2, -0.1, 0.3, 0.0], subject = [1, 1, 2, 2], item = [1, 2, 1, 2]) begin
        b_subject ~ plate(; outer = (2,)) do s
            z ~ std_normal()
            z
        end
        b_item ~ plate(; outer = (2,)) do i
            z ~ std_normal()
            z
        end
        y ~ normal(b_subject[subject] + b_item[item], 1.0)
    end
    @test transpiles(crossed)
    @test stanc_compiles(crossed)
    let params = stan_block(stan_code(crossed), "parameters")
        # Two INDEPENDENT carriers, each namespaced under its plate's result name.
        @test occursin(r"vector\[2\] b_subject_z;", params)
        @test occursin(r"vector\[2\] b_item_z;", params)
    end
    let mb = stan_block(stan_code(crossed), "model")
        @test occursin(r"b_subject_z\[plate_i\w*\] ~ std_normal", mb)
        @test occursin(r"b_item_z\[plate_i\w*\] ~ std_normal", mb)
        @test occursin(r"y ~ normal\(\(b_subject\[subject\] \+ b_item\[item\]\)", mb)
    end
    # Independence at the sampler: 2 + 2 = 4 scalar dims (the two vector[2]
    # carriers), b_subject/b_item are deterministic transformed-parameter fills.
    @test LogDensityProblems.dimension(instantiate(crossed)) == 4

    # Same hygiene must reach a CALLED submodel's internal fresh name: two plates
    # bound to the same direct name `cell`, each calling the same submodel whose
    # internal fresh name is `z`, must not collide.
    @slic mp_ncp(k::int) = begin
        z::vector[k] ~ std_normal()
        return z
    end
    crossed_sub = @slic (; a = 1.0) begin
        g1 ~ plate(; outer = (2,)) do s
            cell ~ mp_ncp(3)
            cell
        end
        g2 ~ plate(; outer = (2,)) do s
            cell ~ mp_ncp(3)
            cell
        end
        a ~ normal(sum(g1[1]) + sum(g2[1]), 1.0)
    end
    @test transpiles(crossed_sub)
    @test stanc_compiles(crossed_sub)
    let params = stan_block(stan_code(crossed_sub), "parameters")
        @test occursin(r"matrix\[3, 2\] g1_cell_z;", params)
        @test occursin(r"matrix\[3, 2\] g2_cell_z;", params)
    end

    # Ragged cell-locals reusing the same name across plates stay independent too.
    crossed_ragged = @slic (; K = [2, 3, 4], y = 0.3) begin
        r1 ~ plate(; outer = length(K)) do g
            z::vector[K[g]] ~ std_normal()
            z
        end
        r2 ~ plate(; outer = length(K)) do g
            z::vector[K[g]] ~ std_normal()
            z
        end
        y ~ normal(sum(r1[1]) + sum(r2[2]), 1.0)
    end
    @test transpiles(crossed_ragged)
    @test stanc_compiles(crossed_ragged)
end

"""
Verify `slic: public plate promotes called-submodel bindings` in an isolated test item.
"""
@testitem "slic: public plate promotes called-submodel bindings" tags=[:slic, :plate] setup=[StanBlocksImports, StanBlocksTestSetup] begin
    @test transpiles(c3_plate_submodel_model)
    @test stanc_compiles(c3_plate_submodel_model)
    code = stan_code(c3_plate_submodel_model)

    let parameters = stan_block(code, "parameters")
        # Submodel-internal `z` is namespaced under the plate result `theta` via
        # the direct binding `cell` (`theta` + `cell` + `z`).
        @test occursin("vector[6] theta_cell_z", parameters)
        @test occursin("real<lower=0.0> sigma", parameters)
    end
    let tp = stan_block(code, "transformed parameters")
        @test occursin(r"theta_cell_shifted\[plate_i__pl_\d+\]\s*=", tp)
        @test occursin(r"theta_cell\[plate_i__pl_\d+\]\s*=", tp)
        @test occursin(r"theta\[plate_i__pl_\d+\]\s*=\s*theta_cell\[plate_i__pl_\d+\]", tp)
    end
    let model_block = stan_block(code, "model")
        @test occursin(r"theta_cell_z\[plate_i__pl_\d+\]\s*~\s*std_normal", model_block)
        @test occursin(r"y\[plate_i__pl_\d+\]\s*~\s*normal\(theta_cell\[plate_i__pl_\d+\]", model_block)
    end

    @test transpiles(c3_plate_vector_submodel_model)
    @test stanc_compiles(c3_plate_vector_submodel_model)
    vector_code = stan_code(c3_plate_vector_submodel_model)
    @test occursin("matrix[k, n] theta_cell_z", stan_block(vector_code, "parameters"))
    @test occursin(r"theta_cell_z\[:, plate_i__pl_\d+\]\s*~\s*std_normal", stan_block(vector_code, "model"))
    @test occursin(r"theta\[:, plate_i__pl_\d+\]\s*=\s*theta_cell\[:, plate_i__pl_\d+\]", stan_block(vector_code, "transformed parameters"))
end

"""
Verify `slic: public plate emits N-dimensional outer loops` in an isolated test item.
"""
@testitem "slic: public plate emits N-dimensional outer loops" tags=[:slic, :plate] setup=[StanBlocksImports, StanBlocksTestSetup] begin
    @test transpiles(c3_plate_outer_int_model)
    @test stanc_compiles(c3_plate_outer_int_model)
    int_code = stan_code(c3_plate_outer_int_model)
    @test occursin("vector[4] theta_z", stan_block(int_code, "parameters"))
    @test occursin("vector[4] theta", stan_block(int_code, "transformed parameters"))
    @test occursin(r"for\(plate_i__pl_\d+ in 1:4\)", stan_block(int_code, "model"))

    @test transpiles(c3_plate_outer_2d_scalar_model)
    @test stanc_compiles(c3_plate_outer_2d_scalar_model)
    scalar2_code = stan_code(c3_plate_outer_2d_scalar_model)
    @test occursin("matrix[2, 3] theta_z", stan_block(scalar2_code, "parameters"))
    @test occursin("matrix[2, 3] theta", stan_block(scalar2_code, "transformed parameters"))
    let model_block = stan_block(scalar2_code, "model")
        @test occursin(r"for\(plate_i1__pl_\d+ in 1:2\)", model_block)
        @test occursin(r"for\(plate_i2__pl_\d+ in 1:3\)", model_block)
        @test occursin(r"y\[plate_i1__pl_\d+, plate_i2__pl_\d+\]\s*~\s*normal", model_block)
        @test occursin(r"theta_z\[plate_i1__pl_\d+, plate_i2__pl_\d+\]\s*~\s*normal", model_block)
    end

    @test transpiles(c3_plate_outer_2d_vector_model)
    @test stanc_compiles(c3_plate_outer_2d_vector_model)
    vector2_code = stan_code(c3_plate_outer_2d_vector_model)
    @test occursin("array[3] matrix[k, 2] theta_z", stan_block(vector2_code, "parameters"))
    @test occursin("array[3] matrix[k, 2] theta", stan_block(vector2_code, "transformed parameters"))
    @test occursin(
        r"theta_z\[plate_i2__pl_\d+, :, plate_i1__pl_\d+\]\s*~\s*std_normal",
        stan_block(vector2_code, "model"),
    )

    @test transpiles(c3_plate_outer_3d_scalar_model)
    @test stanc_compiles(c3_plate_outer_3d_scalar_model)
    scalar3_code = stan_code(c3_plate_outer_3d_scalar_model)
    @test occursin("array[4] matrix[2, 3] theta_z", stan_block(scalar3_code, "parameters"))
    @test occursin(
        r"theta_z\[plate_i3__pl_\d+, plate_i1__pl_\d+, plate_i2__pl_\d+\]\s*~\s*normal",
        stan_block(scalar3_code, "model"),
    )

    @test transpiles(c3_plate_outer_4d_scalar_model)
    @test stanc_compiles(c3_plate_outer_4d_scalar_model)
    scalar4_code = stan_code(c3_plate_outer_4d_scalar_model)
    @test occursin("array[4, 5] matrix[2, 3] theta_z", stan_block(scalar4_code, "parameters"))
    @test occursin(
        r"theta_z\[plate_i3__pl_\d+, plate_i4__pl_\d+, plate_i1__pl_\d+, plate_i2__pl_\d+\]\s*~\s*normal",
        stan_block(scalar4_code, "model"),
    )

    @test !transpiles(@slic (;) begin
        theta ~ plate(; outer=()) do i
            i
        end
    end; re=false)
    @test !transpiles(@slic (;) begin
        theta ~ plate(; outer=(2, 3)) do i
            i
        end
    end; re=false)
end

"""
Verify `slic: public plate emits heterogeneous vector cells` in an isolated test item.
"""
@testitem "slic: public plate emits heterogeneous vector cells" tags=[:slic, :plate, :ragged] setup=[StanBlocksImports, StanBlocksTestSetup] begin
    for model in (c3_plate_ragged_model, c3_plate_ragged_submodel_model)
        @test transpiles(model)
        @test stanc_compiles(model)
    end

    direct = stan_code(c3_plate_ragged_model)
    let td = stan_block(direct, "transformed data")
        # rv `b`, fresh cell-local `z` → `b_z`; ragged carriers keep the `__pl_*` suffixes.
        @test occursin(r"array\[num_elements\(K\)\] int b_z__pl_len_\d+", td)
        @test occursin(r"b_z__pl_len_\d+\[plate_i__pl_\d+\] = K\[plate_i__pl_\d+\]", td)
        @test occursin("cumulative_sum", td)
    end
    @test occursin(r"vector\[sum\(b_z__pl_len_\d+\)\] b_z__pl_mem_\d+", stan_block(direct, "parameters"))
    @test occursin(
        r"b_z__pl_mem_\d+\[ragged_start\(.*\):ragged_end\(.*\)\] ~ std_normal",
        stan_block(direct, "model"),
    )
    @test occursin(r"vector\[sum\(b__pl_len_\d+\)\] b__pl_mem_\d+", stan_block(direct, "transformed parameters"))

    called = stan_code(c3_plate_ragged_submodel_model)
    @test occursin(r"vector\[sum\(b_cell_z__pl_len_\d+\)\] b_cell_z__pl_mem_\d+", stan_block(called, "parameters"))
    @test occursin(r"(?s)b_cell_z__pl_mem_\d+\[.*?\]\s*~\s*std_normal", stan_block(called, "model"))
    @test occursin(r"(?s)b__pl_mem_\d+\[.*?\]\s*=\s*b_cell__pl_mem_\d+\[", stan_block(called, "transformed parameters"))
end

"""
Verify `slic: ragged obs broadcasts a distribution across groups (obs-outside)`.

Snag ragged-dist-arg-dcffbc1b: a plate collects per-cell VECTORS of differing
lengths into a ragged `mu` (`tuple(vector, array[] int)`); the user then applies a
TOP-LEVEL observation model to the WHOLE ragged result — `ys ~ normal(mu, sigma)` —
instead of keeping the obs in-cell. Previously this aborted at TRACING time: the
auto-generated posterior-predictive `_rng` draw hit
`normal_rng(::tuple(vector, array[] int), ::real)`, which has no tracetype.

The obs is now BROADCAST ACROSS the ragged groups (never flattened): it lowers to a
compiler-owned per-group loop `for g in 1:length(ys): ys[g] ~ dist(mu[g], …)`
(ragged distribution args sliced per group via `_ragged_group_arg`; shared
scalar/dense args pass through), reusing the ragged-prior density-loop machinery.
The indexed data obs routes to the model block only (no auto-GQ), matching the
obs-in-cell plate form.

Correctness for BOTH:
- univariate `normal` reduces to the same per-group density as the obs-in-cell
  twin — numerically identical log density (checked via BridgeStan);
- vector-valued `multi_normal` KEEPS the ragged structure — each `ys[g]` is ONE
  multivariate observation with its own group vector `mu[g]`, NOT a flattened
  backing (which would silently change the model).
"""
@testitem "slic: ragged obs broadcasts a distribution across groups (obs-outside)" tags=[:slic, :plate, :ragged, :bridgestan] setup=[StanBlocksImports, StanBlocksTestSetup] begin
    # (A) univariate normal — per-group broadcast, equal to the obs-in-cell twin.
    model = c3_plate_ragged_obs_outside_model
    @test transpiles(model)
    @test stanc_compiles(model)
    let mb = stan_block(stan_code(model), "model")
        # Per-group loop over the ragged obs — NOT a flattened `.mem` density.
        @test occursin(r"for\(g__ro_\d+ in 1:num_elements_RaggedVector\(ys\)\)", mb)
        @test occursin(r"getindex_RaggedVector\(ys, g__ro_\d+\) ~ normal\(mu__pl_mem_\d+\[", mb)
    end
    # Numerically identical to the obs-in-cell twin, end-to-end through BridgeStan.
    p_out = instantiate(model)
    p_in  = instantiate(c3_plate_ragged_obs_incell_model)
    @test LogDensityProblems.dimension(p_out) == 1
    let x = [0.3]
        lp_out, g_out = LogDensityProblems.logdensity_and_gradient(p_out, x)
        @test isfinite(lp_out)
        @test all(isfinite, g_out)
        @test lp_out ≈ LogDensityProblems.logdensity(p_in, x)
    end

    # (B) vector-valued multi_normal — the ragged structure MUST be preserved: each
    # group is ONE multivariate observation `ys[g] ~ multi_normal(mm[g], Sig)`.
    mvn = c3_ragged_obs_broadcast_mvn_model
    @test transpiles(mvn)
    @test stanc_compiles(mvn)
    let mb = stan_block(stan_code(mvn), "model")
        @test occursin(r"for\(g__ro_\d+ in 1:num_elements_RaggedVector\(ym\)\)", mb)
        @test occursin(
            r"getindex_RaggedVector\(ym, g__ro_\d+\) ~ multi_normal\(getindex_RaggedVector\(mm, g__ro_\d+\), Sig\)",
            mb,
        )
    end
end

"""
Snag build-a-declarat-ab2d2471 (reported by BRM): a declaration-driven prior /
posterior generative artifact needs two things StanBlocks did not give it.

(A) A per-cell DATA observation inside a plate produced NO generated quantity at
all — the indexed data-LHS `~` routed to `(:model,)` because the whole-LHS
`_gen`/`_likelihood` expansion cannot write one cell of a read-only Stan data
variable. It now also emits a gq clone of the loop that fills a compiler-owned
`<obs>_gen` twin, declared with the observation's OWN type.

(B) `lkj_corr_cholesky_rng` was declared without a return type, so the auto-GQ
redraw of a cv-tainted `L::cholesky_factor_corr[K] ~ lkj_corr_cholesky(eta)` died
in `stan_code` with `tracetype not defined for L::anything = …_rng(…)` — the trap
already recorded in the primer's cv section ("mark the group count, NEVER
n_terms").

Out of scope by design and asserted as such: the pointwise `<obs>_likelihood`
companion, and RAGGED observation bases (a RaggedVector is a compile-time view
over flat memory with no declarable Stan twin, so it keeps the model-only route).
"""
@testitem "slic: plate per-cell observations emit a _gen twin; lkj gq redraw" tags=[:slic, :plate, :bridgestan] setup=[StanBlocksImports, StanBlocksTestSetup] begin
    # (A) per-cell observation inside a plate → `y_gen` in generated quantities.
    obs_plate = @slic (; n_groups = 5, k = 3, y = [0.2, -0.1, 0.3, 0.0, 0.4]) begin
        L::cholesky_factor_corr[k] ~ lkj_corr_cholesky(2.0)
        tau::vector[k] ~ normal(0.0, 1.0; lower = 0.0)
        b::vector[k] ~ plate(y; outer = (n_groups,)) do yi
            cell ~ c3_plate_fixed_correlated_cell(k, L, tau)
            yi ~ normal(cell[1], 0.5)
            cell
        end
    end
    @test transpiles(obs_plate)
    @test stanc_compiles(obs_plate)
    let code = stan_code(obs_plate)
        gq = stan_block(code, "generated quantities")
        @test occursin("vector[y_n] y_gen", gq)
        @test occursin(r"y_gen\[plate_i\w*\] = normal_rng\(", gq)
        # The observation still contributes its density to the model block, and the
        # twin is a gq OUTPUT — never a re-declared data input.
        @test occursin(r"y\[plate_i\w*\] ~ normal\(", stan_block(code, "model"))
        @test !occursin("y_gen", stan_block(code, "data"))
        # Out of scope: the pointwise likelihood companion.
        @test !occursin("y_likelihood", code)
    end
    # End-to-end: the draws are real gq output and finite.
    let p = instantiate(obs_plate), d = LogDensityProblems.dimension(p)
        names = BridgeStan.param_names(p.model; include_tp = true, include_gq = true)
        idx = findall(n -> startswith(n, "y_gen"), names)
        @test length(idx) == 5
        drawn = BridgeStan.param_constrain(
            p.model, zeros(d);
            include_tp = true, include_gq = true, rng = BridgeStan.StanRNG(p.model, 1234),
        )
        @test all(isfinite, drawn[idx])
    end

    # A RAGGED observation base has no declarable twin → model-only, unchanged.
    let code = stan_code(c3_plate_ragged_obs_outside_model)
        @test !occursin("ys_gen", code)
    end

    # A COMPOUND data-qualified LHS is not an indexed observation. The twin
    # matcher must not descend into it the way `_base_lhs_symbol` does for a
    # compiler-injected fill — doing so found the innermost leaf `v` and emitted
    # the nonsense `(sum(v_gen) + c) = normal_rng(...)`.
    compound_lhs = @slic (; v = [0.1, 0.2, 0.3], c = 1.0) begin
        mu ~ std_normal()
        (sum(v) + c) ~ normal(mu, 1.0)
    end
    @test transpiles(compound_lhs)
    @test stanc_compiles(compound_lhs)
    @test !occursin("v_gen", stan_code(compound_lhs))

    # (B) cv-tainted lkj prior redraws in generated quantities instead of blowing
    # up at render time; `matrix`, not `cholesky_factor_corr`, matching the
    # dirichlet/simplex precedent for a gq redraw's container.
    lkj_base = @slic (; K = 3, eta = 2.0, y = 0.3) begin
        L::cholesky_factor_corr[K] ~ lkj_corr_cholesky(eta)
        y ~ normal(sum(to_vector(L)), 0.1)
    end
    @test transpiles(lkj_base)
    @test stanc_compiles(lkj_base)
    let cv_model = lkj_base(; eta = StanBlocks.stan.maybecv(:eta, 2.0)),
        code = stan_code(cv_model)
        @test stanc_compiles(cv_model)
        @test !occursin("L", stan_block(code, "parameters"))
        @test occursin(r"matrix\[K, K\] L = lkj_corr_cholesky_\w*rng\(K, eta\)",
                       stan_block(code, "generated quantities"))
    end
end

"""
Verify `slic: plate supports dense native-constrained vector cells` in an isolated test item.

Snag plate-constraine-90607054: a native-constrained per-cell parameter inside a
plate (`cell::simplex[k] ~ dirichlet(…)`) used to report `transpiles() == true`
while emitting invalid Stan — an UNCONSTRAINED `matrix[k,n]` cell decl (dropping
the constraint + jacobian) plus an `anything`-returning `dirichlet_lpdfs` helper
that `stanc` rejects. It now emits a native Stan `array[N…] <ct>[K]` parameter, so
Stan applies the constraint transform + jacobian per cell. Dense simplex / ordered
/ positive_ordered vector cells (fixed `K`, any `outer` rank) are supported;
ragged constrained cells (`K` per plate index) and constrained MATRIX families
(cholesky_*) remain rejected pending follow-up.
"""
@testitem "slic: plate supports dense native-constrained vector cells" tags=[:slic, :plate, :bridgestan] setup=[StanBlocksImports, StanBlocksTestSetup] begin
    simplex_cell = @slic (; n = 3, k = 3) begin
        p ~ plate(; outer = (n,)) do g
            cell::simplex[k] ~ dirichlet(rep_vector(1.0, k))
            cell
        end
    end
    @test transpiles(simplex_cell)
    @test stanc_compiles(simplex_cell)
    let code = stan_code(simplex_cell)
        # cell-local `cell` is namespaced under the plate result `p` → `p_cell`.
        @test occursin(r"array\[n\] simplex\[k\] p_cell", stan_block(code, "parameters"))
        @test occursin(r"p_cell\[plate_i\w*\] ~ dirichlet", stan_block(code, "model"))
        @test !occursin("anything", code)           # the bug: anything-typed _lpdfs helper
    end
    # Stan applies the simplex transform: n cells × (k-1) free coords each.
    @test LogDensityProblems.dimension(instantiate(simplex_cell)) == 3 * (3 - 1)

    # `ordered` family: N cells × K free coords each.
    ordered_cell = @slic (; n = 2, k = 4) begin
        p ~ plate(; outer = (n,)) do g
            c::ordered[k] ~ normal(0.0, 1.0)
            c
        end
    end
    @test stanc_compiles(ordered_cell)
    @test LogDensityProblems.dimension(instantiate(ordered_cell)) == 2 * 4

    # N-dimensional outer: `array[a, b] simplex[k]`.
    nd_cell = @slic (; a = 2, b = 2, k = 3) begin
        p ~ plate(; outer = (a, b)) do i, j
            c::simplex[k] ~ dirichlet(rep_vector(1.0, k))
            c
        end
    end
    @test stanc_compiles(nd_cell)
    @test occursin(r"array\[a, b\] simplex\[k\]", stan_block(stan_code(nd_cell), "parameters"))

    # Control: an UNCONSTRAINED `vector[k]` cell keeps the dense `matrix[k,n]` packing.
    plain_cell = @slic (; n = 3, k = 3) begin
        p ~ plate(; outer = (n,)) do g
            z::vector[k] ~ std_normal()
            z
        end
    end
    @test transpiles(plain_cell)
    @test occursin("matrix[k, n] p_z", stan_block(stan_code(plain_cell), "parameters"))

    # Still-unsupported constrained cells reject cleanly (no invalid Stan):
    #   ragged simplex — Stan cannot declare `array[n] simplex[K[g]]` (varying K).
    @test !transpiles(@slic (; K = [2, 3, 4]) begin
        p ~ plate(; outer = length(K)) do g
            c::simplex[K[g]] ~ dirichlet(rep_vector(1.0, K[g]))
            c
        end
    end; re = false)
    #   constrained MATRIX family — matrix cells are not yet supported in plate at all.
    @test !transpiles(@slic (; n = 2, k = 3) begin
        p ~ plate(; outer = (n,)) do g
            L::cholesky_factor_corr[k] ~ lkj_corr_cholesky(2.0)
            L
        end
    end; re = false)
end

"""
Verify `slic: BRM-shaped ragged constraints compose with plate` in an isolated test item.
"""
@testitem "slic: BRM-shaped ragged constraints compose with plate" tags=[:slic, :plate, :ragged] setup=[StanBlocksImports, StanBlocksTestSetup] begin
    @test transpiles(c3_plate_ragged_brm_model)
    @test stanc_compiles(c3_plate_ragged_brm_model)
    code = stan_code(c3_plate_ragged_brm_model)

    let parameters = stan_block(code, "parameters")
        @test occursin("p_free__rcm_", parameters)
        @test occursin(r"vector\[sum\(b_cell_z__pl_len_\d+\)\] b_cell_z__pl_mem_\d+", parameters)
    end
    let tp = stan_block(code, "transformed parameters")
        @test occursin("cholesky_factor_corr_jacobian", tp)
        @test occursin("to_matrix", tp)
        @test occursin(r"(?s)cell__pl_mem_\d+\[.*?\]\s*=\s*\(.*?to_matrix.*?\*.*?cell_z__pl_mem_\d+", tp)
    end
    let model_block = stan_block(code, "model")
        @test occursin("~ lkj_corr_cholesky(2.0)", model_block)
        @test occursin(r"(?s)cell_z__pl_mem_\d+\[.*?\]\s*~\s*std_normal", model_block)
        @test occursin(r"y ~ normal\(sum\(b__pl_mem_\d+\[", model_block)
    end
end

"""
Verify `slic: fixed-width constrained matrix accepted by matrix-typed plate cell` in an isolated test item.
"""
@testitem "slic: fixed-width constrained matrix accepted by matrix-typed plate cell" tags=[:slic, :plate, :ragged] setup=[StanBlocksImports, StanBlocksTestSetup] begin
    @test transpiles(c3_plate_fixed_correlated_model)
    @test stanc_compiles(c3_plate_fixed_correlated_model)
    code = stan_code(c3_plate_fixed_correlated_model)
    # Constraint metadata preserved at the caller: `L` keeps its cholesky
    # declaration and lkj prior, and the matrix-typed cell inlines against it.
    @test occursin("cholesky_factor_corr[k] L", stan_block(code, "parameters"))
    @test occursin("~ lkj_corr_cholesky(2.0)", stan_block(code, "model"))
    @test occursin("diag_pre_multiply(tau, L)", stan_block(code, "transformed parameters"))
end

"""
Verify `slic: public plate inside a called submodel` in an isolated test item.
"""
@testitem "slic: public plate inside a called submodel" tags=[:slic, :plate, :bridgestan] setup=[StanBlocksImports, StanBlocksTestSetup] begin
    # Snag regression (plate-inside-cal-69cec8bc): a fixed-vector plate in a CALLED
    # submodel body. Previously threw `TypeError: ... expected StanModel, got
    # SubModel` at `_plate_discover`. Fresh cell collections discover + emit under
    # their flattened (global) names at the root; `rv` keeps its local alias.
    @test transpiles(c3_plate_in_submodel_model)
    @test stanc_compiles(c3_plate_in_submodel_model)
    code = stan_code(c3_plate_in_submodel_model)
    # The submodel-internal cell `z` is a root parameter under its flattened name.
    # Namespaced under the inner plate result `b` (→ `b_z`), then flattened by the
    # submodel binding `b` (→ `b_b_z`).
    @test occursin(r"matrix\[k, n_groups\] b_b_z", stan_block(code, "parameters"))
    @test occursin(r"b_b_z\[:, b_plate_i__pl_\d+\]\s*~\s*std_normal", stan_block(code, "model"))
    let tp = stan_block(code, "transformed parameters")
        @test occursin(r"b_b\[:, b_plate_i__pl_\d+\]\s*=", tp)
        @test occursin(r"matrix\[k, n_groups\] b\s*=\s*b_b", tp)
    end

    # Identical parameters + target as the top-level plate ⇒ identical log density
    # and gradient. The top-level plate path is already covered above.
    p_sub = instantiate(stan_model(c3_plate_in_submodel_model))
    p_ref = instantiate(stan_model(c3_plate_in_submodel_reference))
    @test LogDensityProblems.dimension(p_sub) == LogDensityProblems.dimension(p_ref)
    for v in (
        cos.(1:LogDensityProblems.dimension(p_sub)) .* 0.7,
        sin.(1:LogDensityProblems.dimension(p_sub)),
        fill(0.15, LogDensityProblems.dimension(p_sub)),
        collect(range(-1.0, 1.0; length=LogDensityProblems.dimension(p_sub))),
    )
        @test LogDensityProblems.logdensity(p_sub, v) ≈
              LogDensityProblems.logdensity(p_ref, v) atol=1e-8
        _, g_sub = LogDensityProblems.logdensity_and_gradient(p_sub, v)
        _, g_ref = LogDensityProblems.logdensity_and_gradient(p_ref, v)
        @test g_sub ≈ g_ref atol=1e-8
    end

    # Snag regression (plate-submodel-c-f792c57a): caller names ≠ submodel arg names.
    # Both the fresh cell `b_b_z` (`z` namespaced under the plate result `b`, then
    # flattened by the submodel binding `b`) and the `rv` `b_b` must be sized by the
    # CALLER's `P`/`G`, never the raw submodel-local `k`/`n_groups` (absent at the root).
    @test transpiles(c3_plate_in_submodel_renamed)
    @test stanc_compiles(c3_plate_in_submodel_renamed)
    let rcode = stan_code(c3_plate_in_submodel_renamed)
        @test occursin(r"matrix\[P, G\] b_b_z", stan_block(rcode, "parameters"))
        let tp = stan_block(rcode, "transformed parameters")
            @test occursin(r"b_b\[:, b_plate_i__pl_\d+\]\s*=", tp)
            @test occursin(r"matrix\[P, G\] b\s*=\s*b_b", tp)
        end
        # The raw submodel-local size names must NOT leak into the emitted Stan.
        @test !occursin("n_groups", rcode)
        @test !occursin(r"\[k,|, k\]", rcode)
    end
    p_ren = instantiate(stan_model(c3_plate_in_submodel_renamed))
    p_ren_ref = instantiate(stan_model(c3_plate_in_submodel_renamed_reference))
    @test LogDensityProblems.dimension(p_ren) == LogDensityProblems.dimension(p_ren_ref)
    for v in (
        cos.(1:LogDensityProblems.dimension(p_ren)) .* 0.7,
        sin.(1:LogDensityProblems.dimension(p_ren)),
        collect(range(-1.0, 1.0; length=LogDensityProblems.dimension(p_ren))),
    )
        @test LogDensityProblems.logdensity(p_ren, v) ≈
              LogDensityProblems.logdensity(p_ren_ref, v) atol=1e-8
        _, g_ren = LogDensityProblems.logdensity_and_gradient(p_ren, v)
        _, g_ren_ref = LogDensityProblems.logdensity_and_gradient(p_ren_ref, v)
        @test g_ren ≈ g_ren_ref atol=1e-8
    end

    # Ragged vector cells inside a submodel are not wired yet — must fail loud
    # rather than emit invalid Stan (`matrix[K[sub_g], …]`).
    @slic c3_plate_ragged_in_sub(K::int[3]) = begin
        b ~ plate(; outer=length(K)) do g
            z::vector[K[g]] ~ std_normal()
            z
        end
        return b
    end
    @test_throws "ragged vector cells inside a called" stan_code(@slic (;K=[2, 3, 4], y=0.3) begin
        b ~ c3_plate_ragged_in_sub(K)
        y ~ normal(sum(b[1]), 1.0)
    end)
end

"""
Verify `slic: ragged simplex uses TP-inlined constraint transforms` in an isolated test item.
"""
@testitem "slic: ragged simplex uses TP-inlined constraint transforms" tags=[:slic, :ragged] setup=[StanBlocksImports, StanBlocksTestSetup] begin
    @test transpiles(c3_ragged_simplex_model)
    @test stanc_compiles(c3_ragged_simplex_model)
    code = stan_code(c3_ragged_simplex_model)

    @test occursin(r"vector\[sum\(jbroadcasted_sub", stan_block(code, "parameters"))
    let tp = stan_block(code, "transformed parameters")
        @test occursin("simplex_jacobian", tp)
        @test occursin("for(", tp)
        @test !occursin("tuple(", tp)
        @test !occursin(r"array\[.*\]\s+int", tp)
    end

    @test transpiles(c3_ragged_simplex_informative_model)
    @test stanc_compiles(c3_ragged_simplex_informative_model)
    informative_code = stan_code(c3_ragged_simplex_informative_model)
    let model_block = stan_block(informative_code, "model")
        @test occursin(r"for\(g__rc_\d+ in 1:num_elements\(K\)\)", model_block)
        @test occursin(r"p_mem__rc_\d+\[.*\] ~ dirichlet", model_block)
        @test occursin("alpha", model_block)
    end
end

"""
Verify `slic: ragged Cholesky factors use flattened matrix carriers` in an isolated test item.
"""
@testitem "slic: ragged Cholesky factors use flattened matrix carriers" tags=[:slic, :ragged, :bridgestan] setup=[StanBlocksImports, StanBlocksTestSetup] begin
    cases = (
        (c3_ragged_cholesky_corr_model, "cholesky_factor_corr_jacobian", 4),
        (c3_ragged_cholesky_cov_model, "cholesky_factor_cov_jacobian", 10),
    )
    for (model, jacobian, expected_dimension) in cases
        @test transpiles(model)
        problem = instantiate(model)
        @test LogDensityProblems.dimension(problem) == expected_dimension
        @test isfinite(LogDensityProblems.logdensity(problem, zeros(expected_dimension)))

        code = stan_code(model)
        @test occursin("p_free__rcm_", stan_block(code, "parameters"))
        let tp = stan_block(code, "transformed parameters")
            @test occursin(jacobian, tp)
            @test occursin("to_vector", tp)
            @test occursin("for(", tp)
            @test !occursin("tuple(", tp)
            @test !occursin(r"array\[.*\]\s+int", tp)
        end
        @test occursin("to_matrix", stan_block(code, "model"))
    end
    @test transpiles(c3_ragged_cholesky_corr_informative_model)
    @test stanc_compiles(c3_ragged_cholesky_corr_informative_model)
    @test occursin(
        r"to_matrix\(.*\) ~ lkj_corr_cholesky\(2\.0\)",
        stan_block(stan_code(c3_ragged_cholesky_corr_informative_model), "model"),
    )
end

"""
Verify `slic: ragged ordered constrained parameters` in an isolated test item.
"""
@testitem "slic: ragged ordered constrained parameters" tags=[:slic, :ragged, :bridgestan] setup=[StanBlocksImports, StanBlocksTestSetup] begin
    Ks = [1, 2, 4]
    ordered_model = @slic (;Ks, y=0.2) begin
        p::ordered[Ks] ~ flat()
        y ~ normal(sum(p[1]), 1.0)
    end
    positive_model = @slic (;Ks, y=0.2) begin
        p::positive_ordered[Ks] ~ flat()
        y ~ normal(sum(p[1]), 1.0)
    end

    for (model, jacobian) in (
        (ordered_model, "ordered_jacobian"),
        (positive_model, "positive_ordered_jacobian"),
    )
        @test transpiles(model)
        @test stanc_compiles(model)
        code = stan_code(model)
        @test LogDensityProblems.dimension(instantiate(stan_model(model))) == sum(Ks)

        let parameters = stan_block(code, "parameters")
            @test occursin(r"\bvector\[[^]]+\] p_free__rc_\d+;", parameters)
            @test !occursin("p_mem__rc_", parameters)
        end
        let tp = stan_block(code, "transformed parameters")
            @test occursin(r"\bvector\[[^]]+\] p_mem__rc_\d+;", tp)
            @test occursin(jacobian, tp)
        end
        @test !occursin(r"p_free__rc_\d+\s*~", stan_block(code, "model"))
    end

    @test transpiles(c3_ragged_ordered_informative_model)
    @test stanc_compiles(c3_ragged_ordered_informative_model)
    @test occursin(
        r"p_mem__rc_\d+\[.*\] ~ normal\(0\.0, 1\.0\)",
        stan_block(stan_code(c3_ragged_ordered_informative_model), "model"),
    )

    unsupported_matrix_family = @slic (;Ks, y=0.2) begin
        p::corr_matrix[Ks] ~ flat()
        y ~ normal(p[1, 1], 1.0)
    end
    @test !transpiles(unsupported_matrix_family; re=false)
end

"""
Verify `slic: ragged data indexes as a first-class container outside a plate`.

Decision `2026-07-17T00-14-01-598-1g0cf6y` (approach B, commit 197f5be): ragged
`Vector{Vector{<:Real}}` DATA now mints a NOMINAL `RaggedVector` usertype at ingest
(`stan_type`, builtin.jl), so `y[g]` slices to the g-th group vector and `length(y)`
counts groups in ANY model body — not only inside a plate. Backward-compatible by
construction: `RaggedVector <: types.usertype <: types.ntup` with `(:mem, :ends)`
arg_types, so every certified-ntup consumer (`_ragged_group_arg`, the plate ragged
slicer) keeps matching; the NEW `y[g]`-anywhere capability comes from RaggedVector's
own getindex/length UDFs firing on a bound (non-construction) data name. This is the
non-plate regression the plate testitems above never exercised (their ragged carriers
are PARAMETERS built by the plate, not ingested `Vector{Vector}` DATA).
"""
@testitem "slic: ragged data indexes as a first-class container outside a plate" tags=[:slic, :ragged, :bridgestan] setup=[StanBlocksImports, StanBlocksTestSetup] begin
    ragged = [[0.1, 0.2], [0.3, 0.4, 0.5], [0.6]]

    # `y[g]` (literal AND data-driven index) + `length(y)`, all OUTSIDE any plate.
    model = @slic (; y=ragged, g=2) begin
        mu ~ std_normal()
        first_group = y[1]
        driven_group = y[g]
        n_groups = length(y)
        (sum(first_group) + sum(driven_group) + n_groups) ~ normal(mu, 1.0)
    end

    @test transpiles(model)
    @test stanc_compiles(model)

    code = stan_code(model)
    # Ragged data materialises as a SINGLE `tuple(vector, array[] int)` data var —
    # both components are DATA, so Stan's "no integer (transformed) parameter tuple"
    # restriction does not apply.
    @test occursin(r"tuple\(vector\[y_mem_n\], array\[y_ends_n\] int\) y;", stan_block(code, "data"))
    let td = stan_block(code, "transformed data")
        # `y[g]` routes through RaggedVector's getindex UDF (a bound data name is NOT a
        # construction), sized by the data-qualified per-group length.
        @test occursin("getindex_RaggedVector(y, 1)", td)
        @test occursin("getindex_RaggedVector(y, g)", td)
        @test occursin(r"vector\[ragged_length_RaggedVector\(y, 1\)\]", td)
        # `length(y)` counts groups via `size(ends)`.
        @test occursin("num_elements_RaggedVector(y)", td)
    end

    # End-to-end through BridgeStan: one free parameter (`mu`), finite lp + gradient.
    problem = instantiate(model)
    @test LogDensityProblems.dimension(problem) == 1
    lp, grad = LogDensityProblems.logdensity_and_gradient(problem, [0.5])
    @test isfinite(lp)
    @test all(isfinite, grad)
end

"""
Verify `slic: EachCol / EachRow are first-class column/row matrix views`.

Decision `2026-07-17T02-25-53-568-igj5dy` (commit 38bbada): `EachCol(X)` / `EachRow(X)`
make a matrix's columns / rows a first-class indexable container everywhere — the
eachcol/eachrow analogue of the ragged-data `RaggedVector` (item above). A construction
is a zero-cost COMPILE-TIME view (verbatim-bound, no wrapper tuple): `EachCol(X)[j]`
lowers to Stan's native `col(X, j)`, `EachRow(X)[i]` to `row(X, i)`, and
`length(EachCol(X))` to `cols(X)`. Pairs with `plate` to broadcast a submodel over
columns. (EachSlice deferred — it would need 3-D+ container support.)
"""
@testitem "slic: EachCol / EachRow are first-class column/row matrix views" tags=[:slic, :bridgestan] setup=[StanBlocksImports, StanBlocksTestSetup] begin
    X = reshape(collect(1.0:15.0), 3, 5)   # deterministic 3×5

    # (1) column/row indexing OUTSIDE a plate, end-to-end through BridgeStan.
    model = @slic (; X=X, y=0.3) begin
        beta ~ std_normal(; n=3)           # sized == rows(X)
        w    ~ std_normal(; n=5)           # sized == cols(X)
        a = EachCol(X)[1]' * beta          # col(X,1)' * beta  (scalar)
        b = EachRow(X)[2] * w              # row(X,2) * w      (scalar)
        y ~ normal(a + b, 1.0)
    end
    @test transpiles(model)
    @test stanc_compiles(model)
    code = stan_code(model)
    # Zero-cost view: the matrix materialises directly, no wrapper tuple.
    @test occursin(r"matrix\[X_m, X_n\] X;", stan_block(code, "data"))
    @test !occursin("tuple(", code)
    # `[j]`/`[i]` lower to Stan-native col/row.
    @test occursin("col(X, 1)", code)
    @test occursin("row(X, 2)", code)
    # dim = beta(3) + w(5); finite lp + gradient.
    problem = instantiate(model)
    @test LogDensityProblems.dimension(problem) == 8
    lp, grad = LogDensityProblems.logdensity_and_gradient(problem, fill(0.2, 8))
    @test isfinite(lp)
    @test all(isfinite, grad)

    # (2) `length(EachCol(X))` == cols(X), usable as a parameter size.
    sized = @slic (; X=X, y=0.3) begin
        n = length(EachCol(X))
        w ~ std_normal(; n=n)
        y ~ normal(sum(w), 1.0)
    end
    @test transpiles(sized)
    @test stanc_compiles(sized)
    @test occursin("cols(X)", stan_code(sized))

    # (3) plate broadcasts a submodel over columns — `EachCol(X)` as the iterable.
    plated = @slic (; X=X, y=0.3) begin
        beta ~ std_normal(; n=3)
        s ~ plate(EachCol(X); outer=5) do xj
            dot_product(xj, beta)
        end
        y ~ normal(sum(s), 1.0)
    end
    @test transpiles(plated)
    @test stanc_compiles(plated)
    @test occursin(
        r"s\[plate_i\w*\] = dot_product\(col\(X, plate_i\w*\), beta\)",
        stan_block(stan_code(plated), "transformed parameters"),
    )
end

"""
Run stanc over every PosteriorDB model for which StanBlocks ships a SLIC implementation.
"""
@testitem "posteriordb: implemented models pass stanc" tags=[:posteriordb, :stanc] setup=[StanBlocksImports, StanBlocksTestSetup] begin
    pdb = PosteriorDB.database()
    failures = Pair{String,String}[]
    for posterior_name in PosteriorDB.posterior_names(pdb)
        post = StanBlocks.slic_implementation(PosteriorDB.posterior(pdb, posterior_name))
        isnothing(post) && continue
        try
            result = stanc_check(stan_code(post); warn_pedantic=false)
            result.ok || push!(failures, posterior_name => result.output)
        catch err
            push!(failures, posterior_name => sprint(showerror, err))
        end
    end
    if !isempty(failures)
        summaries = map(failures) do (name, message)
            compact = replace(strip(message), r"\s+" => " ")
            "$(name): $(first(compact, min(length(compact), 300)))"
        end
        error("$(length(failures)) PosteriorDB implementations failed stanc:\n" * join(summaries, "\n"))
    end
    @test isempty(failures)
end

"""
Verify the descriptor surface — `stan_descriptor` / `stan_operation` /
`stan_execute` — reflects one `@slic` declaration as identity + inputs + outputs
+ DERIVED operations, and fails closed on the cases a consumer cannot recover
from (an operation the model does not offer, a name that is both an input and an
output, an unbound required input).

Todo `Expose descriptor-bearing executable semantic models`: downstream
(BRM / SbPMX) must mount a StanBlocks model without maintaining its own parallel
operation list, so every assertion here is about something being DERIVED rather
than declared twice.
"""
@testitem "slic: model descriptor reflects inputs, outputs and derived operations" tags=[:slic, :descriptor] setup=[StanBlocksImports, StanBlocksTestSetup] begin
    base = @slic (; x = [1.0, 2.0, 3.0, 4.0, 5.0], y = [0.2, -0.1, 0.3, 0.0, 0.4]) begin
        sigma ~ exponential(1.0)
        beta ~ normal(0.0, 10.0)
        y ~ normal(beta * x, sigma)
    end
    d = stan_descriptor(base; name = :demo)

    # --- identity ------------------------------------------------------------
    @test d.name == :demo
    # Content-derived, so reflecting the same model twice agrees, and it agrees
    # with the key `instantiate` caches the compiled artifact under.
    @test d.id == stan_descriptor(base).id
    @test d.id == string(hash(stan_code(base)); base = 16)
    # The gensym'd model name must NOT leak into the identity.
    @test stan_descriptor(base; name = :other).id == d.id
    changed = @slic (; x = [1.0, 2.0, 3.0, 4.0, 5.0], y = [0.2, -0.1, 0.3, 0.0, 0.4]) begin
        sigma ~ exponential(2.0)
        beta ~ normal(0.0, 10.0)
        y ~ normal(beta * x, sigma)
    end
    @test stan_descriptor(changed).id != d.id

    # --- inputs --------------------------------------------------------------
    ins = Dict(i.name => i for i in d.inputs)
    @test haskey(ins, :y) && haskey(ins, :x)
    # `y` is what the model conditions on; `x` is a covariate. That distinction
    # comes from walking the traced `~` statements, not from any naming rule.
    @test ins[:y].observed
    @test !ins[:x].observed
    @test ins[:y].type == :vector && ins[:x].type == :vector
    @test ins[:y].value == [0.2, -0.1, 0.3, 0.0, 0.4]
    @test !ins[:y].held_out
    @test !any(i -> i.inlined, d.inputs)
    # `y_n` IS a data-block entry (it reaches the JSON), but it is another
    # input's declared size — re-binding `y` re-derives it, so a generated form
    # must never ask for it.
    @test ins[:y].size == (:y_n,)
    @test ins[:y_n].derived && !ins[:y].derived
    @test StanBlocks.required_inputs(d) == (:y, :x)

    # --- outputs + generative semantics --------------------------------------
    outs = Dict(o.name => o for o in d.outputs)
    @test outs[:sigma].kind == :parameter && outs[:sigma].generative == :posterior
    @test outs[:beta].kind == :parameter
    # `exponential` implies a positive support; the descriptor reports the
    # constraint the declaration actually carries.
    @test outs[:sigma].constraints.lower == 0.0
    # The compiler-owned twins are labelled by ROLE and point back at the
    # observation they derive from — a consumer never parses the `_gen` suffix.
    @test outs[:y_gen].kind == :generated_quantity
    @test outs[:y_gen].generative == :draw
    @test outs[:y_gen].source == :y
    @test outs[:y_likelihood].generative == :pointwise_loglik
    @test outs[:y_likelihood].source == :y
    @test [o.name for o in d.outputs if o.generative == :draw] == [:y_gen]

    # --- operations are DERIVED, not listed ----------------------------------
    opnames = [op.name for op in d.operations]
    @test opnames == [:transpile, :instantiate, :fit, :predict, :pointwise_loglik]
    @test stan_operation(d, :predict).outputs == (:y_gen,)
    @test stan_operation(d, :pointwise_loglik).outputs == (:y_likelihood,)
    @test :y in stan_operation(d, :fit).inputs
    @test stan_operation(d, :transpile).inputs == ()
    @test stan_execute(d, :transpile) == stan_code(base)

    # A prior-predictive model has no likelihood, hence no `:fit` — and no
    # observation, hence no `:predict`.
    prior_only = @slic (;) begin
        sigma ~ exponential(1.0)
        beta ~ normal(0.0, 10.0)
    end
    pd = stan_descriptor(prior_only; name = :prior_only)
    @test [op.name for op in pd.operations] == [:transpile, :instantiate]

    # cv-held-out: the ONLY observation moves to generated quantities, so the
    # model can still be instantiated but there is nothing left to fit.
    heldout = base(; y = StanBlocks.stan.maybecv(:y, [0.2, -0.1, 0.3, 0.0, 0.4]))
    hd = stan_descriptor(heldout; name = :heldout)
    @test Dict(i.name => i for i in hd.inputs)[:y].held_out
    @test :fit ∉ [op.name for op in hd.operations]
    @test :predict in [op.name for op in hd.operations]

    # --- fail closed ---------------------------------------------------------
    # (a) an operation the model does not offer names the ones it does.
    err = try; stan_operation(pd, :fit); catch e; sprint(showerror, e); end
    @test occursin("has no `fit` operation", err)
    @test occursin("`transpile`", err) && occursin("`instantiate`", err)
    @test_throws ErrorException stan_execute(pd, :predict; draws = [0.0], seed = 1)

    # (b) `:transpile` takes no keywords — silently ignoring them would look
    #     like a data re-bind that never happened.
    @test_throws ErrorException stan_execute(d, :transpile; y = [1.0])

    # (c) a data variable colliding with a compiler-owned twin cannot be
    #     resolved automatically: every consumer keys on the name.
    collide = @slic (; y = [0.2, -0.1], y_gen = [1.0, 2.0]) begin
        mu ~ normal(sum(y_gen), 1.0)
        y ~ normal(mu, 1.0)
    end
    cerr = try; stan_descriptor(collide); catch e; sprint(showerror, e); end
    @test occursin("BOTH a data input and a model output", cerr)
end

"""
Verify the descriptor's operations actually EXECUTE through the normal
StanBlocks → BridgeStan path: `:fit` yields the differentiable log-density a
sampler consumes, and `:predict` / `:pointwise_loglik` draw exactly the
generated quantities the descriptor advertised, addressed by the descriptor's
own names rather than by BridgeStan's flattened `name.i` spelling.
"""
@testitem "slic: descriptor operations execute through BridgeStan" tags=[:slic, :descriptor, :bridgestan] setup=[StanBlocksImports, StanBlocksTestSetup] begin
    m = @slic (; x = [1.0, 2.0, 3.0, 4.0, 5.0], y = [0.2, -0.1, 0.3, 0.0, 0.4]) begin
        sigma ~ exponential(1.0)
        beta ~ normal(0.0, 10.0)
        y ~ normal(beta * x, sigma)
    end
    d = stan_descriptor(m; name = :demo)

    prob = stan_execute(d, :fit)
    dim = LogDensityProblems.dimension(prob)
    @test dim == 2
    @test isfinite(LogDensityProblems.logdensity(prob, zeros(dim)))

    # One draw → one vector per advertised output, sized by the declaration.
    pred = stan_execute(d, :predict; problem = prob, draws = zeros(dim), seed = 1234)
    @test keys(pred) == (:y_gen,)
    @test length(pred.y_gen) == 5
    @test all(isfinite, pred.y_gen)

    ll = stan_execute(d, :pointwise_loglik; problem = prob, draws = zeros(dim), seed = 1234)
    @test keys(ll) == (:y_likelihood,)
    @test length(ll.y_likelihood) == 5
    @test all(isfinite, ll.y_likelihood)

    # Several draws → one column each.
    multi = stan_execute(d, :predict; problem = prob, draws = zeros(dim, 3), seed = 7)
    @test size(multi.y_gen) == (5, 3)
    # Same seed, same draws ⇒ reproducible.
    @test stan_execute(d, :predict; problem = prob, draws = zeros(dim), seed = 99).y_gen ==
          stan_execute(d, :predict; problem = prob, draws = zeros(dim), seed = 99).y_gen

    # Without `problem` the operation compiles for itself (same cached artifact).
    @test length(stan_execute(d, :predict; draws = zeros(dim), seed = 1234).y_gen) == 5

    # A generated quantity is an RNG draw: refuse rather than silently pick a seed.
    @test_throws ErrorException stan_execute(d, :predict; problem = prob, draws = zeros(dim))
    @test_throws ErrorException stan_execute(d, :predict; problem = prob, seed = 1)

    # Data re-binding flows through the executor, so a consumer never has to
    # rebuild the model by hand to predict on new data.
    rebound = stan_execute(d, :predict;
        y = [1.0, 1.0, 1.0], x = [1.0, 2.0, 3.0], draws = zeros(dim), seed = 5)
    @test length(rebound.y_gen) == 3
end

"""
Verify the descriptor over the PLATE shape — the case `e4f0964`'s two testitems
did not reach. A per-cell observation inside a compiler-owned plate loop gets
its `<base>_gen` twin from `_push_obs_gen_decl!` INSIDE a `ForExpr`, while
`_output_symbols` deliberately does not descend into such a loop (the base is
declared separately). That interaction, the ragged carve-out, and the
distinction between a gq observation-twin and a gq prior RE-DRAW are what this
item pins down.

Todo `2026-07-22T11-53-04-745-0ly84er`, from decision `0a6ftf8` resolving "no
preference": the two gaps I had flagged are reachable from pure `@slic`, so
none of this needs BRM.
"""
@testitem "slic: descriptor over plate observations, ragged carve-out and gq re-draws" tags=[:slic, :descriptor, :plate] setup=[StanBlocksImports, StanBlocksTestSetup] begin
    # --- per-cell plate observation: the twin is reached through a loop -------
    obs_plate = @slic (; n_groups = 5, k = 3, y = [0.2, -0.1, 0.3, 0.0, 0.4]) begin
        L::cholesky_factor_corr[k] ~ lkj_corr_cholesky(2.0)
        tau::vector[k] ~ normal(0.0, 1.0; lower = 0.0)
        b::vector[k] ~ plate(y; outer = (n_groups,)) do yi
            cell ~ c3_plate_fixed_correlated_cell(k, L, tau)
            yi ~ normal(cell[1], 0.5)
            cell
        end
    end
    d = stan_descriptor(obs_plate; name = :obs_plate)
    outs = Dict(o.name => o for o in d.outputs)

    # The twin is classified even though the `~` it came from lives inside a
    # compiler-owned loop and was rewritten to an assignment on the way in.
    @test haskey(outs, :y_gen)
    @test outs[:y_gen].kind == :generated_quantity
    @test outs[:y_gen].generative == :draw
    @test outs[:y_gen].source == :y
    @test outs[:y_gen].size == (:y_n,)
    # `y` is the observation, reached through the getindex chain of `y[plate_i…]`.
    @test Dict(i.name => i for i in d.inputs)[:y].observed

    # The loop index is NOT an output: `_output_symbols` must not descend into a
    # compiler-owned `for`, or every plate would advertise a phantom quantity.
    @test !any(o -> occursin("plate_i", string(o.name)), d.outputs)

    # A per-cell observation gets the DRAW but deliberately no pointwise
    # log-likelihood companion (the cell shape does not fix its container —
    # `d243f24`). The operation set must reflect that, not assume the pair.
    @test !haskey(outs, :y_likelihood)
    opnames = [op.name for op in d.operations]
    @test :predict in opnames
    @test :pointwise_loglik ∉ opnames
    @test stan_operation(d, :predict).outputs == (:y_gen,)

    # Derived sizes stay out of what a consumer must supply; the plate's own
    # scalars stay in.
    @test Set(StanBlocks.required_inputs(d)) == Set([:n_groups, :k, :y])

    # The plate's own carriers are bare `out::T` declarations too, and were
    # dropped by the same bug that dropped `y_gen` — assert them so a future
    # skip-the-DeclExpr regression fails here loudly rather than silently
    # shrinking every plate model's output set.
    @test outs[:b].kind == :transformed_parameter
    @test outs[:b_cell].kind == :transformed_parameter
    @test outs[:b_cell_z].kind == :parameter

    # --- ragged carve-out: no declarable shape ⇒ no twin ⇒ no :predict -------
    rd = stan_descriptor(c3_plate_ragged_obs_outside_model; name = :ragged)
    @test !any(o -> o.generative == :draw, rd.outputs)
    @test :predict ∉ [op.name for op in rd.operations]
    # It is still an observation — pass (1) sees the model-block `~` even though
    # pass (2) has no twin to find. The two passes are not redundant.
    @test Dict(i.name => i for i in rd.inputs)[:ys].observed
    @test :fit in [op.name for op in rd.operations]

    # --- a gq prior RE-DRAW is :derived, not :draw ---------------------------
    # cv-tainting `eta` flips `L` out of `parameters` into a generated-quantities
    # re-draw. That is a latent, not a predictive draw of an observation, and a
    # consumer plotting posterior predictions must be able to tell them apart.
    lkj_base = @slic (; K = 3, eta = 2.0, y = 0.3) begin
        L::cholesky_factor_corr[K] ~ lkj_corr_cholesky(eta)
        y ~ normal(sum(to_vector(L)), 0.1)
    end
    cd = stan_descriptor(lkj_base(; eta = StanBlocks.stan.maybecv(:eta, 2.0)); name = :cv_lkj)
    couts = Dict(o.name => o for o in cd.outputs)
    @test couts[:L].kind == :generated_quantity
    @test couts[:L].generative == :derived
    @test couts[:L].source === nothing
    # `y` is untainted, so its own twins are unaffected and still :draw.
    @test couts[:y_gen].generative == :draw && couts[:y_gen].source == :y
    @test [o.name for o in cd.outputs if o.generative == :draw] == [:y_gen]
    # `held_out` is CONTAGION-aware, not a record of what the user marked. Only
    # `eta` was passed through `maybecv`, but cv propagates through every
    # expression it reaches, so `y`'s own likelihood contribution is dropped too
    # — which is exactly why `:fit` is gone. Reporting `y` as un-held-out here
    # would contradict the operation set.
    cins = Dict(i.name => i for i in cd.inputs)
    @test cins[:eta].held_out && cins[:y].held_out
    @test :fit ∉ [op.name for op in cd.operations]
end
