# StanBlocks.jl (Stan backend)

[![Dev Docs](https://img.shields.io/badge/docs-dev-blue.svg)](https://nsiccha.github.io/StanBlocks.jl/dev/)
[![CI](https://github.com/nsiccha/StanBlocks.jl/actions/workflows/test.yml/badge.svg)](https://github.com/nsiccha/StanBlocks.jl/actions/workflows/test.yml)

Brings Julia syntax to Stan models by implementing a (limited) Julia to Stan transpilation with many caveats. 
See [`test/slic.jl`](test/slic.jl) for implementations of a few simple [`posteriordb`](https://github.com/stan-dev/posteriordb) models
and see [`src/slic_stan/builtin.jl`](src/slic_stan/builtin.jl) for a list of built-in functions and examples of user defined functions.

Current features include

* activity analysis (automatically determines what is `data`, `transformed_data`, `parameters`, `transformed parameters`, `model`, or `generated quantities`),
* automatically inferred types, shapes and constraints - including for user defined functions (including the function arguments, function body, and function return type),
* higher order (user defined) functions (such as `map`, `broadcasted`, `sum` and more),
* (limited) dynamic dispatch - it's currently possible to dispatch on type (base type + number of dimensions), just the number of dimensions, and function argument types for higher order functions à la `f(::typeof(g)) = x`,
* automatic extraction of shapes for user defined functions - define a function like `f(x::vector[n]) = ...` and `n` will be available in the function body,
* sub models,
* post-hoc model adjustment,
* (variadic) user defined functions,
* named tuples,
* automatic posterior pointwise likelihood and predictive generation,
* (approximate) automatic code formatting à la [Blue](https://github.com/JuliaDiff/BlueStyle),
* and more.


Recently shipped:

* `return_type_of(f, args...)` — public transpile-time return-type and shape queries for registered SLIC callables,
* user-defined Julia macros expand transparently inside `@slic` / `@deffun` bodies (`@views`, `@.`, `@inbounds`, anything),
* `@stan_assert cond [msg]` — runtime assertion that lowers to `if !cond reject(msg)`,
* `@inline` UDFs / trailing `!`: every call expands at the call site, no Stan `functions {}` entry — supports multi-statement bodies, varargs, higher-order arguments, and caller-scope mutation through `f!(buf, …)`-style helpers,
* module-aware name resolution: `@deffun`s defined in package extensions are found automatically.

Upcoming features, in rough priority order:

* generated functions — already implicit (every call site re-traces with concrete arg types) but no compile-once cache,
* model docstrings (top-level `@slic` doc-prefix propagation),
* custom types (for method dispatch — would help "Julia-style" broadcasting, e.g. via `Ref`),
* closures via Julia's [`Do-Block Syntax`](https://docs.julialang.org/en/v1/manual/functions/#Do-Block-Syntax-for-Function-Arguments) (to make within-chain parallelization via [`reduce_sum`](https://mc-stan.org/docs/stan-users-guide/parallelization.html#reduce-sum) less painful),
* lower transpilation runtimes,
* type annotations as transpile-time assertions,
* automatic (runtime) shape-compatibility checks — auto-emit `if (rows(x) != rows(y)) reject(…)` at function entry,
* a much better user experience,
* more and better tests,
* keyword + default arguments to `@deffun`,
* `void` UDFs (side-effect-only functions),
* easier custom parameter transformations (sampler parametrization ↔ user parametrization),
* broader array-comprehension forms beyond the bounded one-dimensional `@deffun` subset,
* a more complete (and more correct) coverage of built-in Stan functions,
* elimination of unused (size) variables in UDFs,
* and more.

Almost anything that's possible in Julia should be possible to be transpiled to Stan. 
Of course, unless Stan is much faster than Julia (+Mooncake or Enzyme) for the model in question, 
just sticking to Julia comes with many advantages. 

Features which I am on the fence about, but currently not planning to implement:

* a Julia backend,
* `target +=` statements,
* top level control flow,
* top level mutability,
* getting rid of superfluous parentheses.

Features which are **NOT** planned:

* (automatically) transpiling Julia functions which have not been defined via `@deffun`. 

The `earn_height.stan` model below becomes 

```julia
using StanBlocks
import PosteriorDB, StanLogDensityProblems, JSON

# Get data from PosteriorDB
pdb = PosteriorDB.database()
post = PosteriorDB.posterior(pdb, "earnings-earn_height")
(;earn, height) = (;Dict([Symbol(k)=>v for (k, v) in pairs(PosteriorDB.load(PosteriorDB.dataset(post)))])...)

# Model definition
earn_height_model = @slic begin 
    beta ~ flat(;n=2)
    sigma ~ flat(;lower=0.)
    earn ~ normal(beta[1]+beta[2]*to_vector(height), sigma)
end
# Not compiled yet
earn_height_posterior = earn_height_model(; earn, height)
# Prints the Stan model code
println(stan_code(earn_height_posterior))
# Compiled (requires StanLogDensityProblems and JSON)
earn_height_problem = stan_instantiate(earn_height_posterior)
```

See the [user guide](https://nsiccha.github.io/StanBlocks.jl/dev/) for `@deffun`, `@inline` UDFs, sub-models, posterior pointwise likelihood / predictive draws, the full built-in catalog, and a Bruno- / BRM-scale example.

# Caveats

## Constant terms in the log density

Stan's `~` statement [drops constants](https://mc-stan.org/docs/reference-manual/statements.html#log-probability-increment-vs.-distribution-statement). StanBlocks does **not** — it always emits the full log density. The absolute value differs from a hand-written Stan model, but the posterior geometry is identical and sampling is unaffected.

## No control flow at the model level

`for`/`while`/`if`/`&&`/`||`/ternary/comprehensions are not allowed in `@slic` bodies. Move that logic into a `@deffun` body, or use a vectorised form.
