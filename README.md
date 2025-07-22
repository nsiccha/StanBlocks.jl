# StanBlocks.jl (Stan backend)

Brings Julia syntax to Stan models by implementing a (limited) Julia to Stan transpilation with many caveats. 
See [`test/slic.jl`](test/slic.jl) for implementations of a few simple [`posteriordb`](https://github.com/stan-dev/posteriordb) models
and see [`src/slic_stan/builtin.jl`](src/slic_stan/builtin.jl) for a list of built-in functions and examples of user defined functions.

Current features include

* activity analysis (automatically determines what is `data`, `transformed_data`, `parameters`, `transformed parameters`, `model`, or `generated quantities`),
* automatically inferred types, shapes and constraints - including for user defined functions (including the function arguments, function body, and function return type),
* automatic posterior pointwise likelihood and predictive generation,
* (variadic) user defined functions,
* higher order (user defined) functions (such as `map`, `broadcasted`, `sum` and more),
* sub models,
* post-hoc model adjustment,
* named tuples,
* (approximate) automatic code formatting à la [Blue](https://github.com/JuliaDiff/BlueStyle),
* and more.


Upcoming features include, in order of priority and estimated arrival,

* easy runtime assertions (like Julia's `@assert`) - and support for other macros,
* model docstrings, 
* custom types (for method dispatch - this would help with more "Julia-style" broadcasting, e.g. via `Ref`),
* closures via Julia's [`Do-Block Syntax`](https://docs.julialang.org/en/v1/manual/functions/#Do-Block-Syntax-for-Function-Arguments) (to make within chain parallelization via [`reduce_sum`](https://mc-stan.org/docs/stan-users-guide/parallelization.html#reduce-sum) less painful),
* lower transpilation runtimes (currently, transpilation can sometimes take longer than compilation - there is currently at least one algorithmic inefficiency on top of the systemic implementation inefficiency),
* a much better user experience,
* more and better tests,
* keyword arguments,
* default arguments,
* inlining (to reduce potential runtime overhead),
* easier custom parameter transformations (going from sampler parametrization to user parametrization - aka as constraining),
* array comprehensions,
* a more complete (and more correct) coverage of built-in Stan functions,
* better name resolution (currently user defined functions or sub models have to be defined in `Main`),
* functions that mutate their arguments (solved via inlining),
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


# StanBlocks.jl (Julia backend - deprecated)

Implements many - but currently not all - of the Bayesian models in [`posteriordb`](https://github.com/stan-dev/posteriordb)
by implementing Julia macros and functions which mimick Stan blocks and functions respectively, with relatively light dependencies. 
Using the macros and functions defined in this package, the "shortest" `posteriordb` model ([`earn_height.stan`](https://github.com/stan-dev/posteriordb/blob/master/posterior_database/models/stan/earn_height.stan))

```stan
data {
  int<lower=0> N;
  vector[N] earn;
  vector[N] height;
}
parameters {
  vector[2] beta;
  real<lower=0> sigma;
}
model {
  earn ~ normal(beta[1] + beta[2] * height, sigma);
}
```

becomes

```julia
julia_implementation(::Val{:earn_height}; N, earn, height, kwargs...) = begin 
    @stan begin 
        @parameters begin
            beta::vector[2]
            sigma::real(lower=0.)
        end
        @model begin
            earn ~ normal(@broadcasted(beta[1] + beta[2] * height), sigma);
        end
    end
end
```

Instantiating the posterior (i.e. model + data) requires loading [`PosteriorDB.jl`](https://github.com/sethaxen/PosteriorDB.jl),
which provides access to the datasets, e.g. to load the `earnings-earn_height` posterior (`earn_height` model + `earning` data):

```julia
import StanBlocks, PosteriorDB

pdb = PosteriorDB.database()
post = PosteriorDB.posterior(pdb, "earnings-earn_height")

jlpdf = StanBlocks.julia_implementation(post)
jlpdf(randn(StanBlocks.dimension(jlpdf))) # Returns some number
```

# Caveats

## Differences in the returned log-density

Stan's default "sampling statement" (e.g. `y ~ normal(mu, sigma);`) automatically drops constant terms (unless configured differently), see [https://mc-stan.org/docs/reference-manual/statements.html#log-probability-increment-vs.-distribution-statement](https://mc-stan.org/docs/reference-manual/statements.html#log-probability-increment-vs.-distribution-statement). 
Constant terms are terms which do not depend on model parameters, and this package's macros and functions currently do not try to figure out which terms do not depend on model parameters, and as such we never drop them.
This may lead to (constant) differences in the computed log-densities from the Stan and Julia implementations.

## Some models are not implemented yet, or may have smaller or bigger errors

I've implemented many of the models, but I haven't implemented all of them, and I probably have made some mistakes in implementing some of them.

## Some models may have been implemented suboptimally

Just that.

# Using and testing the implementations

See [https://nsiccha.github.io/StanBlocks.jl/performance.html#tabular-data](https://nsiccha.github.io/StanBlocks.jl/performance.html#tabular-data) for an overview of (hopefully) correctly implemented models.

See [`test/runtests.jl`](https://github.com/nsiccha/StanBlocks.jl/blob/main/test/runtests.jl) for a way to run and check the models. 
After importing `PosteriorDB`, `StanLogDensityProblems` and `LogDensityProblems`, you should have access to reference Stan implementations of the log density and of its gradient, see the documentation of `StanLogDensityProblems.jl`.
The Stan log density can then be compared to the Julia log density as is, and after loading Julia's AD packages, you can also compare the Stan log density gradient to the Julia log density gradient.