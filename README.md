# StanBlocks.jl

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
jlpdf(randn(3)) # Returns some number
```

# Caveats

## Getting the dimension of a posterior

I hadn't really thought things fully through at the beginning, so right now there's no "easy" way to get the dimension of a posterior.
The easiest way to get that dimension is currently to also load `StanLogDensityProblems` and `LogDensityProblems` to compile and load the 
underlying Stan model, and than query the resulting `StanProblem` for its dimension, e.g. via 

```julia
import StanLogDensityProblems, LogDensityProblems

# Initialize `post` as above. 
stan_problem = StanLogDensityProblems.StanProblem(post, ".", force=true)
n = LogDensityProblems.dimension(stan_problem)
```

## Differences in the returned log-density

Stan's default "sampling statement" (e.g. `y ~ normal(mu, sigma);`) automatically drops constant terms (unless configured differently), see [https://mc-stan.org/docs/reference-manual/statements.html#log-probability-increment-vs.-distribution-statement](https://mc-stan.org/docs/reference-manual/statements.html#log-probability-increment-vs.-distribution-statement). 
Constant terms are terms which do not depend on model parameters, and this package's macros and functions currently do not try to figure out which terms do not depend on model parameters, and as such we never drop them.
This may lead to (constant) differences in the computed log-densities from the Stan and Julia implementations.

## Some models are not implemented yet, or may have smaller or bigger errors

I've implemented many of the models, but I haven't implemented all of them, and I probably have made some mistakes in implementing some of them.

## Some models may have been implemented suboptimally

Just that.

# Using and testing the implementations

See `test/runtests.jl` for a way to run and check the models. 
After importing `PosteriorDB`, `StanLogDensityProblems` and `LogDensityProblems`, you should have access to reference Stan implementations of the log density and of its gradient, see the documentation of `StanLogDensityProblems.jl`.
The Stan log density can then be compared to the Julia log density as is, and after loading Julia's AD packages, you can also compare the Stan log density gradient to the Julia log density gradient.