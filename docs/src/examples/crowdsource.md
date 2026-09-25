# A family of latent-truth crowdsourcing models

This page ports the model family from
[`crowdsource-computo-bayes`](https://github.com/seongwoohan/crowdsource-computo-bayes).
Repeated binary ratings are observed, but the true class of each item is not.
The family asks how much item-specific and rater-specific structure is needed
to explain those ratings.

The complete Julia source is shown below. The documentation build evaluates
that exact source, then obtains all 19 complete Stan programs from the resulting
model family. The historical custom distribution name `no_good_name` is
kept solely so the port remains traceable to its source; it represents the
latent-truth, marginalized crowdsourcing likelihood described here.

## The full model

Let `z_i` be the unobserved binary truth for item `i` and
`pi_` its population prevalence. For rating `k`, write
`i = item[k]` and `j = rater[k]`; those indices select the relevant
parameters.
Conditional on a positive truth, the probability of a positive rating is

`lambda[i] + (1 - lambda[i]) *
inv_logit(delta[i] * (alpha_sens[j] - beta_[i]))`.

Conditional on a negative truth, it is

`(1 - lambda[i]) *
inv_logit(-delta[i] * (alpha_spec[j] - beta_[i]))`.

Thus the full model contains:

- `pi_`, the prevalence of positive items;
- `alpha_sens[j]` and `alpha_spec[j]`, separate sensitivity- and
  specificity-side abilities for each rater;
- `beta_[i]`, an item location or difficulty;
- positive `delta[i]`, an item discrimination scale; and
- `lambda[i]`, an item-specific response pathway that raises the
  positive-truth probability while reducing the negative-truth probability.

`no_good_name_lpmfs` integrates out each `z_i` analytically with
`log_sum_exp`. It first accumulates the log probabilities of all ratings
for an item under each possible truth, then mixes those two item-level terms
with `log(pi_)` and `log1m(pi_)`. The generated quantities simulate
ratings from the same model and compare observed and replicated vote totals by
item and by rater.

## From one model to 19 variants

The short labels record which replacement transforms have been applied to
`full`. Their letters have a precise meaning:

| Transform | Restriction |
|:--|:--|
| `a` | Fix every `lambda[i]` to zero, removing the special response pathway. |
| `b` | Fix every discrimination `delta[i]` to one. |
| `c` | Fix every item location `beta_[i]` to zero. |
| `d` | Replace separate sensitivity and specificity vectors with one positive per-rater accuracy vector. |
| `de` | Replace both ability vectors with one positive scalar shared by every rater and both truth classes. This is one transform, not `d` followed by `e`. |
| `e` | Keep distinct sensitivity and specificity abilities, but make each a scalar shared by every rater. |

Concatenation means composition, so `abc` applies `a`, `b`,
and `c` to the full model. Every displayed member is accounted for below:

| Label | Resulting model |
|:--|:--|
| `full` | All item and rater effects vary. |
| `a` | No `lambda` pathway. |
| `ab` | No `lambda` pathway; discrimination fixed. |
| `abc` | As `ab`; item location also fixed. |
| `abcd` | As `abc`; one accuracy per rater. |
| `abcde` | As `abc`; one accuracy shared globally. |
| `abce` | As `abc`; global but distinct sensitivity and specificity. |
| `abd` | As `ab`; one accuracy per rater. |
| `abde` | As `ab`; one accuracy shared globally. |
| `ac` | No `lambda` pathway or item location; discrimination varies. |
| `acd` | As `ac`; one accuracy per rater. |
| `ad` | No `lambda` pathway; one accuracy per rater; item location and discrimination vary. |
| `b` | Discrimination fixed; all other full-model effects remain. |
| `bc` | Discrimination and item location fixed. |
| `bcd` | As `bc`; one accuracy per rater. |
| `bd` | Discrimination fixed; one accuracy per rater; item location varies. |
| `c` | Item location fixed; all other full-model effects remain. |
| `cd` | Item location fixed; one accuracy per rater. |
| `d` | One accuracy per rater; all item-level effects remain. |

This construction isolates statistical comparisons: each derived model shows
only the assumption that changed, while the custom likelihood and posterior
predictive checks stay identical.

## StanBlocks features used by the port

- `@deffun` defines reusable Stan functions in Julia syntax. The two
  `increment_at` methods demonstrate both ordinary multiple dispatch and
  a higher-order method whose first argument is itself a function.
- Sized arguments such as `item::int[n]` make the generated Stan signature
  explicit. The vararg definition `args...` forwards the complete custom
  distribution interface without repeating it.
- `jbroadcasted` records element-wise calls when the element operation is
  a function argument, as in the likelihood and replicated-data comparisons.
- The `_lpmfs` helper returns one contribution per item;
  `@lpxf no_good_name_lpmf` registers the scalar aggregate used by
  `rating ~ no_good_name(...)`. Matching scalar and sized `_rng`
  methods provide generated-data simulation.
- `mock_data` supplies representative shapes and element types while the
  family is traced. Real observations replace those values when a model is
  instantiated.
- `Base.merge` performs the post-hoc model adjustment. A replacement can
  turn a sampled vector into a fixed vector, or replace two sampled vectors
  with shared parameters, without copying the full model body. StanBlocks then
  emits declarations from the dataflow of that derived model.


## Full Julia source and generated Stan code

The build evaluates the displayed source once and emits all 19 named model
variants from the resulting `posteriors` family.

```@raw html
<div class="atlas-comparison" data-atlas-comparison data-stan-label="Generated Stan models">
```

```@eval
Main.FeatureAtlasDocs.comparisons(@__MODULE__, raw"""
using StanBlocks

@deffun begin 
    increment_at(rv0, idxs::int[n], arg1) = begin 
        rv = rv0
        for i in 1:n
            idx = idxs[i]
            rv[idx] += arg1[idx]
        end
        rv
    end
    increment_at(f, rv0, idxs::int[n], arg1, arg2) = begin 
        rv = rv0
        for i in 1:n
            idx = idxs[i]
            rv[idx] += f(arg1[idx], arg2[idx])
        end
        rv
    end
    vote_count(rating, item, rater, I, J) = increment_at(
        rep_array(0, J + 1), increment_at(rep_array(1, I), item, rating), rep_array(1, I)
    )
    rater_count(rating, rater, J) = increment_at(rep_array(0, J), rater, rating) 
    lte_sim_rng(x, y) = if x == y
        bernoulli_rng(.5)
    else
        x < y
    end
    lte_sim_rng(x::anything[_], y) = jbroadcasted(lte_sim_rng, x, y)
    pos_probs(lambda, delta, alpha_sens, beta_, item, rater) = (
        lambda[item] + (1 - lambda[item]) * inv_logit(delta[item] * (alpha_sens[rater] - beta_[item]))
    )
    neg_probs(lambda, delta, alpha_spec, beta_, item, rater) = (
        (1 - lambda[item]) * inv_logit(-delta[item] * (alpha_spec[rater] - beta_[item]))
    )
    no_good_name_lpmfs(rating, I, item, rater, pi_, alpha_spec, alpha_sens, beta_, delta, lambda) = jbroadcasted(
        log_sum_exp, 
        increment_at(bernoulli_lpmf, rep_vector(log(pi_), I), item, rating, pos_probs(lambda, delta, alpha_sens, beta_, item, rater)),
        increment_at(bernoulli_lpmf, rep_vector(log1m(pi_), I), item, rating, neg_probs(lambda, delta, alpha_spec, beta_, item, rater))
    ) 
    @lpxf no_good_name_lpmf(args...) = sum(no_good_name_lpmfs(args...))
    no_good_name_rng(I, item::int[n], rater, pi_, alpha_spec, alpha_sens, beta_, delta, lambda)::int[n] = begin
        z_sim = to_vector(bernoulli_rng(rep_vector(pi_, I)))
        bernoulli_rng(
            z_sim[item] .* pos_probs(lambda, delta, alpha_sens, beta_, item, rater)
            + (1 - z_sim[item]) .* neg_probs(lambda, delta, alpha_spec, beta_, item, rater)
        )
    end
    no_good_name_rng(int[n], I, item::int[n], rater, pi_, alpha_spec, alpha_sens, beta_, delta, lambda)::int[n] =
        no_good_name_rng(I, item, rater, pi_, alpha_spec, alpha_sens, beta_, delta, lambda)
    StanBlocks.stan.log_sum_exp(::real, ::real)::real
end

mock_data = (;I=1,J=1,item=[1],rater=[1],rating=[1])

full = @slic mock_data begin 
    votes_data = vote_count(rating, item, rater, I, J)
    rater_data = rater_count(rating, rater, J)
    pi_ ~ beta(2, 2)
    alpha_spec ~ normal(2, 2; n=J)
    alpha_sens ~ normal(1, 2; n=J, lower=-alpha_spec)
    beta_ ~ normal(0, 1; n=I)
    delta ~ lognormal(0, 0.25; n=I)
    lambda ~ beta(2, 2; n=I)
    rating ~ no_good_name(I, item, rater, pi_, alpha_spec, alpha_sens, beta_, delta, lambda)

    rating_sim = no_good_name_rng(I, item, rater, pi_, alpha_spec, alpha_sens, beta_, delta, lambda)
    votes_sim = vote_count(rating_sim, item, rater, I, J)
    votes_sim_lt_data = lte_sim_rng(votes_sim, votes_data)
    rater_sim = rater_count(rating_sim, rater, J)
    rater_sim_lt_data = lte_sim_rng(rater_sim, rater_data)
end

a_transform = quote 
    lambda = rep_vector(0, I)
end
b_transform = quote 
    delta = rep_vector(1, I)
end
c_transform = quote 
    beta_ = rep_vector(0, I)
end
d_transform = quote 
    alpha_acc ~ normal(1, 2; n=J, lower=0)
    alpha_sens = alpha_acc
    alpha_spec = alpha_acc
end
de_transform = quote 
    alpha_acc_scalar ~ normal(1, 2; lower=0)
    alpha_sens = rep_vector(alpha_acc_scalar, J)
    alpha_spec = rep_vector(alpha_acc_scalar, J)
end
e_transform = quote 
    alpha_spec_scalar ~ normal(2, 2)
    alpha_sens_scalar ~ normal(1, 2; lower=-alpha_spec_scalar)
    alpha_spec = rep_vector(alpha_spec_scalar, J)
    alpha_sens = rep_vector(alpha_sens_scalar, J)
end

a = Base.merge(full, a_transform)
ab = Base.merge(a, b_transform)
abc = Base.merge(ab, c_transform)
abcd = Base.merge(abc, d_transform)
abcde = Base.merge(abc, de_transform)
abce = Base.merge(abc, e_transform)
abd = Base.merge(ab, d_transform)
abde = Base.merge(ab, de_transform)
ac = Base.merge(a, c_transform)
acd = Base.merge(ac, d_transform)
ad = Base.merge(a, d_transform)
b = Base.merge(full, b_transform)
bc = Base.merge(b, c_transform)
bcd = Base.merge(bc, d_transform)
bd = Base.merge(b, d_transform)
c = Base.merge(full, c_transform)
cd_ = Base.merge(c, d_transform)
d = Base.merge(full, d_transform)

posteriors = (;
    full,
    a, ab, abc, abcd, abcde, abce, abd, abde, ac, acd, ad,
    b, bc, bcd, bd,
    c, cd=cd_,
    d,
)
""", :posteriors)
```

```@raw html
</div>
```
