# Partial pooling

Construct a repeated submodel and check its conditional posterior against a
closed-form calculation. The data are the public eight-schools treatment-effect
estimates used in hierarchical-model examples: one estimated effect and its
reported standard error for each school. Effects and standard errors are in
score points. The source file contains those published aggregates explicitly.
For a source implementation, see the
[RStan getting-started example](https://github.com/stan-dev/rstan/wiki/RStan-Getting-Started#example-1-eight-schools).
The calculation here fixes the population mean and standard deviation as
scenario inputs. It does not estimate their posterior distribution.

## Model

Each school has a latent treatment effect. The sampling model treats the
reported estimate as normal around that effect, with the reported standard
error known. The population model is also normal. Fixing its mean and standard
deviation lets us isolate the effect of measurement uncertainty on shrinkage
and validate the compiled likelihood without a sampling algorithm.

The inner model describes one school and returns its latent effect. The outer
model applies that submodel across aligned estimates and standard errors with
`plate`. Each invocation receives its own observed estimate and uncertainty;
the population inputs are shared. The returned effects form the outer model's
parameter array. The generated Stan program makes this repeated structure
visible, including the data, parameter dimensions and density contributions.

```@eval
Main.DocsExamples.model_block("pooling")
```

A school with an imprecise estimate contributes less information about its
latent effect. Its conditional posterior is consequently influenced more by
the fixed population distribution. This is a consequence of the probability
model, not a separate adjustment applied to fitted values. The
[composition guide](../guides/composition.md) explains submodel inputs and
returns; the [array guide](../guides/arrays.md) explains repeated models and
their parameter representation.

The assumptions are substantive. Reported standard errors are treated as
known, school estimates are conditionally independent, and the population
scale represents variation in true effects rather than measurement noise.
Normal sampling distributions can be reasonable for estimated effects even
when individual outcomes are not normally distributed. Their adequacy depends
on how those estimates and standard errors were obtained.

## Conditional calculation

For each school, posterior precision is the sum of observation precision and
population precision. Posterior variance is its reciprocal. The conditional
mean is that variance multiplied by the sum of the observation and population
means weighted by their respective precisions. With the fixed population mean
used here, it can also be read as the observed estimate multiplied by its data
weight. The remaining weight belongs to the population mean.

The build evaluates these expressions from the data in the model file. It also
compiles the Stan model and compares its log density with an independent sum
of normal log densities. The table therefore shows a calculation associated
with the executed model rather than hand-entered illustrative results.

```@eval
Main.DocsExamples.pooling_summary()
```

Read across a row to connect observation uncertainty with the conditional
estimate. The data weight is dimensionless. The observed effect, its standard
error, the conditional mean and conditional standard deviation all retain
score-point units. Comparing weights across rows shows why observations with
similar effects can undergo different amounts of shrinkage.

These conditional distributions are independent across schools because both
population parameters are fixed. If the population parameters were inferred
jointly, their uncertainty would couple the school effects. A full hierarchical
analysis would average over that uncertainty, so its intervals generally would
not equal the conditional intervals on this page. Fixing a population scale
for demonstration should not be mistaken for estimating that scale from the
school data.

## Checks

The density comparison evaluates distinct parameter vectors, including the
conditional mean and a displacement scaled by the conditional standard
deviations. Stan may discard constants that do not depend on parameters.
Accordingly, the check compares changes in log density, allowing one shared
additive constant. A mismatch would expose a problem in the repeated model,
its observation binding or the analytic reference calculation.

```@eval
Main.DocsExamples.pooling_diagnostics()
```

The weight and uncertainty checks are consequences of combining two normal
information sources with positive, finite variances. They help detect an
inverted precision, a mismatched data column or another mistake in the
calculation. They are algebraic and implementation checks. They do not assess
the empirical suitability of a normal population distribution for these
schools.

No chain diagnostics are appropriate here: no chain is run. The exact
conditional formulas remove Monte Carlo error, but they do not remove model
uncertainty or uncertainty about the inputs that were held fixed. The build
receipt in the reproduction footer identifies the source used for this
calculation. No fitted draws or numerical summaries need to be committed.

## Shrinkage

The figure places the observed estimates beside the conditional posterior
means and their central intervals. Every interval uses the same posterior
probability level, computed from the normal quantile function during the
build. The common point scale lets the displacement toward the population mean
be compared across schools.

```@eval
Main.DocsExamples.figure("pooling")
```

```@raw html
<ScientificFigure src="/figures/pooling.html" title="pooling model result" />
```

The orange markers represent noisy estimates, while the blue markers represent
conditional means for latent effects. Their displacement is expected under
the model and is not evidence that the original estimates were biased. The
blue intervals quantify uncertainty about latent effects conditional on the
population inputs. They are not prediction intervals for a new reported
estimate; a future estimate would also include its sampling uncertainty.

A new school can be represented by the same submodel with its own estimate
and standard error. A school without an observed estimate instead requires a
prediction view that leaves the corresponding outcome unconditioned. The
[prediction guide](../guides/prediction.md) describes that distinction and why
marking an entire input is different from withholding selected observations.
Changing the population inputs is also a useful sensitivity calculation, but
each scenario must be interpreted conditionally rather than pooled into an
unlabeled set of fitted results.

```@raw html
<details><summary>Reproduce</summary>
```

```@eval
Main.DocsExamples.reproduce("pooling")
```

```@raw html
</details>
```
