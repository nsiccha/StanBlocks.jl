# Golf putting

Estimate an angular error scale from putting counts, then compare replicated
success proportions with the observations. The example uses the public putting
data discussed in [the Stan case study](https://mc-stan.org/learn-stan/case-studies/golf.html),
which attributes the observations to Berry's study of professional golf. Each
row records distance in feet, the number of attempts and the number made. These
are aggregated counts, not trajectories of individual balls. The source file
contains the observations explicitly; no private records or external download
are required to reproduce the calculation.

```@eval
Main.DocsExamples.golf_data()
```

## Geometry

A putt succeeds when its direction falls within an angular tolerance set by the
distance and the clearance between ball and hole. The model treats angular
error as a centered normal variable. It assumes the ball travels far enough,
that the green introduces no additional directional variation, and that putts
at a given distance share one success probability. Those assumptions define
what the fitted error scale measures. They do not identify separate sources of
error such as player variation, slope or speed control.

The clearance is half the difference between hole and ball diameters, converted
to feet. Dividing that clearance by distance gives the sine of the limiting
angle. The probability that a centered normal angular error falls between the
negative and positive limit is therefore a difference of normal distribution
functions. The model evaluates that probability for every observed distance and
uses the attempt counts in a binomial likelihood.

Angular standard deviation is positive. Its prior is a normal distribution
restricted to that support; the standard deviation of the prior is expressed in
radians. The derived degree scale changes only the presentation. It does not
change the density or introduce another parameter. The following source is the
file executed by this build. Its generated Stan program is compiled with the
Stan compiler before being included on the page.

```@eval
Main.DocsExamples.model_block("golf")
```

Read the generated program as a check on the model specification. The bound
observations belong in the data block, the positive error scale belongs in the
parameters block, and the binomial observations contribute to the model block.
The angle calculation is shared across those contributions. The
[model guide](../guides/models.md) explains binding and generated blocks; the
[functions guide](../guides/functions.md) explains distribution functions.

## Inference

This model has one free scalar. Numerical quadrature is sufficient to compute
its posterior without a Markov chain. The generator evaluates the compiled
Stan density on a grid in the logarithm of the angular standard deviation.
Stan's unconstrained density includes the change-of-variables Jacobian, so
normalizing the grid weights approximates integration in that coordinate.
Omitting that Jacobian would define a different posterior.

The documentation build reads the saved summary and checks the hashes of the
model source, generator, generated Stan program and summary. It does not rerun
the fit. Changing any of those inputs invalidates the receipt and stops the
build. This makes the displayed results attributable to a particular model and
calculation while keeping ordinary documentation builds independent of fitting.

```@eval
Main.DocsExamples.golf_summary()
```

The interval describes uncertainty about the shared angular standard deviation
under this model and prior. It is not an interval for the angular error of a
particular putt. That distinction matters when interpreting predictions: a
replicated count depends both on uncertainty in the shared parameter and on
binomial variation at the observed number of attempts.

## Numerical checks

The generator compares the Stan log density with a separate calculation using
Julia's normal and binomial distributions. Terms that are constant in the
parameter may be omitted by Stan, so the comparison checks that the difference
is constant across parameter values. The reported range measures departures
from that constant, in log-density units. Near-unit success probabilities at
the smallest error scales amplify floating-point differences between normal
CDF implementations; the generator applies an explicit tolerance there.

```@eval
Main.DocsExamples.golf_diagnostics()
```

The refined grid checks the posterior median's sensitivity to grid spacing.
Boundary probability checks whether appreciable posterior weight accumulates
at either end of the integration interval. These checks address numerical
integration of the stated model. They do not establish that its geometric
assumptions describe putting behavior, and they do not substitute for checking
the observations against predictions. There are no convergence statistics for
chains because this calculation uses no chains.

## Predictions

At each observed distance, the calculation transforms every weighted parameter
value into a success probability. It then integrates the binomial distribution
at the corresponding number of attempts. The resulting intervals describe
replicate success proportions, with the same attempt counts as the original
data. Using the observed counts preserves the differing sampling precision
across distances.

```@eval
Main.DocsExamples.figure("golf")
```

```@raw html
<ScientificFigure src="/figures/golf.html" title="golf model result" />
```

Compare the orange observations with the blue intervals and the shape of the
blue probability curve. A persistent pattern of discrepancies across adjacent
distances would suggest that the angle-only assumption misses structure. An
isolated discrepancy has a different interpretation because these intervals
are marginal for each distance, not a simultaneous band for the whole curve.
The observed data were used to estimate the angular scale, so this is a
posterior predictive comparison rather than an evaluation on held-out putts.

The model can be extended with a mechanism for distance control, but that would
require another likelihood calculation and another checked result. This page
keeps the inference and predictive comparison attached to the single model
shown above. The [prediction guide](../guides/prediction.md) explains how
conditioning and replay change the role of observations when constructing such
extensions.

```@raw html
<details><summary>Reproduce</summary>
```

```@eval
Main.DocsExamples.reproduce("golf")
```

```@raw html
</details>
```
