# Epidemic dynamics

Solve a susceptible–infectious–removed system and distinguish infectious
prevalence from incident infections. The example is a synthetic prior-predictive
scenario, inspired by the compartment structure in the
[Stan boarding-school case study](https://mc-stan.org/learn-stan/case-studies/boarding_school_case_study.html).
It does not use the case study's observed time series or estimate transmission
parameters. Initial compartment counts, population size, output times and rate-prior
parameters are explicit inputs in the source below. Time is measured in days,
compartment sizes in people, and both rate constants in inverse days.

## Differential equations

The susceptible compartment loses people through infection. That flow is
proportional to the susceptible and infectious counts divided by the total
population. The infectious compartment receives the same flow and loses
people through removal. The removed compartment receives those removals. This
closed system has no births, migration or return from removal to susceptibility.
Adding the three derivatives therefore gives zero.

The right-hand side is a typed deterministic function defined with `@deffun`.
Its signature identifies time, the state vector and shared rate inputs. The
function returns a derivative vector with one entry per compartment. The model
passes that function and the initial state to Stan's Runge–Kutta solver, then
returns the trajectory at the requested output times.

```@eval
Main.DocsExamples.model_block("epidemic")
```

The generated Stan program exposes both the function definition and the solver
call. Reading them together checks that the same population and rates reach
every evaluation of the derivative. The [functions guide](../guides/functions.md)
explains typed function definitions. The [execution guide](../guides/execution.md)
shows how the compiled model is evaluated through BridgeStan.

The model gives both rates lognormal priors and has no observations or
likelihood. StanBlocks therefore draws the rates in generated quantities and
evaluates the trajectory conditional on those draws. The build uses a fixed
seed and displays one prior-predictive scenario. It is useful for checking units,
state ordering and observation definitions before adding a likelihood. One
scenario does not summarize the uncertainty induced by the prior distributions.

## Run

The build evaluates the compiled model with an explicit RNG seed and extracts
the rates and trajectory using the model descriptor and BridgeStan output names.
The requested output times determine the points displayed on this page. They
do not force the numerical solver to use those times as its internal integration
steps; the solver chooses its own steps to meet its numerical tolerances.

```@eval
Main.DocsExamples.epidemic_summary()
```

The reproduction number reported here is the drawn infection-rate constant divided
by the removal-rate constant. That ratio describes invasion into a fully
susceptible population under this system's assumptions. The effective ratio
later in the scenario also depends on the susceptible fraction. The mean
infectious period is the reciprocal removal rate, corresponding to the
constant individual removal hazard implied by the compartment equations.

The largest sampled prevalence is a property of the evaluated output grid.
Its reported time is therefore the time of the largest displayed value, not
an independently optimized continuous-time peak. Using a finer output grid
could change that reported time without changing the differential equations.
The table keeps those quantities separate from the drawn rates so that a
numerical summary is not presented as an estimated biological parameter.

## Diagnostics

Conservation of population provides a direct check on the solver result and
on how its output is reshaped. At every output time, the sum of the susceptible,
infectious and removed compartments should equal the initial population up to
numerical error. The model also implies nonnegative compartment sizes and a
susceptible count that does not increase.

```@eval
Main.DocsExamples.epidemic_diagnostics()
```

These checks would detect several common implementation mistakes, including a
sign error in a flow, swapped state indices or a reshaping error in the output.
The conservation residual is expressed in people. It is an invariant check,
not an estimate of the solver's total trajectory error: a numerically
inaccurate trajectory can still conserve population. For an application that
requires tighter numerical accuracy, vary solver tolerances and compare the
result at the same output times.

There are no chain diagnostics because this example draws directly from its
priors. There is also no comparison with observed infections, so it offers no
empirical estimate of goodness of fit. These limitations are part of the model
shown, rather than missing results to be supplied by interpreting its smooth
trajectory as evidence.

## Observation definitions

Prevalence is the number of people currently infectious at a particular time.
Incidence counts transitions into infection over an interval. In this closed
system, interval incidence equals the decrease in susceptible count between
the interval's endpoints. The build calculates that difference directly from
the trajectory, including the initial state for the first interval.

```@eval
Main.DocsExamples.figure("epidemic")
```

```@raw html
<ScientificFigure src="/figures/epidemic.html" title="epidemic model result" />
```

The curves refer to different quantities even though both are expressed as
numbers of people. The blue value is a count at an instant. The orange value
is a count accumulated during the preceding output interval. Changing the
interval length changes the scale of the incidence curve; it does not change
what infectious prevalence means. A rate of new infections would instead
divide the interval count by its duration and would have units of people per
day.

Choosing a likelihood requires matching its expected value to the recorded
measurement. Daily newly reported cases do not directly observe the infectious
compartment. Counts of people infectious at a survey time do not directly
observe the decrease in susceptibility over a day. Reporting delays,
ascertainment and overdispersion may require additional assumptions even after
the appropriate state or flow has been selected.

A statistical extension can attach an observation distribution to the relevant
trajectory quantity and bind observed outcomes. The deterministic function
remains reusable, while the rates become inference parameters through their
likelihood contributions. The [composition guide](../guides/composition.md) explains how
to separate reusable components, and the [prediction guide](../guides/prediction.md)
explains how observed outcomes become replicated outcomes. Any such extension
needs its own data provenance, numerical checks and predictive assessment.

```@raw html
<details><summary>Reproduce</summary>
```

```@eval
Main.DocsExamples.reproduce("epidemic")
```

```@raw html
</details>
```
