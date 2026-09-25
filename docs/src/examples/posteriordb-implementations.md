# Julia posteriordb implementations

StanBlocks keeps its PosteriorDB implementations in the optional
[`PosteriorDBExt` extension](https://github.com/nsiccha/StanBlocks.jl/blob/devibe/ext/PosteriorDBExt.jl).
Loading `PosteriorDB` activates that extension and adds
`slic_implementation(::Val{:model_name}; data...)` methods for the supported
catalogue entries.

## Why this page links to the extension

The legacy Quarto page used an include directive to paste the entire extension
source into the rendered notebook. That produced a second, enormous copy of a
maintained source file without turning it into a worked example. In the main
docs, the extension itself is the canonical, line-addressable implementation;
copying it here would drift immediately.

The implementations demonstrate a consistent set of StanBlocks mechanisms:

- per-model `Val` dispatch selects an implementation without a stringly typed
  runtime switch;
- `@deffun` supplies catalogue-specific recurrences and likelihood helpers;
- `@lhs`/`@lpxf` register custom distribution families for ordinary `~`
  statements and generated likelihood/prediction companions;
- `@slic` attaches PosteriorDB data and builds each model from the same public
  authoring surface used elsewhere in these examples.

## Maintained model inventory

The extension currently defines 77 named implementations. Related names are
variants of one statistical example rather than unrelated snippets:

- **Generic GLM, rate, and introductory models:** `GLM_Binomial_model`,
  `GLM_Poisson_model`, `GLMM_Poisson_model`, `Rate_1_model`, `Rate_2_model`,
  `Rate_3_model`, `Rate_4_model`, `Rate_5_model`, `blr`,
  `logistic_regression_rhs`, `pilots`, `surgical_model`, `diamonds`,
  `dugongs_model`, `kilpisjarvi`, and `accel_splines`. These cover binomial
  and Poisson regression, random effects, nonlinear growth, and spline bases.
- **Canonical hierarchical and applied examples:**
  `eight_schools_centered`, `eight_schools_noncentered`, `seeds_model`,
  `seeds_centered_model`, `seeds_stanified_model`, `sesame_one_pred_a`,
  `election88_full`, `nes`, `nes_logit_model`, and `bym2_offset_only`.
  The paired names expose centered/non-centred parameterisations or alternative
  spellings of the same likelihood.
- **Child test-score regressions:** `kidscore_momhs`, `kidscore_momiq`,
  `kidscore_momhsiq`, `kidscore_mom_work`, `kidscore_interaction`,
  `kidscore_interaction_c`, `kidscore_interaction_c2`, and
  `kidscore_interaction_z`. The suffixes add predictors, interactions,
  centering, or standardisation while preserving the same outcome.
- **Earnings and height regressions:** `earn_height`, `log10earn_height`,
  `log10earn_height_male`, `logearn_height`, `logearn_height_male`,
  `logearn_interaction`, `logearn_interaction_z`, and
  `logearn_logheight_male`. These make the response/predictor transforms and
  sex interaction explicit in the implementation name.
- **Mesquite regressions:** `mesquite`, `logmesquite`, `logmesquite_logvolume`,
  `logmesquite_logva`, `logmesquite_logvas`, and `logmesquite_logvash`.
  Successive variants change the response scale and add transformed canopy,
  shrub, and height predictors.
- **Well-switching regressions:** `wells_dist`, `wells_dist100_model`,
  `wells_dist100ars_model`, `wells_dae_model`, `wells_dae_c_model`,
  `wells_dae_inter_model`, `wells_daae_c_model`, `wells_interaction_model`, and
  `wells_interaction_c_model`. The family varies distance scaling, arsenic,
  education, interactions, and centered predictors.
- **Radon multilevel family:** `radon_pooled`, `radon_county`,
  `radon_county_intercept`, `radon_partially_pooled_centered`,
  `radon_partially_pooled_noncentered`,
  `radon_hierarchical_intercept_centered`,
  `radon_hierarchical_intercept_noncentered`,
  `radon_variable_intercept_centered`,
  `radon_variable_intercept_noncentered`, `radon_variable_slope_centered`,
  `radon_variable_slope_noncentered`,
  `radon_variable_intercept_slope_centered`, and
  `radon_variable_intercept_slope_noncentered`. Together they cover the
  pooling ladder and the centered/non-centred hierarchy choices.
- **Time-series, Gaussian-process, mixture, and spatial specialists:**
  `arma11`, `garch11`, `gp_regr`, `gp_pois_regr`, `normal_mixture`,
  `low_dim_gauss_mix`, and `low_dim_gauss_mix_collapse`. These are where the
  extension-local recurrence, custom-density, GP, and mixture helpers are most
  visible.

This inventory is intentionally explanatory rather than a pasted source dump.
For example, every `_centered`/`_noncentered` pair is a parameterisation
comparison, while the radon, wells, kid-score, earnings, and mesquite suffixes
record the covariate/model-building progression. The full method bodies at the
source link remain the executable authority.

For small, side-by-side executable examples of those mechanisms, use the
[feature atlas](../feature-atlas.md). For the complete catalogue mapping, read
the extension source above; it is what the package loads and tests.
