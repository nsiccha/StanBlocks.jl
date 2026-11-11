# Documentation

The site is written for readers who know Stan or hierarchical models. Each
contract has one guide; examples link to it. The following list is the complete
reader path, in navigation order.

| Part | Page | Purpose |
|---|---|---|
| Start | `index.md` | Install StanBlocks and inspect a complete model from explicit input. |
| Guides | `guides/models.md` | Define, bind and inspect a probabilistic model. |
| Guides | `guides/composition.md` | Reuse a submodel and replace one modeling assumption. |
| Guides | `guides/arrays.md` | Choose vector, matrix, ragged, plate and scan representations. |
| Guides | `guides/functions.md` | Define deterministic Stan functions and custom probability families. |
| Guides | `guides/prediction.md` | Distinguish conditioning, prior prediction and replay on new inputs. |
| Guides | `guides/execution.md` | Compile with BridgeStan and use descriptors to execute named operations. |
| Examples | `examples/golf.md` | Estimate angular putting error and compare observed success rates with predictions. |
| Examples | `examples/pooling.md` | Compute conditional normal shrinkage and show its dependence on measurement uncertainty. |
| Examples | `examples/epidemic.md` | Solve an SIR system and distinguish prevalence from incident infections. |
| Reference | `reference/models.md` | Look up exported model-definition and function APIs. |
| Reference | `reference/execution.md` | Look up exported inspection, execution, descriptor and error APIs. |

Build with `julia --project=docs docs/make.jl`. The build executes examples,
checks exported docstrings, runs the documentation lint, and generates figure
assets under `docs/build/`. Fits are not run by the documentation build. The
golf example reads a content-keyed summary and receipt from `research/golf/`;
its generator and regeneration instructions live there.

The previous feature atlas, gallery, model catalogs, historical proposals,
duplicated golf case and source-port inventories are removed from navigation
and from `docs/src`. Scientific cases without a checked result are deferred;
their source remains in Git history. The standalone StanCon presentation is
independent of the documentation build.
