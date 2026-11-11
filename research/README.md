# Documentation calculations

| Directory | Input | Calculation |
|---|---|---|
| [golf](golf/README.md) | Public aggregated putting counts, source linked in the directory | Cached one-parameter quadrature and posterior predictive summaries |
| [pooling](pooling/README.md) | Public eight-schools effect estimates, source linked in the directory | Build-time conditional normal moments and density checks |
| [epidemic](epidemic/README.md) | Explicit synthetic scenario | Build-time seeded prior-predictive SIR trajectory and conservation checks |

Each model was written for the documentation. The site links these files from
its Reproduce footers. Figures are generated during the build and are not
committed. The golf cache contains summaries and a receipt, without draws.
