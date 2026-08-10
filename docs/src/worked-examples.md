# Worked examples

These are the former Quarto notebooks, migrated page-for-page into the main
documentation. Runnable model examples are evaluated during the documentation
build: the Julia source shown on the page is the source that produced the
complete Stan program shown beside or below it.

## Model families

- [Golf models](examples/golf-models.md) reproduces five progressively richer
  putting models.
- [PCR sensitivity versus time](examples/isba-2024.md) contains the 2×5 model
  matrix presented at ISBA 2024.
- [Crowdsourced ratings](examples/crowdsource.md) builds the full model and its
  18 post-hoc variants.
- [Reusable constraints](examples/constraints.md) covers the disk constraint
  and ten simplex transforms.

## Case studies

- [Golf putting](examples/case-studies/golf.md)
- [Motorcycle data](examples/case-studies/motorcycle.md)
- [Multilevel radon regression](examples/case-studies/radon.md)
- [Planetary motion](examples/case-studies/planets.md)
- [Disease transmission](examples/case-studies/school.md)
- [Multiple species-site occupancy](examples/case-studies/species.md)
- [Soil carbon](examples/case-studies/soil.md)

## Design and historical material

- [The original `@slic` design overview](examples/slic-overview.md)
- [Monster-model notebook](examples/monster.md)
- [Simplex transform experiments](examples/simplex-transforms.md)
- [PosteriorDB implementations](examples/posteriordb-implementations.md)

The historical notebooks are retained as historical source when they contain
unfinished experiments, proposed syntax, host-specific benchmark setup, or
other material that cannot honestly be presented as a current executable
example. Their pages say so explicitly rather than silently dropping them.
