# Golf putting

`model.jl` contains the complete input and model. The counts are the first
Berry (1996) dataset in the Stan case study
[Model building and expansion for golf putting](https://mc-stan.org/learn-stan/case-studies/golf.html),
also preserved in this [revision-pinned data file](https://gist.githubusercontent.com/nsiccha/553d6ce6a784142e87fbfd7aaf4c5e99/raw/668b86ddd84b45b707e594dd313f717bef1a553d/dataset1.txt).
Distances and the clearance between the cup and ball radii are in feet.

Run `julia --project=docs research/golf/fit.jl` from the repository root to
regenerate `summary.toml` and `receipt.toml`. The calculation integrates one
log-scale parameter on a deterministic grid, checks a refined grid, and compares
Stan's target with a separately evaluated normalized likelihood up to an
additive constant. No sampler is used and no draws are stored.

The receipt hashes the model, generator, emitted Stan and summary. The docs
build checks these hashes and uses only the summary; it does not refit. The
rendered figure is produced during the build and is never committed.
