# StanBlocks.jl

StanBlocks turns probabilistic models written in Julia syntax into Stan programs that run through BridgeStan.

```sh
julia -e 'using Pkg; Pkg.add(url="https://github.com/nsiccha/StanBlocks.jl", rev="devibe")'
```

[Start with an executed model](https://nsiccha.github.io/StanBlocks.jl/dev/).
The documentation covers [model construction](https://nsiccha.github.io/StanBlocks.jl/dev/guides/models),
[composition](https://nsiccha.github.io/StanBlocks.jl/dev/guides/composition)
and [compilation and execution](https://nsiccha.github.io/StanBlocks.jl/dev/guides/execution).
Worked examples use public or synthetic data and show the model, generated
Stan, numerical checks and a figure.

[Documentation build instructions](docs/README.md) describe the page structure
and fit-cache policy. Reproduction scripts and data provenance live under
[research/](research/README.md).
