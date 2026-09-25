# API Reference

## Model Definition

```@docs
StanBlocks.@slic
StanBlocks.@deffun
StanBlocks.@juliacompat
StanBlocks.@stanonly
StanBlocks.@defsig
StanBlocks.@usertype
```

## Sampling-form Dispatch

```@docs
StanBlocks.@lpxf
StanBlocks.@lhs
```

## Runtime Assertions

```@docs
StanBlocks.@stan_assert
```

## Model Inspection and Compilation

```@docs
StanBlocks.return_type_of
StanBlocks.compile_slic_bundle
StanBlocks.stan_code
StanBlocks.stan_model
StanBlocks.stan_instantiate
StanBlocks.instantiate
```

## Model Descriptors

```@docs
StanBlocks.stan_descriptor
StanBlocks.required_inputs
StanBlocks.stan_definition
StanBlocks.stan_definition_closure
StanBlocks.stan_operation
StanBlocks.stan_execute
StanBlocks.ModelDescriptor
StanBlocks.ModelInput
StanBlocks.ModelOutput
StanBlocks.ModelDefinition
StanBlocks.ModelOperation
```

## Smoke Tests

```@docs
StanBlocks.transpiles
StanBlocks.compiles
StanBlocks.stanc_check
```

## Types

```@docs
StanBlocks.SlicModel
StanBlocks.StanModel
```

## Errors

```@docs
StanBlocks.StanBlocksError
StanBlocks.StanBlocksDiagnostic
StanBlocks.diagnostic
```
