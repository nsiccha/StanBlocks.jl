# Execution APIs

The [execution guide](../guides/execution.md) connects source generation,
BridgeStan compilation and descriptor operations. The
[prediction guide](../guides/prediction.md) explains the generative roles of
observed and unobserved quantities.

## Source and compilation

```@docs
stan_model
stan_code
stanc_check
stan_instantiate
stan_library
StanBlocks.release_tape_memory
StanBlocks.reuse_tape_memory
```

## Inspection and operations

```@docs
stan_descriptor
stan_definition
stan_definition_closure
stan_operation
stan_execute
stan_interface
stan_data_json
```

## Descriptor types

```@docs
ModelDescriptor
ModelInput
ModelOutput
ModelDefinition
ModelOperation
```

## Diagnostics

```@docs
StanBlocksError
StanBlocksDiagnostic
diagnostic
```
