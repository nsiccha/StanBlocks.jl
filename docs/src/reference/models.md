# Model APIs

The [model guide](../guides/models.md) introduces binding and block placement.
The [composition guide](../guides/composition.md) covers reusable submodels,
and the [functions guide](../guides/functions.md) covers deterministic functions
and probability families.

## Model construction

```@docs
@slic
@plate
@scan
compile_slic_bundle
```

## Function definitions

```@docs
@deffun
@defsig
@juliacompat
@stanonly
@stan_assert
return_type_of
slic_eval
```

## Probability families

```@docs
@lpxf
@lhs
ValueFamily
ValueUDF
family_support
```
