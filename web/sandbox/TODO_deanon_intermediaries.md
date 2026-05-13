The deanon_type flow in stan_expr (slic.jl) currently only replaces _argN symbols
back to real args. This could be extended to also replace intermediate size expressions
(e.g. sizes derived from function body computations) back to expressions of the direct
function arguments. This would allow @defsig return types to depend on computed
intermediaries, not just direct arg dimensions.
