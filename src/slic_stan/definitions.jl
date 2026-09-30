# Definitions modify Julia method tables; tracing only reads them. Writers are
# exclusive, while independent traces share the read side. Task ownership (not
# thread ids) makes both sides safe across task migration and nested tracing.
mutable struct SlicDefinitionGate
    condition::Threads.Condition
    readers::IdDict{Task,Int}
    writer::Union{Nothing,Task}
    depth::Int
    waiting::Int
end
const _slic_definition_gate = SlicDefinitionGate(
    Threads.Condition(), IdDict{Task,Int}(), nothing, 0, 0)

function _with_slic_definitions(f)
    gate, task = _slic_definition_gate, current_task()
    lock(gate.condition) do
        if gate.writer === task
            gate.depth += 1
        else
            haskey(gate.readers, task) && throw(ArgumentError(
                "SLIC extension hooks must not register definitions during tracing"))
            gate.waiting += 1
            try
                while gate.writer !== nothing || !isempty(gate.readers)
                    wait(gate.condition)
                end
                gate.writer = task
                gate.depth = 1
            finally
                gate.waiting -= 1
                notify(gate.condition; all=true)
            end
        end
    end
    try
        f()
    finally
        lock(gate.condition) do
            gate.depth -= 1
            if gate.depth == 0
                gate.writer = nothing
                notify(gate.condition; all=true)
            end
        end
    end
end

function _with_slic_read(f)
    gate, task = _slic_definition_gate, current_task()
    lock(gate.condition) do
        # A writer may trace its completed definitions before publishing them.
        # Nested readers must pass a waiting writer to avoid self-deadlock.
        while gate.writer !== task && (gate.writer !== nothing ||
                (gate.waiting > 0 && !haskey(gate.readers, task)))
            wait(gate.condition)
        end
        gate.readers[task] = get(gate.readers, task, 0) + 1
    end
    try
        # The caller may be a task created before the definitions existed.
        Base.invokelatest(f)
    finally
        lock(gate.condition) do
            depth = gate.readers[task] - 1
            if depth == 0
                delete!(gate.readers, task)
                notify(gate.condition; all=true)
            else
                gate.readers[task] = depth
            end
        end
    end
end

"""
    slic_eval(mod::Module, expression)

Evaluate trusted SLIC definitions in `mod` as one exclusive publication phase.
Use a fresh module and put a complete runtime-generated family (all `@deffun`
companions and extension methods such as `autokwargs`) in one expression before
sharing its bindings. `compile_slic_bundle` already uses this protocol.

Individual SLIC definition macros synchronize their generated methods. This
outer boundary also covers macro expansion and families spanning several macros
or ordinary Julia extension methods. `stan_model` observes completed definitions
in the newest method world, including from previously created tasks; independent
traces remain parallel. Direct Julia calls to newly defined functions still need
Julia's usual `Base.invokelatest` boundary.

This is not a sandbox or a rollback facility: discard a fresh module if evaluation
fails. Do not spawn and wait for other definition/tracing tasks inside this
boundary. Published inputs must stay read-only and tracing hooks must be pure;
uncoordinated `Core.eval` of ordinary extension methods is outside this protocol.
"""
slic_eval(mod::Module, expression) = _with_slic_definitions() do
    Core.eval(mod, expression)
end

# Preserve call-site symbol resolution when evaluating generated top-level code.
_slic_definition_expr(mod, expression) =
    :($slic_eval($(QuoteNode(mod)), $(QuoteNode(expression))))
