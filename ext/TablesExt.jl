module TablesExt

# Table (Tables.jl source, incl. `DataFrame`) → `@slic` data kwarg.
#
# A table is special precisely because all of its columns share ONE length (the
# row count). So a table data kwarg ingests as a single `ntup` whose fields are
# the columns, ALL sharing one row-count size `<name>_nrow` — never a bag of
# columns with independent per-column sizes. The columns are addressed by name in
# the model body (`df.age`), reusing the existing `ntup` field-access machinery
# (`forward!(::GetPropertyExpr)`); float columns become `vector[<name>_nrow]`,
# integer columns `array[<name>_nrow] int`.
#
# Registered into the core's `_FOREIGN_DATA_INGESTERS` hook at load, so it is a
# weakdep: a table type already pulls in Tables, so this activates automatically,
# and users who never touch tables pay no dependency.

import Tables
import StanBlocks
import StanBlocks: StanType, stan_expr, types

# Column → Stan center type, all sharing the caller's single `nrow` size expr.
_column_stan_type(name, col::AbstractVector{<:AbstractFloat}, nrow) =
    StanType(types.vector, (nrow,); value=col)
_column_stan_type(name, col::AbstractVector{<:Integer}, nrow) =
    StanType(types.int, (nrow,); value=col, qual=:data)
_column_stan_type(name, col, nrow) = error(
    "StanBlocks Tables ingest: column `$name` has eltype $(eltype(col)), which is " *
    "neither a real nor an integer vector. Convert it to a numeric column, or drop " *
    "it and pass the numeric columns.")

# Ingest a Tables.jl source as an ntup of equal-length columns.
function _table_stan_type(expr, tbl; kwargs...)
    cols = Tables.columntable(tbl)   # NamedTuple of columns, equal length by Tables contract
    isempty(cols) && error(
        "StanBlocks Tables ingest: table `$expr` has no columns; nothing to ingest.")
    nrow = length(first(cols))
    # ONE shared row-count data int (`<expr>_nrow`), referenced by every column.
    nrow_expr = stan_expr(Symbol(expr, "_nrow"), nrow)
    arg_types = (; (colname => _column_stan_type(Symbol(expr, "_", colname), col, nrow_expr)
                    for (colname, col) in pairs(cols))...)
    StanType(types.ntup, tuple(); arg_types, value=cols, kwargs...)
end

# The registered hook: accept only genuine Tables.jl sources; decline everything
# else (returns `nothing`, so the core falls through to its usual error).
_ingest_table(expr, value; kwargs...) =
    Tables.istable(value) ? _table_stan_type(expr, value; kwargs...) : nothing

function __init__()
    push!(StanBlocks._FOREIGN_DATA_INGESTERS, _ingest_table)
end

end # module TablesExt
