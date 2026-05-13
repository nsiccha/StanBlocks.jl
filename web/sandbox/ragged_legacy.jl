# Sanity probe for the legacy `ragged_n` / `ragged_start` / `ragged_end`
# accessors (which take `::ntup`) — verifies that my `<:` → bare-`typeof(f)`
# change in `@deffun` didn't break the existing path.
@slic (;mem=randn(8), ends=[3, 5, 8]) begin
    rv  = (;mem, ends)
    n   = ragged_n(rv)
    s1  = ragged_start(rv, 1)
    e1  = ragged_end(rv, 1)
end
