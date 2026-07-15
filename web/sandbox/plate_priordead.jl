# Regression (todo fspmjv): a FULLY prior-only plate — no likelihood anywhere,
# output unused — must keep its fresh per-cell samples as PARAMETERS (not
# prior-only-lowered to generated quantities), so the transformed-parameters
# return-fill stays in scope. Previously emitted invalid Stan ("w not in scope").
@slic (;) begin
    tau ~ normal(0.0, 1.0; lower = 0.0)
    z ~ plate(; outer = (4,)) do i
        w ~ normal(0.0, tau)      # fresh per-cell param — stays a parameter
        w                         # cell output → z[i] (transformed parameters)
    end
end
