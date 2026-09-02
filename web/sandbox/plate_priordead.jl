# Regression (todo fspmjv; snag prior-predictive-7e463983): a FULLY prior-only
# plate — no likelihood anywhere, output unused — is a prior-predictive program.
# The whole plate (fresh per-cell sample, collected result, compiler-owned loop)
# lowers to generated quantities together with `tau`, so `parameters {}` and
# `model {}` are empty and the program is a `fixed_param` simulator. Two past
# states: the sample was GQ-lowered while the return fill stayed in transformed
# parameters ("w not in scope"); then aef6a42 pinned the sample as a parameter
# (a prior sampled with NUTS). The fill now follows its sources to GQ.
@slic (;) begin
    tau ~ normal(0.0, 1.0; lower = 0.0)
    z ~ plate(; outer = (4,)) do i
        w ~ normal(0.0, tau)      # fresh per-cell param — re-drawn in generated quantities
        w                         # cell output → z[i] (generated quantities)
    end
end
