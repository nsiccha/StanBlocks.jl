nan_on_error(f, x) = try
    f(x)
catch e
    NaN 
end
nan_on_error(f) = Base.Fix1(nan_on_error, f)
check_implementation(reference, test; stat_f=median, m=200, kwargs...) = begin 
    n = dimension(test)
    X = randn((n, m))
    reference_lpdfs = mapreduce(nan_on_error(reference), vcat, eachcol(X)) 
    test_lpdfs = mapreduce(nan_on_error(test), vcat, eachcol(X)) 
    finite_idxs = filter(i->isfinite(reference_lpdfs[i]+test_lpdfs[i]), 1:m)
    length(finite_idxs) == 0 && return 
    test_lpdfs = test_lpdfs[finite_idxs]
    reference_lpdfs = reference_lpdfs[finite_idxs]
    adjusted_lpdfs = test_lpdfs .+ median(reference_lpdfs-test_lpdfs) 
    return (;
        absolute_constant_difference=stat_f(abs.(reference_lpdfs .- test_lpdfs)),
        relative_remaining_difference=stat_f(abs.((reference_lpdfs .- adjusted_lpdfs)./reference_lpdfs))
    )
end