@slic (;ztime=randn(3), assay_idx=[1,2,1,2,3,1]) begin
    raw ~ std_normal(; n=num_elements(ztime))
    ftime = raw * 2.0
    X_loc_loc = hcat(rep_vector(1.0, num_elements(ztime)), ftime)
    Z_loc_loc = hcat(rep_vector(1.0, num_elements(assay_idx)), ftime)
    y = X_loc_loc
end