@slic (;ztime=randn(3), dose_times=[[0.0,1.0],[2.0,3.0,4.0]]) begin
    ftime ~ std_normal(; n=num_elements(ztime))
    X_log_err = hcat(rep_vector(1.0, num_elements(dose_times)))
    X_loc = hcat(ftime)
    y = X_loc
end