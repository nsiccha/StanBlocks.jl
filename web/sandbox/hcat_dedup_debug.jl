begin
    m = @slic (;ztime=randn(3), dose_times=[[0.0,1.0],[2.0,3.0,4.0]]) begin
        ftime ~ std_normal(; n=num_elements(ztime))
        X_log_err = hcat(rep_vector(1.0, num_elements(dose_times)))
        X_loc = hcat(ftime)
        y = X_loc
    end
    import StanBlocks.stan: sig_expr, head
    sm = StanBlocks.stan_model(m)
    b = sm.blocks[1]
    ks = [k for (k,_) in b.content if isa(head(k), Function) && nameof(head(k)) == :hcat]
    for k in ks
        println("  key = ", k)
        println("  sig = ", sig_expr(k))
    end
    println("hcat keys: ", length(ks))
end