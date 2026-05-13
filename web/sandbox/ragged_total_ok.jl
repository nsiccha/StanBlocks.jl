@slic (;dose_times=[[0.0, 1.0], [2.0, 3.0, 4.0]]) begin
    x ~ std_normal(; n=ragged_total(dose_times))
end