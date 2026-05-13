sb_bad = StanBlocks.@slic begin
    param_arg ~ std_normal(; n=5)
    return param_arg
end
@slic begin
    theta ~ std_normal(; n=5)
    sub ~ sb_bad(; param_arg=theta)
    return sub
end