infers(p; re=true) = try
    stan_model(p)
    return true
catch e
    re && rethrow()
    return false
end
full_cqual_eq(p; kwargs...) = begin
    rv = true
    for (key, value) in pairs(stan_model(p).vars)
        rvalue = get(kwargs, key, :missing)
        if cqual(value) !== rvalue
            @error "cqual($key) == $(cqual(value)) != $rvalue"
            rv = false
        end
    end 
    rv || @info map(cqual, (;pairs(stan_model(p).vars)...))
    return rv
end

transpiles(p; re=true) = try
    stan_code(p)
    return true
catch e
    re && rethrow()
    return false
end
compiles(p; re=true) = try
    instantiate(p)
    return true
catch e
    re && rethrow()
    return false
end