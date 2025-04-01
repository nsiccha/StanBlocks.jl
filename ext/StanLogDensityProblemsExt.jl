module StanLogDensityProblemsExt
import StanLogDensityProblems, StanBlocks

StanBlocks.stan.instantiate(x::StanBlocks.stan.SlicModel; nan_on_error=true, kwargs...) = begin 
    code_info = StanBlocks.code(x)
    buf = IOBuffer()
    print(buf, code_info)
    stan_code = String(take!(buf))
    stan_path = get(kwargs, :path, joinpath("tmp", string(hash(stan_code)) * ".stan"))
    lib_path = replace(stan_path, ".stan"=>"_model.so")
    mkpath(dirname(stan_path))
    if !isfile(stan_path)
        open(stan_path, "w") do fd
            write(fd, stan_code)
        end
    end
    if mtime(lib_path) < mtime(stan_path)
        @info (stan_code) 
        @info "Compiling $stan_path..."
    else
         @info "Not compiling $stan_path..."
    end
    stan_data = Dict()
    for name in keys(code_info)
        name in keys(x.data) || continue
        stan_data!(code_info[name], x.data[name]; stan_data)
    end
    StanProblem(stan_path, StanBlocks.bridgestan_data(stan_data); nan_on_error)
end

end