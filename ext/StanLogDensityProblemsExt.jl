module StanLogDensityProblemsExt
import StanLogDensityProblems, StanBlocks

StanBlocks.stan.instantiate(x::Union{StanBlocks.stan.SlicModel,StanBlocks.stan.StanModel}; nan_on_error=true, make_args=["STAN_THREADS=true"], warn=false, kwargs...) = begin 
    stan_code = StanBlocks.stan.stan_code(x)
    stan_path = get(kwargs, :path, joinpath("tmp", string(hash(stan_code)) * ".stan"))
    mkpath(dirname(stan_path))
    if !isfile(stan_path)
        open(stan_path, "w") do fd
            write(fd, stan_code)
        end
    end
    StanLogDensityProblems.StanProblem(
        stan_path, 
        StanBlocks.stan.bridgestan_data(StanBlocks.stan.stan_data(x)); 
        nan_on_error,
        make_args,
        warn
    )
end

end