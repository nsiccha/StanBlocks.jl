using Pkg, TOML

python = get(ENV, "RK_JAX_PYTHON", "")
isempty(python) && error("Set RK_JAX_PYTHON to a Python executable with JAX, NumPy and SciPy")
root = normpath(joinpath(@__DIR__, "..", ".."))
sources = TOML.parsefile(joinpath(@__DIR__, "Project.toml"))["sources"]
mktempdir() do env
    Pkg.activate(env)
    withenv("JULIA_PKG_PRECOMPILE_AUTO" => "0") do
        Pkg.add([PackageSpec(; name, url=source["url"], rev=source["rev"])
            for (name, source) in sources if haskey(source, "url")])
        Pkg.develop([
            PackageSpec(path=root),
            PackageSpec(path=joinpath(@__DIR__, "fixtures", "SourceProviderFixture")),
        ])
    end
    Pkg.precompile()
    check = """
    using SourceProviderFixture, StanBlocks, ReactiveKernels
    f = SourceProviderFixture.precompiled_loop
    method = which(f, Tuple{Float64})
    source = StanBlocks._deffun_julia_source(method.sig)
    @assert source !== nothing
    @assert source[1] === method
    @assert source[2] == (:x,)
    @assert source[4] === SourceProviderFixture
    @assert f(2.0) == 20.0
    ext = Base.get_extension(StanBlocks, :StanBlocksReactiveKernelsExt)
    @assert ext.deffun_source(f, Tuple{Float64}) isa TraceableSource
    @kernel precompiled_kernel(x::Float64) = begin
        result = SourceProviderFixture.precompiled_loop(x)
        return result
    end
    write("precompiled_kernel.py", jax_source(precompiled_kernel; have=(:x,), want=:result))
    println("Fresh process loaded downstream metadata and emitted JAX")
    """
    cd(env) do
        run(`$(Base.julia_cmd()) --startup-file=no --project=$env -e $check`)
        write("check.py", """
        import jax
        import precompiled_kernel as model
        assert abs(float(jax.jit(model.kernel)(2.0)) - 20.0) < 1e-10
        assert abs(float(jax.jit(jax.grad(model.kernel))(2.0)) - 10.0) < 1e-10
        print('DOWNSTREAM PRECOMPILE, FRESH SOURCE DISPATCH, JAX VALUE AND GRADIENT OK')
        """)
        run(`$python check.py`)
    end
end
