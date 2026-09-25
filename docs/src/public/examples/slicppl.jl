# Historical SlicPPL prototype retained from docs/examples/slicppl.jl.
using OrderedCollections
using Distributions, Turing, LinearAlgebra, Tullio, LogExpFunctions

abstract type AbstractModel{M} end
const ORIGINAL_X = gensym("x")
const ORIGINAL_RNG = gensym("rng")
struct Parameter end
const parameter = Parameter()
struct Generated end
const generated = Generated()
Base.getproperty(x::AbstractModel{M}, k::Symbol) where {M} = hasfield(typeof(x), k) ? getfield(x, k) : getproperty(x.meta, k)
(x::AbstractModel)(;kwargs...) = incremental(x, (;kwargs...))
jslic_top(x::LineNumberNode) = x
jslic_top(x::Expr) = if x.head == :block
    Expr(x.head, jslic_top.(x.args)...)
else
    @assert x.head == :(=)
    lhs, rhs = x.args
    qrhs = Meta.quot(rhs)
    quote 
        struct $lhs{M} <: AbstractModel{M}
            meta::M
            $lhs(meta) = new{typeof(meta)}(meta)
            $lhs(;kwargs...) = $lhs((;))(;kwargs...)
        end
        specification(::Type{<:$lhs}) = $qrhs
        @generated incremental(x::$lhs{M}, kwargs::N) where {M,N} = incremental_expr($qrhs, M, N)
        @generated Base.rand(rng::AbstractRNG, x::$lhs{M}) where {M} = rand_expr($qrhs, M; full=false)
        @generated fullrand(rng::AbstractRNG, x::$lhs{M}) where {M} = rand_expr($qrhs, M; full=true)
        remake(x::$lhs; kwargs...) = $lhs((;kwargs...))
    end
end
macro jslic(x)
    esc(jslic_top(x))
end
begin
    autobroadcasted(fargs...) = Base.broadcasted(fargs...)
    # isamodel(x) = false
    # isamodel(::Type{<:AbstractModel}) = true
    struct Data{Age,P}
        parent::P
        Data(Age, parent)=new{Age,typeof(parent)}(parent)
    end
    Base.parent(x::Data) = x.parent
    ConstData{P} = Data{-1,P}
    OldData{P} = Data{0,P}
    NewData{P} = Data{1,P}
    old_data(x) = Data(0, x)
    new_data(x) = Data(1, x)
    old_data(x::Data) = old_data(parent(x))
    new_data(x::Data) = new_data(parent(x))
    old_data(x::Parameter) = x
    new_data(x::Parameter) = x
    old_data(x::Generated) = x
    new_data(x::Generated) = x
    struct Delayed{F,A,K}
        f::F
        args::A
        kwargs::K
    end
    delayed(f, args...; kwargs...) = Delayed(f, args, (;kwargs...))

    collect_locals!(x; locals) = x
    collect_locals!(x::Expr; locals) = if x.head == :(=)
        lhs, rhs = x.args
        collect_locals!(rhs; locals)
        # @assert isa(lhs, Symbol) lhs
        push_locals!(lhs; locals)
    elseif x.head == :call && x.args[1] == :(~)
        _, lhs, rhs = x.args 
        collect_locals!(rhs; locals)
        # @assert isa(lhs, Symbol)
        push_locals!(lhs; locals)
    else
        collect_locals!.(x.args; locals)
    end
    push_locals!(x::Symbol; locals) = push!(locals, x)
    push_locals!(x::Expr; locals) = if x.head == :tuple
        push_locals!(Symbol(x); locals)
        push_locals!.(x.args; locals)
    else
        error(x)
    end
    maybematerialize(x) = x
    maybematerialize(x::Expr) = if Meta.isexpr(x, :call) && x.args[1] == :delayed && x.args[2] == autobroadcasted
        Expr(:call, :delayed, Base.materialize, x)
    else
        x
    end
    delayed_expr(x) = x
    delayed_expr(x::Expr) = if x.head == :(=)
        lhs, rhs = x.args
        # @assert isa(lhs, Symbol)
        Expr(x.head, lhs, maybematerialize(delayed_expr(rhs)))
    elseif x.head == :call
        f, args... = x.args
        if isa(f, Symbol) && startswith(string(f), ".")
            return delayed_expr(Expr(:call, autobroadcasted, Symbol(string(f)[2:end]), args...))
        end
        fargs = if length(args) > 0 && Meta.isexpr(args[1], :parameters)
            args[1], f, args[2:end]...
        else
            f, args...
        end
        Expr(x.head, :delayed, maybematerialize.(delayed_expr.(fargs))...)
    elseif x.head == :(.)
        lhs, rhs = x.args
        if isa(rhs, QuoteNode)
            delayed_expr(Expr(:call, :getproperty, lhs, rhs))
        else
            @assert Meta.isexpr(rhs, :tuple)
            delayed_expr(Expr(:call, autobroadcasted, lhs, rhs.args...))
        end
    elseif x.head == Symbol("'")
        delayed_expr(Expr(:call, :adjoint, x.args...))
    elseif x.head == :vect
        delayed_expr(Expr(:call, :vect, x.args...))
    elseif x.head == :ref
        delayed_expr(Expr(:call, :getindex, x.args...))
    elseif x.head == :tuple
        delayed_expr(Expr(:call, :tuple, x.args...))
    else
        @assert x.head in (:block, :parameters, :kw, :macrocall) (;x.head, x.args)
        Expr(x.head, delayed_expr.(x.args)...)
    end
    forward_expr(x) = x
    forward_expr(x::Expr) = if x.head == :(=)
        lhs, rhs = x.args
        # @assert isa(lhs, Symbol)
        if isa(lhs, Symbol)
            Expr(x.head, lhs, Expr(:call, :incremental, lhs, forward_expr(rhs)))
        elseif lhs.head == :tuple
            tmp = Symbol(lhs)
            Expr(:block,
                forward_expr(Expr(:(=), tmp, rhs)),
                [
                    forward_expr(Expr(:(=), arg, forward_expr(delayed_expr(:($tmp[$i])))))
                    for (i, arg) in enumerate(lhs.args)
                ]...
            )
        else
            error(lhs)
        end
    else
        Expr(x.head, forward_expr.(x.args)...)
    end
    # incremental_lhs_expr(x::Symbol) = x
    # incremental_lhs_expr(x::Expr) = if x.head == :tuple
    #     push_locals!.(x.args; locals)
    # else
    #     error(x)
    # end
    backward_expr(x; locals) = nothing
    backward_expr(x::Expr; locals) = if x.head == :block
        Expr(x.head, filter(!isnothing, backward_expr.(reverse(x.args); locals))...)
    elseif x.head == :(=)
        lhs, rhs = x.args
        stmts = OrderedSet()
        backward_expr!(rhs; stmts, locals)
        length(stmts) > 0 || return
        Expr(:if, :(data_age($lhs) == 2), Expr(:block, stmts...))
    elseif x.head == :call
        @assert x.args[1] == :delayed x
        x.args[2] == :~ || return
        _, _, lhs, rhs = x.args
        stmts = OrderedSet()
        backward_expr!(rhs; stmts, locals)
        length(stmts) > 0 || return
        Expr(:if, :(data_age($lhs) <= 1), Expr(:block, stmts...))
    else
        @assert false x
    end
    backward_expr!(x; stmts, locals) = nothing
    backward_expr!(x::Symbol; stmts, locals) = x in locals && push!(stmts, :($x = affects_likelihood($x)))
    backward_expr!(x::Expr; stmts, locals) = if x.head == :call
        @assert x.args[1] == :delayed x
        x.args[2] == :getproperty || return backward_expr!.(x.args; stmts, locals)
        _, _, lhs, rhs = x.args
        lhs in locals && push!(stmts, :($lhs = affects_likelihood($lhs, $rhs)))
    else
        backward_expr!.(x.args; stmts, locals)
    end

    data_age(::Any) = -1
    data_age(::Data{Age}) where {Age} = Age
    data_age(::Parameter) = 2
    data_age(::Generated) = 4
    data_age(x::AbstractModel) = map(data_age, x.meta)
    data_age(x::Delayed) = maximum(data_age, (x.f, x.args..., x.kwargs...))
    max_data_age(x) = maximum(data_age(x))
    undelay(x) = x
    undelay(x::Tuple) = map(undelay, x)
    undelay(x::NamedTuple) = map(undelay, x)
    undelay(x::Data) = undelay(parent(x))
    undelay(x::Delayed) = undelay(x.f)(undelay(x.args)...; undelay(x.kwargs)...)
    delayed(::Type{T}; kwargs...) where {T<:AbstractModel} = T(;kwargs...)
    delayed(::typeof(getproperty), x::AbstractModel, k::Symbol) = getproperty(x, k)
    affects_likelihood(x) = data_age(x) > 2 ? parameter : x
    affects_likelihood(x::AbstractModel, k::Symbol) = begin
        xk = getproperty(x, k)
        data_age(xk) > 2 ? remake(x; x.meta..., k=>affects_likelihood(xk)) : x
        
    end
    incremental(x::Generated, y::AbstractModel) = y
    incremental(x::Generated, y) = if data_age(y) >= 2
        x
    else
        Data(data_age(y), undelay(y))
    end
    incremental(x::Tuple, y) = error()#map(incremental, x, y)
    incremental(x::ConstData, y) = error()#undelay(y)
    incremental_expr(x::Expr, M, N) = begin 
        old_locals = fieldnames(M)
        new_locals = fieldnames(N)
        locals = OrderedSet()
        collect_locals!(x; locals)
        forward_x = forward_expr(delayed_expr(x))
        backward_x = backward_expr(delayed_expr(x); locals)
        all_locals = union(locals, old_locals, new_locals)
        quote
            $ORIGINAL_X = x
            (
                ($(locals...),), 
                (;$(old_locals...)), 
                (;$(new_locals...))
            ) = (
                $(Expr(:tuple, fill(generated, length(locals))...)),
                map(old_data, x.meta), 
                map(new_data, kwargs)
            )
            $forward_x
            $backward_x
            remake($ORIGINAL_X; (;$(all_locals...))...)
        end
    end
    rand_expr(x) = x
    rand_expr(x::Expr) = if x.head == :(=)
        lhs, rhs = x.args
        # @assert isa(lhs, Symbol)
        Expr(x.head, lhs, Expr(:call, maybeassign, lhs, ORIGINAL_RNG, rand_expr(rhs)))
    elseif x.head == :call && x.args[1] == :delayed && x.args[2] == :~
        _, _, lhs, rhs = x.args
        Expr(:(=), lhs, Expr(:call, mayberand, lhs, ORIGINAL_RNG, rand_expr(rhs)))
    else
        Expr(x.head, rand_expr.(x.args)...)
    end
    maybeassign(x, rng, y::AbstractModel) = fullrand(rng, y)
    maybeassign(x, rng, y) = undelay(y)
    # maybeassign(rng, y::Sampleable) = rand(rng, y)
    # maybeassign(rng, y) = y
    mayberand(x, rng, y::AbstractModel) = rand(rng, y)
    mayberand(x, rng, y) = rand(rng, undelay(y))
    # mayberand(rng, y::Sampleable) = rand(rng, y)
    # mayberand(rng, y) = y
    rand_expr(x::Expr, M; full) = begin 
        old_locals = fieldnames(M)
        locals = OrderedSet()
        collect_locals!(x; locals)
        all_locals = union(locals, old_locals)
        x = rand_expr(delayed_expr(x))
        qx = Meta.quot(x)
        if full
            quote
                $ORIGINAL_RNG = rng
                (;$(old_locals...)) = x
                @info $qx
                $x
                (;$(all_locals...))
            end
        else
            quote
                $ORIGINAL_RNG = rng
                (;$(old_locals...)) = x
                @info $qx
                $x
            end
        end
    end
    replace(x; d) = x in keys(d) ? getindex(d, x) : x
    replace(x::Expr; d) = if x.head == :kw && length(x.args) == 2
        Expr(x.head, x.args[1], replace(x.args[2]; d))
    else
        Expr(x.head, replace.(x.args; d)...)
    end
    status(x::AbstractModel) = begin 
        age = map(data_age, x.meta)
        @info age
        age_types = Dict(-1=>:Const, 0=>:Data,1=>:Data, 2=>:Parameters,3=>:TransformedParameters,4=>:Generated)
        d = (;[
            key=>:($key::$(get(age_types, value, :Mixed))) for (key, value) in pairs(age)
        ]...)
        replace(specification(typeof(x)); d)
    end
    Base.show(io::IO, x::AbstractModel) = print(io, status(x))
end
begin
    conv(X::AbstractMatrix, w::AbstractVector) = @tullio rv[i,j+_] := X[i, pad(j - k, length(w) - 1)] * w[k]
    AR1_unwhiten(xi; alpha) = begin 
        delta = sqrt(1 - alpha^2)
        x = similar(xi)
        x[1] = xi[1]
        for t in 2:length(x)
            x[t] = alpha * x[t-1] + delta * xi[t]
        end
        x
    end
    compute_flux(;F_id, F_in, F_out, beta, rho) = begin 
        @tullio F[i,j,k] := (1-rho[k]) * F_id[i,j] + rho[k] * (beta*F_out[i,j]+(1-beta)*F_in[i,j]) 
    end
    RegionalInfections_unwhiten(xi; F, reproduction_number, infection_weights, global_scale) = begin 
        num_regions, num_days = size(reproduction_number)
        current_mean = fill(global_scale, num_regions)
        tmp = similar(current_mean)
        num_infections = similar(xi)
        psi = 1.
        for day in 1:num_days
            n_convolve = min(length(infection_weights), day-1)
            mul!(tmp, num_infections[:, day-n_convolve:day-1], infection_weights[end+1-n_convolve:end])
            mul!(current_mean, F[:, :, day], tmp)
            current_mean .= reproduction_number[:, day] .* current_mean .+  global_scale
            num_infections[:, day] .= invlogcdf.(
                truncated.(Normal.(current_mean, sqrt.((1 + psi) .* current_mean)); lower=0.), 
                logcdf.(Normal(), xi[:, day])
            )
        end
        num_infections
    end
@jslic begin 
    MaternHalf = begin 
        distances = sqrt.(abs2.(x .- x'))
        unit_K = exp.(.-distances)
        x_scale ~ truncated(Normal(0, 5); lower=0.)
        y_scale ~ truncated(Normal(0, .5); lower=0.)
        K = y_scale^2 .* unit_K .^ inv.(x_scale)
        nugget = 1e-8
        y ~ MvNormal(zeros(length(x)), K + nugget * I)
    end
    SpatioTemporalGP = begin 
        num_regions = length(global_space)
        num_times = length(time)
        global_GP = MaternHalf(;x=global_space)
        local_GP = MaternHalf(;x=local_space, x_scale=1.)
        time_GP = MaternHalf(;x=time, y_scale=1.)
        xi ~ filldist(Normal(0, 1), num_regions, num_times)
        y = cholesky(global_GP.K + local_GP.K).L * xi * cholesky(time_GP.K).U
    end
    CenteredSpatioTemporalGP = SpatioTemporalGP(begin 
        xi = nothing
        y ~ KronMvNormal(zeros((num_regions, num_times)), global_GP.K + local_GP.K, time_GP.K)
    end)
    AR1 = begin 
        alpha ~ Uniform(-1, 1)
        location ~ Normal(0, 1)
        scale ~ truncated(Normal(0, 1); lower=0.)
        xi ~ filldist(Normal(0, 1), num_steps)
        x = location .+ scale .* AR1_unwhiten(xi; alpha)
    end
    # CenteredAR1 = AR1(...)
    LogisticAR1 = begin 
        xi ~ AR1(;num_steps)
        x = logistic.(xi)
    end
    RegionalInfections = begin 
        num_regions, num_days = size(reproduction_number)
        F_id = Diagonal(ones(num_regions))
        latent_scale ~ truncated(Normal(0, 5); lower=0)
        global_scale ~ truncated(Normal(0, 5); lower=0)
        global_scale2 ~ truncated(Normal(0, global_scale); lower=0)
        beta ~ Uniform(0, 1)
        rho ~ LogisticAR1(;num_steps=num_days)
        F = compute_flux(;F_id, F_in, F_out, beta, rho)
        xi ~ filldist(Normal(0, 1), num_regions, num_days)
        num_infections = RegionalInfections_unwhiten(xi; F, reproduction_number, infection_weights, global_scale)
    end
    # CenteredRegionalInfections = RegionalInfections(...)
    NegBinomialWeeklyAdjustedTesting = begin 
        num_regions, num_days = size(num_infections)
        weekly_case_variation ~ Dirichlet(fill(7., 7))
        expected_num_detections = clamp.(conv(num_infections, detection_weights), 1e-3, 1e7)
        expected_num_detections_weekly_adj = 7 .* expected_num_detections .* getindex.(
            Ref(weekly_case_variation),
            mod1.(1:num_days, 7)
        )
        overdispersion ~ filldist(truncated(Normal(0, 5); lower=0.), num_regions)
        num_detections ~ arraydist(Normal.(expected_num_detections_weekly_adj, overdispersion))
    end
    EpiMap = begin 
        log_reproduction_number ~ SpatioTemporalGP(;global_space, local_space, time)
        reproduction_number = exp.(log_reproduction_number)
        num_infections ~ RegionalInfections(;reproduction_number, infection_weights, F_in, F_out)
        num_detections ~ NegBinomialWeeklyAdjustedTesting(;num_infections, detection_weights)
    end
end
# @generated logpdf(x::MaternHalf) = QuoteNode(specification(x))
# logpdf(MaternHalf())
# display(MaternHalf(;x=randn(10), y=1))
n_x = 40
n_t = 40
n_clusters = 4
local_space = randn(n_x)
global_means = randn(n_clusters)
global_space = global_means[argmin.(eachcol(abs2.(local_space' .- global_means)))]
my_time = randn(n_t)
infection_weights = detection_weights = exp.(-(1:10))
F_id = F_in = F_out = zeros((n_x, n_x))
prior = EpiMap(;global_space, local_space, time=my_time, infection_weights, detection_weights, F_in, F_out)
# display(prior)
rand(Xoshiro(0), prior)
# display(prior.GP.time_GP.unit_K)
# display(EpiMap())
end
# begin 
#     @jslic ms_garch_marginalized = begin 
#         transition_matrix ~ filldist(Dirichlet(ones(K)), K)
#         initial_volatility ~ ordered(filldist(HalfNormal(0, 0.5), K))
#         persistence0 ~ filldist(Uniform(0, 1), K)
#         persistence1 ~ arraydist(Uniform.(0, 1 .- persistence0))
#         volatilities = compute_volatilities(initial_volatility)
#         initial_probability = fill(1/K, K)
#         trans = permutedims(transition_matrix)
#         emissions = eachrow(Normal.(0, persistence0))
#         hmm = TimeHMM(initial_probability, trans, emissions)
#         y ~ ObservationSequence(hmm, 1:7)
#     end 
#     # ms_garch_marginalized(;K=2, T=7)
# end
# begin 
#     using DifferentialEquations
#     vect(args...) = [args...]
#     const N = 763
#     sir_ode!(du, u, p, t) = begin 
#         S, I, R = u
#         beta, gamma = p
#         du[1] = -beta*S*I/N
#         du[2] = beta*S*I/N-gamma*I
#         du[3] = gamma*I
#     end
#     u0 = [N-1, 1, 0]
#     beta, gamma = 2., .6
#     tspan = (.0, 14.)
#     problem = ODEProblem(sir_ode!, u0, tspan, [beta, gamma])
#     @jslic sir_model = begin 
#         beta ~ truncated(Normal(2, 1); lower=0)
#         gamma ~ truncated(Normal(.4, .5); lower=0)
#         inverse_dispersion_parameter ~ Exponential(1/5)
#         dispersion_parameter = inv(inverse_dispersion_parameter)
#         new_problem = remake(problem, p=[beta, gamma])
#         sol = solve(new_problem, saveat=1)
#         sol_for_observed = sol[2, :]
#         n = length(sol_for_observed)
#         in_bed ~ arraydist(NegativeBinomial.(sol_for_observed .+ 1e-5, dispersion_parameter))
#     end
#     sir_model(;problem)
# end

# begin 
#     x = zeros(0)
#     x1 = randn(5)
#     append!(x, x1)
#     x1 = view(x, 1:5)
#     x2 = randn(6)
#     append!(x, x2)
#     x .= 3
#     x1
# end
