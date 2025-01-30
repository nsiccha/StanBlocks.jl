broadcastable(x) = false # avoid dotting spliced objects (e.g. view calls inserted by @view)
# don't add dots to dot operators
broadcastable(x::Symbol) = (!Base.isoperator(x) || first(string(x)) != '.' || x === :..) && x !== :(:)
broadcastable(x::Expr) = x.head !== :$
unbroadcast(x) = x
function unbroadcast(x::Expr)
    if x.head === :.=
        Expr(:(=), x.args...)
    elseif x.head === :block # occurs in for x=..., y=...
        Expr(:block, Base.mapany(unbroadcast, x.args)...)
    else
        x
    end
end
__broadcasted__(x) = x
function __broadcasted__(x::Expr)
    broadcasted = :(Base.broadcasted)
    broadcastargs = Base.mapany(__broadcasted__, x.args)
    return if x.head === :call && broadcastable(x.args[1])
        Expr(:call, broadcasted, broadcastargs...)
    elseif x.head === :comparison
        error()
        Expr(:comparison, (iseven(i) && broadcastable(arg) && arg isa Symbol && Base.isoperator(arg) ?
                               Symbol('.', arg) : arg for (i, arg) in pairs(broadcastargs))...)
    elseif x.head === :$
        x.args[1]
    elseif x.head === :let # don't add dots to `let x=...` assignments
        Expr(:let, unbroadcast(broadcastargs[1]), broadcastargs[2])
    elseif x.head === :for # don't add dots to for x=... assignments
        Expr(:for, unbroadcast(broadcastargs[1]), broadcastargs[2])
    elseif (x.head === :(=) || x.head === :function || x.head === :macro) &&
           Meta.isexpr(x.args[1], :call) # function or macro definition
        Expr(x.head, x.args[1], broadcastargs[2])
    elseif x.head === :(<:) || x.head === :(>:)
        Expr(:call, broadcasted, x.head, broadcastargs...)
    else
        head = String(x.head)::String
        if last(head) == '=' && first(head) != '.' || head == "&&" || head == "||"
            Expr(:call, broadcasted, x.head, broadcastargs...)
        else
            Expr(x.head, broadcastargs...)
        end
    end
end
macro broadcasted(x)
    esc(__broadcasted__(x))
end

macro bsum(x)
    :(bsum($(esc(__broadcasted__(x)))))
end


const X_NAME = gensym("x")
const TMP = gensym("tmp")
const XPOS = gensym("xpos")

begin
sum_expr(f, args) = begin
    fargs = filter(!isnothing, map(f, args))
    length(fargs) > 0 ? Expr(:call, :+, fargs...) : nothing
end
compute_dimension(e) = nothing
compute_dimension(e::Expr) = if e.head == :macrocall && e.args[1] == Symbol("@parameters") 
    sum_expr(compute_dimension_, e.args)
else
    sum_expr(compute_dimension, e.args)
end
compute_dimension_(e) = nothing
compute_dimension_(e::Expr) = if e.head == :block
    sum_expr(compute_dimension_, e.args)
else
    @assert e.head == :(::)
    name, varinfo = e.args
    if !Meta.isexpr(varinfo, :ref)
        type, kws = extract_kws(varinfo)
        @assert type == :real
        1
    else
        type, kws = extract_kws(varinfo.args[1])
        dims = varinfo.args[2:end]
        dim = Symbol(type, "_unconstrained_dim")
        isdefined(StanBlocks, dim) && (dim = :(StanBlocks.$dim))
        :($dim($(dims...)))
    end
end
gq_symbols_lhs(e::Expr) = if e.head in (:(::), :(=))
    gq_symbols_lhs(e.args[1])
else
    []
end
gq_symbols_lhs(e::Symbol) = [e]
gq_symbols(e) = []
gq_symbols(e::Expr) = if e.head in (:(::), :(=))
    gq_symbols_lhs(e.args[1])
elseif e.head in (:for,:while,:do)
    []
else
    mapreduce(gq_symbols, vcat, e.args)
end
strip_gq_lhs(e::Symbol) = e == Symbol("@generated_quantities")
strip_gq_lhs(e::QuoteNode) = strip_gq_lhs(e.value)
strip_gq_lhs(e::Expr) = e.head == :(.) ? strip_gq_lhs(e.args[2]) : error("Something unforeseen: $e")
strip_gq(e) = e
strip_gq(e::Expr) = if e.head == :macrocall && strip_gq_lhs(e.args[1])
    nothing
else
    Expr(e.head, strip_gq.(e.args)...)
end

macro_stan(x) = begin
    lpdf_fname = gensym("stan_lpdf") 
    gq_fname = gensym("generate_quantities") 
    quote 
        function $lpdf_fname($X_NAME) 
            target = 0.
            $(strip_gq(x))
            return target
        end
        function $gq_fname($X_NAME) 
            target = 0.
            $x
            return $(Expr(:tuple, Expr(:parameters, unique(vcat(:target, gq_symbols(x)))...)))
        end
        StanBlocks.VectorPosterior($lpdf_fname, missing, $gq_fname, $(compute_dimension(x)))
    end
end
end
macro stan(x)
    esc(macro_stan(x))
end
macro_parameters(e) = e
macro parameters(block)
    esc(macro_parameters(block))
end
macro_transformed_parameters(e) = e
macro transformed_parameters(block)
    esc(macro_transformed_parameters(block))
end
macro_model(e) = e
macro model(block)
    esc(macro_model(block))
end
function macro_generated_quantities end
macro_generated_quantities(e) = e
macro generated_quantities(block)
    esc(macro_generated_quantities(block))
end
begin
    extract_kws(e::Symbol) = e, ()
    extract_kws(e::Expr) = begin 
        @assert e.head == :call
        e.args[1]::Symbol
        @assert all([Meta.isexpr(arg, :kw) for arg in e.args[2:end]])
        e.args[1], e.args[2:end]
    end
    macro_parameters(e::Expr) = begin 
        if e.head == :block
            Expr(:block, :($XPOS=1), macro_parameters.(e.args)...)
        else
            @assert e.head == :(::)
            name, varinfo = e.args
            stmts = if !Meta.isexpr(varinfo, :ref)
                type, kws = extract_kws(varinfo)
                @assert type == :real
                [
                    :(($TMP, $name) = StanBlocks.constrain($X_NAME[$XPOS]; $(kws...))),
                    :(target += $TMP),
                    :($XPOS += 1)
                ]
            else
                type, kws = extract_kws(varinfo.args[1])
                dims = varinfo.args[2:end]
                constrain = Symbol(type, "_constrain")
                isdefined(StanBlocks, constrain) && (constrain = :(StanBlocks.$constrain))
                dim = Symbol(type, "_unconstrained_dim")
                isdefined(StanBlocks, dim) && (dim = :(StanBlocks.$dim))
                [
                    :($XPOS = $XPOS:($XPOS+$dim($(dims...))-1)),
                    :(($TMP, $name) = $constrain(view($X_NAME, $XPOS), $(dims...); $(kws...))),
                    :(target += $TMP),
                    :($XPOS = $XPOS[end]+1)
                ]
            end
            Expr(:block, stmts...)
        end
    end
    macro_model(e::Expr) = begin 
        if e.head == :call && e.args[1] == :(~)
            lhs = e.args[2]
            rhs = e.args[3]
            @assert Meta.isexpr(rhs, :call)
            lpdf = Symbol("$(rhs.args[1])_lpdf")
            isdefined(StanBlocks, lpdf) && (lpdf = :(StanBlocks.$lpdf))
            rhsargs = rhs.args[2:end]
            rv = :(target += $lpdf($lhs, $(rhsargs...)))
            # display(Pair(e, rv))
            rv
        else
            Expr(macro_model(e.head), macro_model.(e.args)...)
        end
    end
end