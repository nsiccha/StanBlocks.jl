module MarkdownExt
import Markdown, StanBlocks

module Quarto 
    using Markdown
    abstract type Object end
    struct Container <: Object
        children
    end
    struct Heading <: Object 
        level
        obj
    end
    struct Div <: Object
        header
        content
    end
    struct Code <: Object 
        header
        content
    end
    struct Tabset <: Object
        map
    end
    prettyprint(io, args...) = print(io, pretty(args)...)
    pretty(x::Expr) = if x.head == :macrocall
        Expr(x.head, x.args[1:2]..., pretty(x.args[3:end])...)
    else
        Expr(x.head, pretty(x.args)...)
    end
    pretty(x::Union{Vector,Tuple}) = begin
        rv = []
        for xi in x
            rvi = pretty(xi)
            isa(rvi, Vector) ? append!(rv, rvi) : push!(rv, rvi)
        end
        rv
    end
    pretty(x::LineNumberNode) = []
    pretty(x) = x
    Base.show(io::IO, m::MIME"text/markdown", x::Object) = print(io, Markdown.parse(string(x)))
    Base.show(io::IO, x::Container) = for child in x.children
        print(io, child, "\n")
    end
    Base.show(io::IO, x::Heading) = print(io, repeat("#", x.level), " ", x.obj, "\n")
    Base.show(io::IO, x::Div) = begin 
        print(io, "\n::: ", x.header, "\n\n")
        print(io, x.content)
        print(io, "\n:::\n")
    end
    Base.show(io::IO, x::Code) = if x.header == "julia" 
        buf = IOBuffer()
        prettyprint(buf, "\n```", x.header, "\n", x.content, "\n```\n")
        print(io, replace(String(take!(buf)), "\n    "=>"\n"))
    else
        prettyprint(io, "\n```", x.header, "\n", x.content, "\n```\n")
    end
    
    Base.show(io::IO, x::Tabset) = begin
        print(io, Div("{.panel-tabset}", Container([
            Container([
                Heading(5, key),
                value
            ])
            for (key, value) in pairs(x.map)
        ])))
    end
end

mapkv(f, x) = map(f, keys(x), values(x))
msg(e) = try 
    string(e)
catch
    "Something went wrong!"
end
msg(e::ErrorException) = msg(e.msg)
quarto(x::StanBlocks.stan.SlicModel) = try 
    Quarto.Container([
        Quarto.Heading(5, "SlicModel"),
        Quarto.Tabset((;
            Specification=Quarto.Code("julia", x.model),
            # Data=Quarto.Code("", Quarto.Container(mapkv((k,v)->"$k:\t$(typeof(v))", StanBlocks.stan.stan_data(x)))),
            Stan=Quarto.Code("stan", StanBlocks.stan.stan_code(x))
        ))
    ])
catch e
    Quarto.Container([
        Quarto.Heading(5, "SlicModel"),
        Quarto.Tabset((;
            Specification=Quarto.Code("julia", x.model),
            ERROR=Quarto.Code("", msg(e)),
        ))
    ])
end

Base.show(io::IO, m::MIME"text/markdown", x::StanBlocks.stan.SlicModel) = show(io, m, quarto(x))
StanBlocks.Code(args...; kwargs...) = Quarto.Code(args...; kwargs...)
StanBlocks.Tabset(args...; kwargs...) = Quarto.Tabset(args...; kwargs...)
end