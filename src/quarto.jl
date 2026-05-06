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
        pretty(x.args[2:end])
    else
        Expr(x.head, pretty(x.args)...)
    end
    _push_pretty!(rv, rvi::Vector) = (append!(rv, rvi); rv)
    _push_pretty!(rv, rvi) = (push!(rv, rvi); rv)
    pretty(x::Union{Vector,Tuple}) = begin
        rv = []
        for xi in x
            _push_pretty!(rv, pretty(xi))
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
    Base.show(io::IO, x::Code) = prettyprint(io, "\n```", x.header, "\n", x.content, "\n```\n")
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

Base.show(io::IO, m::MIME"text/markdown", x::stan.SlicModel) = show(io, m, quarto(x))

quarto(x::stan.SlicModel) = Quarto.Container([
    Quarto.Heading(5, "SlicModel"),
    Quarto.Tabset((;stan=Quarto.Code("stan", stan_code(x)), julia=Quarto.Code("julia", x.model)))
])
