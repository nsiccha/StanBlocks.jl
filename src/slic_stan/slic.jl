include("definitions.jl")
include("types.jl")
# The compiler proper (tracing, the forward/backward passes, functions-block
# collection) walks expression nodes whose types differ per model, so it
# compiles each method once for its declared signature instead of once per
# node type (snag `stanblocks-trans-f759fe4f`). Dispatch is unchanged. The
# Julia runtime target in functions.jl re-enables specialization for itself.
@nospecialize
include("tracing.jl")
include("forward.jl")
include("passes.jl")

include("functions.jl")
@specialize

include("closures.jl")
include("macros.jl")

include("builtin.jl")

include("lpxf_builtin.jl")
include("valuefamily.jl")
include("show.jl")
include("build.jl")
include("instantiate.jl")
include("descriptor.jl")
include("bundle.jl")

include("test.jl")

include("exports.jl")
