using Revise
using PosteriorDBWeb

begin
    PosteriorDBWeb.terminate()
    port = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 8080
    PosteriorDBWeb.serve(; host="0.0.0.0", revise=:lazy, port, async=true)
end
