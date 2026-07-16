# Two methods of one sub-model function `dcase`, dispatched on argument TYPE:
# a `::real` arg gets a N(0,1) prior; an `::int` arg gets a N(0,5) prior. The
# emitted Stan must show BOTH priors, proving native @deffun-style dispatch.
@slic dcase(s::real) = begin
    z ~ normal(0, 1)
    return z + s
end
@slic dcase(s::int) = begin
    z ~ normal(0, 5)
    return z + s
end
@slic (; sc=2.0, si=3, y=[1.0, 2.0]) begin
    a ~ dcase(sc)   # sc::real  -> N(0,1) method
    b ~ dcase(si)   # si::int   -> N(0,5) method
    y ~ normal(a + b, 1)
end
