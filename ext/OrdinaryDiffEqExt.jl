module OrdinaryDiffEqExt
import OrdinaryDiffEq, StanBlocks

StanBlocks.integrate_ode_rk45(f, y0, t0, t, theta, x_r=missing, x_i=missing, reltol=1e-6, abstol=1e-6, maxiters=100_000_000) = begin 
    jf(u, p, t) = f(t, u, p...)
    prob = OrdinaryDiffEq.ODEProblem(jf, y0, (t0, t[end]), (theta, x_r, x_i))
    sol = OrdinaryDiffEq.solve(prob, OrdinaryDiffEq.RK4(); saveat=t, abstol, reltol, maxiters)
    reduce(hcat, sol.u::Vector{typeof(y0)})'
end
StanBlocks.integrate_ode_bdf(f, y0, t0, t, theta, x_r=missing, x_i=missing, reltol=1e-10, abstol=1e-10, maxiters=100_000_000) = begin 
    jf(u, p, t) = f(t, u, p...)
    prob = OrdinaryDiffEq.ODEProblem(jf, y0, (t0, t[end]), (theta, x_r, x_i))
    # I think Julia doesn't have easy access to the BDF solver used by Stan 
    sol = OrdinaryDiffEq.solve(prob, OrdinaryDiffEq.Cash4(); saveat=t, abstol, reltol, maxiters)
    reduce(hcat, sol.u::Vector{typeof(y0)})'
end
end