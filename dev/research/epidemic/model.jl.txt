using StanBlocks

@deffun sir_rhs(t::real, state::vector[3], infection_rate::real,
                removal_rate::real, population::real)::vector[3] = begin
    susceptible = state[1]
    infectious = state[2]
    new_infections = infection_rate * susceptible * infectious / population
    removals = removal_rate * infectious
    derivative::vector[3]
    derivative[1] = -new_infections
    derivative[2] = new_infections - removals
    derivative[3] = removals
    derivative
end

sir_data = (;
    initial = [762.0, 1.0, 0.0], population = 763.0,
    times = collect(0.5:0.5:20.0), start = 0.0,
    log_infection_location = log(2.0), log_removal_location = log(0.4),
    log_rate_sd = 0.2,
)
const sir = @slic begin
    infection_rate ~ lognormal(log_infection_location, log_rate_sd)
    removal_rate ~ lognormal(log_removal_location, log_rate_sd)
    trajectory = ode_rk45(sir_rhs, initial, start, to_array_1d(times),
                          infection_rate, removal_rate, population)
    return trajectory
end
model = sir(; sir_data...)
