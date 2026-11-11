using StanBlocks

# Published eight-schools treatment-effect estimates and their standard errors.
school_data = (;
    effect = [28.0, 8.0, -3.0, 7.0, -1.0, 1.0, 18.0, 12.0],
    standard_error = [15.0, 10.0, 16.0, 11.0, 9.0, 11.0, 10.0, 18.0],
    population_mean = 0.0,
    population_sd = 10.0,
)

const school_effect = @slic begin
    theta ~ normal(population_mean, population_sd)
    effect ~ normal(theta, standard_error)
    return theta
end
const schools = @slic begin
    theta ~ plate(effect, standard_error; outer=(length(effect),)) do estimate, uncertainty
        theta_i ~ school_effect(; effect=estimate, standard_error=uncertainty,
                                population_mean, population_sd)
        theta_i
    end
end
model = schools(; school_data...)
