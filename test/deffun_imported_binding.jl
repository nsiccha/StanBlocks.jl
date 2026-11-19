"""
A bodyful `@deffun` may define a function whose binding another module owns and
the defining module imports (`import Owner: f`). The Stan definition attaches to
that one function object, and the body's helper calls resolve in the module
where `@deffun` was expanded, not in `parentmodule(f)` (snag
`deffun-body-on-a-64fbbe59`). The owner below loads nothing and has none of the
helpers, as when a lighter package holds a binding for dispatch hooks.

`Imported` and `Owned` evaluate the same source; only `Imported` imports the
binding, so the emitted Stan of the two must be byte-identical.
"""
@testmodule ImportedBindingFixtures begin
    module BindingOwner
        function scaled_locations end
        function compat_locations end
    end

    const DEFINITIONS = quote
        @deffun begin
            step_offset(x::real)::real = x + 1.0
        end
        @deffun @juliacompat begin
            doubled(x::real)::real = 2.0 * x
            compat_locations(x::vector[n])::vector[n] = begin
                out::vector[n]
                for i in 1:n
                    out[i] = doubled(x[i]) + 1.0
                end
                out
            end
            @stanonly scaled_locations(x::vector[n])::vector[n] = begin
                out::vector[n]
                for i in 1:n
                    out[i] = step_offset(doubled(x[i]))
                end
                out
            end
        end
        const model = @slic (; y=[0.1, 0.2]) begin
            x ~ normal(0.0, 1.0)
            y ~ normal(scaled_locations(rep_vector(x, 2)) + compat_locations(rep_vector(x, 2)), 1.0)
        end
    end

    module Imported
        using StanBlocks
        import ..BindingOwner: scaled_locations, compat_locations
    end
    module Owned
        using StanBlocks
    end
    Core.eval(Imported, DEFINITIONS)
    Core.eval(Owned, DEFINITIONS)
end

@testitem "slic: @deffun on an imported binding resolves its body in the defining module" tags=[:slic, :regression] setup=[StanBlocksImports, ImportedBindingFixtures] begin
    using .ImportedBindingFixtures: BindingOwner, Imported, Owned
    # One function object: the import, not a new local function.
    @test Imported.scaled_locations === BindingOwner.scaled_locations
    @test Imported.compat_locations === BindingOwner.compat_locations
    @test !isdefined(BindingOwner, :step_offset) && !isdefined(BindingOwner, :doubled)

    imported = stan_code(Imported.model)
    @test imported == stan_code(Owned.model)
    for name in ("scaled_locations", "compat_locations", "step_offset", "doubled")
        @test occursin(name * "(", imported)
    end

    # A `@juliacompat` member installs its Julia method on the owner's function,
    # since that function had no methods; its helper call resolves in `Imported`.
    @test BindingOwner.compat_locations([0.1, 0.2]) ≈ [1.2, 1.4]
    @test_throws MethodError BindingOwner.scaled_locations([0.1, 0.2])
end

@testitem "slic: @deffun on an imported binding emits Stan that stanc accepts" tags=[:slic, :regression, :stanc] setup=[StanBlocksImports, StanBlocksTestSetup, ImportedBindingFixtures] begin
    using .StanBlocksTestSetup: stanc_compiles
    using .ImportedBindingFixtures: Imported
    @test stanc_compiles(Imported.model)
end
