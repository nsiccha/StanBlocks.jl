"""
An inline @deffun resolves its body and omitted keyword defaults in its
defining module, while explicit arguments and later caller statements keep
the caller's module (todo uej5f). The old trace silently called the caller's
same-named sibling; its Stan compiled but its density was wrong.
"""
@testmodule InlineDefinitionScopeFixtures begin
    module BindingOwner
        function extended_inline end
    end
    module Definitions
        using StanBlocks
        import ..BindingOwner
        @deffun sibling_offset(x::real)::real = x + 1.0
        @deffun @inline inline_offset(x::real)::real = sibling_offset(x)
        @deffun @inline inline_default(x::real; shift=sibling_offset(0.0))::real = x + shift
        @deffun @inline BindingOwner.extended_inline(x::real)::real = sibling_offset(x)
    end
    module Caller
        using StanBlocks
        import ..Definitions: inline_offset, inline_default
        import ..BindingOwner: extended_inline
        @deffun sibling_offset(x::real)::real = x + 100.0
        @deffun wrapper_offset(x::real)::real = inline_offset(x)
        const direct = @slic (; y=0.1) begin
            x ~ normal(0.0, 1.0)
            y ~ normal(inline_offset(x), 1.0)
        end
        const wrapped = @slic (; y=0.1) begin
            x ~ normal(0.0, 1.0)
            y ~ normal(wrapper_offset(x), 1.0)
        end
        const after_inline = @slic (; y=[0.1, 0.2]) begin
            x ~ normal(0.0, 1.0)
            y[1] ~ normal(inline_offset(x), 1.0)
            y[2] ~ normal(sibling_offset(x), 1.0)
        end
        const defaulted = @slic (; y=0.1) begin
            x ~ normal(0.0, 1.0)
            y ~ normal(inline_default(x), 1.0)
        end
        const explicit = @slic (; y=0.1) begin
            x ~ normal(0.0, 1.0)
            y ~ normal(inline_default(x; shift=2.0), 1.0)
        end
        const explicit_call = @slic (; y=0.1) begin
            x ~ normal(0.0, 1.0)
            y ~ normal(inline_default(x; shift=sibling_offset(0.0)), 1.0)
        end
        const inner = @slic begin
            x ~ normal(0.0, 1.0)
            return inline_offset(x)
        end
        const subcalled = @slic (; y=0.1) begin
            mu ~ inner()
            y ~ normal(mu, 1.0)
        end
        const extended = @slic (; y=0.1) begin
            x ~ normal(0.0, 1.0)
            y ~ normal(extended_inline(x), 1.0)
        end
    end
end

@testitem "slic: inline helpers retain their definition module across calls" tags=[:slic, :regression, :stanc, :bridgestan] setup=[StanBlocksImports, StanBlocksTestSetup, InlineDefinitionScopeFixtures] begin
    using .StanBlocksTestSetup: stanc_compiles
    using .InlineDefinitionScopeFixtures: Caller
    normal_lp(y, mu) = -0.5 * (y - mu)^2 - 0.5 * log(2pi)
    for name in (:direct, :wrapped, :after_inline, :defaulted, :explicit, :explicit_call, :subcalled, :extended)
        @testset "$name" begin
            model = getproperty(Caller, name)
            @test stanc_compiles(model)
            p = instantiate(model)
            shift = name === :explicit ? 2.0 : name === :explicit_call ? 100.0 : 1.0
            for x in (-0.4, 0.3, 0.9)
                expected = normal_lp(x, 0.0) + normal_lp(0.1, x + shift)
                expected_grad = -x + 0.1 - (x + shift)
                if name === :after_inline
                    expected += normal_lp(0.2, x + 100.0)
                    expected_grad += 0.2 - (x + 100.0)
                end
                point = [x]
                lp, grad = BridgeStan.log_density_gradient(p.model, point; propto=false, jacobian=false)
                @test lp ≈ expected
                @test grad ≈ [expected_grad]
                @test point == [x]
            end
        end
    end
end
