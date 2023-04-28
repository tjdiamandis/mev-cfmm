using Pkg
Pkg.activate(@__DIR__)

## Routing with middle link removed
using LinearAlgebra, SparseArrays
using JuMP, Ipopt
using ForwardDiff: derivative 

R, Rp = 1, 2
G1(x) = sqrt(x)
G2(x) = x
G3(x) = x
G4(x) = sqrt(x)
@inline G5(delta; R=R, Rp=Rp) = -R*Rp/(R + delta) + Rp
@inline function Gsand(delta, η; R=R, R2=Rp)
    @inline Δsand(η, Δ, R) = (-(Δ + 2*R) + sqrt((Δ + 2*R)^2 - 4*(R^2 + R*Δ)*(-η/(1-η))))/2
    @inline G(R, R2, Δ) = -(R*R2)/(R + Δ) + R2
    Δs = Δsand(η, delta, R)
    Δs_out = G(R, R2, Δs)
    return G(R + Δs, R2 - Δs_out, delta)
end

# Returns output of route
@inline function route1(α1, α2, α3, η)
    out = G2(G1(α1 + α3) * α1/(α1 + α3))
    return out
end
@inline function route2(α1, α2, α3, η)
    out_r2 = G3(α2)
    out_r3 = Gsand(G1(α1 + α3) * α3/(α1 + α3), η)
    out = G4(out_r2 + out_r3) * out_r2 / (out_r2 + out_r3)
    return out
end

@inline function route3(α1, α2, α3, η)
    out_r3 = Gsand(G1(α1 + α3) * α3/(α1 + α3), η)
    out_r2 = G3(α2)
    out = G4(out_r3 + out_r2) * out_r3 / (out_r3 + out_r2)
    return out
end

# Returns average price
@inline function route_output(αi, i, α1, α2, α3, η)
    if i == 1
        return route_output(i, αi, α2, α3, η)
    elseif i == 2
        return route_output(i, α1, αi, α3, η)
    elseif i == 3
        return route_output(i, α1, α2, αi, η)
    end
end
@inline function route_output(i, α1, α2, α3, η)
    if i == 1
        return route1(α1, α2, α3, η)
    elseif i == 2
        return route2(α1, α2, α3, η)
    elseif i == 3
        return route3(α1, α2, α3, η)
    end
    error("Invalid route index")
end

@inline function route_price(i, α, η)
    if α[i] > 1e-4
        return route_output(i, α..., η) / α[i]
    else
        return derivative(αi->route_output(αi, i, α..., η), sqrt(eps()))
    end
end

function compute_optimal(η; middle=true, print_level=0)
    model = Model(Ipopt.Optimizer)
    set_optimizer_attribute(model, "print_level", print_level)
    @variable(model, λ[1:5] ≥ 0)
    @variable(model, δ[1:5] ≥ 0)

    # Forward exchange functions G₁, …, G₄ 
    @NLconstraint(model, sqrt(δ[1]) == λ[1])
    @constraint(model, δ[2] == λ[2])
    @constraint(model, δ[3] == λ[3])
    @NLconstraint(model, sqrt(δ[4]) == λ[4])
    
    if middle 
        @NLparameter(model, ηp == η)
        register(model, :Gsand, 2, Gsand; autodiff = true)
        @NLconstraint(model, Gsand(δ[5], ηp) == λ[5])
    else
        @constraint(model, δ[5] == 0)
        @constraint(model, λ[5] == 0)
    end

    # Flow conservation constraints
    @constraint(model, λ[1] == δ[2] + δ[5])
    @constraint(model, λ[3] + λ[5] == δ[4])
    @constraint(model, δ[1] + δ[3] == 1)

    # Objective: output
    @objective(model, Max, λ[2] + λ[4])
    optimize!(model)

    δv = value.(δ)
    α = [δv[1] * δv[2]/(δv[2] + δv[5]), δv[3], δv[1] * δv[5]/(δv[2] + δv[5])]

    return model, objective_value(model), α, δv[5]
end

function compute_selfish(η; α0=nothing, tol=1e-7)
    function is_equal(α; tol=1e-5)
        r = [route_price(i, α, η) for i in 1:3]
        rmax = maximum(r)
        return all(isapprox.(r[α .> tol], rmax, atol=tol))
    end

    function reallocate!(α, imax, imin; tol=1e-7)
        # f(x) = max_price - min_price
        function f(x, α)
            αp = copy(α)
            αp[imin] -= x
            αp[imax] += x
            return route_price(imax, αp, η) - route_price(imin, αp, η)
        end

        # Want to find the largest x such that f(x) > 0
        l, u = eps(), α[imin]/8-eps()
        for t in 1:20
            m = (l + u) / 2
            if f(m, α) < -tol
                u = m
            elseif f(m, α) > tol
                l = m
            else
                break
            end
        end
        x = (l + u) / 2
        α[imin] -= x
        α[imax] += x
        return nothing
    end

    α = isnothing(α0) ? ones(3) / 3 : α0
    iter = 1
    while !is_equal(α)
        _, imax = findmax(i -> route_price(i, α, η), 1:3)
        _, imin = findmin(i -> α[i] > tol ? route_price(i, α, η) : Inf, 1:3)

        reallocate!(α, imax, imin)
        
        # route_prices = [round.(route_price(i, α, η), digits=3) for i in 1:3]
        # αr = round.(α, digits=3)
        # @show αr, route_prices
        iter += 1
        iter > 20_000 && (@warn "Did not converge on $η"; @show imax, imin; break;)
    end

    out = route_output(1, α..., η) + route_output(2, α..., η) + route_output(3, α..., η)
    δ5 = G1(α[1] + α[3]) * α[3]/(α[1] + α[3])

    return out, α, δ5
end

function compute_pnl_sand(η, δ5; R=1, R2=1)
    input = δ5
    @inline Δsand(η, Δ, R) = (-(Δ + 2*R) + sqrt((Δ + 2*R)^2 - 4*(R^2 + R*Δ)*(-η/(1-η))))/2
    @inline G(Δ, R, R2) = -(R*R2)/(R + Δ) + R2
    @inline Ginv(Δ, R, R2) = -(R*R2)/(R2 + Δ) + R
    
    Δs = Δsand(η, input, R)
    Δs_out = G(Δs, R, R2)
    Δ_out = G(input, R + Δs, R2 - Δs_out)
    return Ginv(Δs_out, R + Δs + input, R2 - Δs_out - Δ_out) - Δs
end


ηs = range(0.0, 0.75, 76) 
δ5s = zeros(length(ηs), 2)
αs = zeros(length(ηs), 2, 3)
optvals = zeros(length(ηs), 2) 
pnl = zeros(length(ηs), 2)
α_old = nothing
for (i, η) in enumerate(ηs)
    model_opt, pstar_opt, αopt, δ5star_opt = compute_optimal(η)
    pstar_eq, αeq, δ5star_eq =  compute_selfish(η)
    if !(termination_status(model_opt) in [MOI.LOCALLY_SOLVED, MOI.ALMOST_LOCALLY_SOLVED])
        @warn "Error in optimal route for η = $η\n status=$(termination_status(model_opt))"
    end

    optvals[i, 1] = pstar_eq
    optvals[i, 2] = pstar_opt
    δ5s[i, 1] = δ5star_eq
    δ5s[i, 2] = δ5star_opt
    αs[i, 1, :] .= αeq
    αs[i, 2, :] .= αopt
    pnl[i, 1] = compute_pnl_sand(η, δ5star_eq)
    pnl[i, 2] = compute_pnl_sand(η, δ5star_opt)
    # α_old = αeq

    rp = [(αeq[i] > 1e-2 ? round(route_price(i, αeq, η), digits=4) : NaN) for i in 1:3]
    @info "Finished i = $i, η = $η\n\t prices = $rp\n\t α = $(round.(αeq, digits=3))"
end

## Plots
using Plots, LaTeXStrings
output_plt = plot(ηs, optvals,
    label=["Equilibrium" "Optimal"],
    lw=3,
    ylabel="Total output",
    xlabel=L"Slippage tolerance $\eta$",
    legend=:bottomright,
    color=[:firebrick :mediumblue],
    dpi=300,
)

optvals_no_sand = sqrt(2) * ones(length(ηs))
hline!(output_plt, 
    optvals_no_sand,
    lw=2,
    ls=:dash,
    label="No middle route",
    color=:black,
)

poa_plt = plot(
    ηs,
    ones(length(ηs)),
    lw=1,
    ls=:dash,
    color=:black,
    ylabel="Price of Anarchy",
    xlabel=L"Slippage tolerance $\eta$",
    legend=false,
)
plot!(poa_plt, ηs, optvals[:, 2] ./ optvals[:, 1],
    lw=4,
    dpi=300,
    color=:coral
)

middle_fraction_plt = plot(ηs, [αs[:, 1, 3] αs[:, 2, 3]],
    label=["Equilibrium" "Optimal"],
    ylabel="Middle route fraction",
    xlabel=L"Slippage tolerance $\eta$",
    legend=:right,
    color=[:firebrick :mediumblue],
    lw=3,
    dpi=300,
)

pnl_plt = plot(ηs, pnl,
    label=["Equilibrium" "Optimal"],
    ylabel="Sandwicher PnL",
    xlabel=L"Slippage tolerance $\eta$",
    legend=:topright,
    color=[:firebrick :mediumblue],
    lw=3,
    dpi=300,
)


FIGS_PATH = joinpath(@__DIR__, "figs")
savefig(output_plt, joinpath(FIGS_PATH, "output-routing.pdf"))
savefig(middle_fraction_plt, joinpath(FIGS_PATH, "middle-fraction-routing.pdf"))
savefig(pnl_plt, joinpath(FIGS_PATH, "pnl-routing.pdf"))
savefig(poa_plt, joinpath(FIGS_PATH, "poa-routing.pdf"))