using Pkg
Pkg.activate(@__DIR__)
using LinearAlgebra
using Plots, LaTeXStrings
using ForwardDiff

FIGS_PATH = joinpath(@__DIR__, "figs")

R, R2 = 1, 2
G1(delta; R=R, R2=R2) = -R*R2/(R + delta) + R2
g1(delta; R=R, R2=R2) = R*R2/(R + delta)^2
G1inv(Δ; R=R, R2=R2) = -(R*R2)/(R2 + Δ) + R
function Gsand(delta, η; R=R, R2=R2)
    @inline Δsand(η, Δ, R) = (-(Δ + 2*R) + sqrt((Δ + 2*R)^2 - 4*(R^2 + R*Δ)*(-η/(1-η))))/2
    Δs = Δsand(η, delta, R)
    Δs_out = G1(Δs; R=R, R2=R2)
    return G1(delta; R=R + Δs, R2=R2 - Δs_out)
end
G2(x) = x


## Generate output figure
Δ = range(0, 1, length=101)
forward_ex_plt = plot(Δ, G1.(Δ),
    label=L"$G_1(\Delta)$",
    ylabel=L"Output $G(\Delta)$",
    xlabel=L"Trade Amount $\Delta$",
    legend=:right,
    lw=4,
    color=:mediumblue,
    dpi=300
)

η1 =  0.1
plot!(forward_ex_plt, Δ, Gsand.(Δ, η1),
    label=L"$G_1^\mathrm{sand}(\Delta, %$η1)$",
    lw=3,
    color=:mediumblue,
    ls=:dash,
    dpi=300
)

η2 =  0.2
plot!(forward_ex_plt, Δ, Gsand.(Δ, η2),
    label=L"$G_1^\mathrm{sand}(\Delta, %$η2)$",
    lw=3,
    color=:mediumblue,
    ls=:dot,
    dpi=300
)

# η3 =  0.5
# plot!(forward_ex_plt, Δ, Gsand.(Δ, η3),
#     label=L"$G_1^\mathrm{sand}(\Delta, %$η3)$",
#     lw=3,
#     color=:mediumblue,
#     ls=:dashdotdot,
#     dpi=300
# )

plot!(forward_ex_plt, Δ, G2.(Δ),
    label=L"$G_2(\Delta)$",
    lw=4,
    color=:firebrick,
    dpi=300
)

savefig(forward_ex_plt, joinpath(FIGS_PATH, "forward-ex-pigou.pdf"))


## Average price plot
Δ = range(0, 1, length=101)
avg_price_plt = plot(Δ, G1.(Δ) ./ Δ,
    label=L"$G_1(\Delta) / \Delta$",
    ylabel=L"Average price $G(\Delta) / \Delta$",
    xlabel=L"Trade Amount $\Delta$",
    legend=:right,
    lw=4,
    color=:mediumblue,
    dpi=300
)

η1 =  0.1
plot!(avg_price_plt, Δ, Gsand.(Δ, η1) ./ Δ,
    label=L"$G_1^\mathrm{sand}(\Delta, %$η1) / \Delta$",
    lw=3,
    color=:mediumblue,
    ls=:dash,
    dpi=300
)

η2 =  0.2
plot!(avg_price_plt, Δ, Gsand.(Δ, η2) ./ Δ,
    label=L"$G_1^\mathrm{sand}(\Delta, %$η2) / \Delta$",
    lw=3,
    color=:mediumblue,
    ls=:dot,
    dpi=300
)

plot!(avg_price_plt, Δ, G2.(Δ) ./ Δ,
    label=L"$G_2(\Delta) / \Delta$",
    lw=4,
    color=:firebrick,
    dpi=300
)

savefig(avg_price_plt, joinpath(FIGS_PATH, "avg-price-pigou.pdf"))


## Compute optimal and equilibrium

# Eq: G1(Δ1)/Δ1 = G2(Δ2)/Δ2
# => find a root of G1(x)/x - 1 = 0 ⟹ G1(x) = x
function compute_equilibrium(η)
    iszero(η) && return 1.0
    @inline f(x) = Gsand(x, η) - x

    # Find a root of f(x) = 0
    l, u = eps(), 1-eps()
    for t in 1:100
        m = (l + u) / 2
        if f(m) < 0
            u = m
        else
            l = m
        end
    end
    return (l + u) / 2
end

function compute_optimal(η)
    @inline f(x) = ForwardDiff.derivative(x->Gsand(x, η), x) - 1

    # Find a root of f(x) = 0
    l, u = eps(), 1-eps()
    for t in 1:100
        m = (l + u) / 2
        if f(m) < 0
            u = m
        else
            l = m
        end
    end
    return (l + u) / 2
end

ηs = range(0.0, 0.5, length=41)
xs_eq = compute_equilibrium.(ηs)
outs_eq = Gsand.(xs_eq, ηs) + G2.(1 .- xs_eq)

xs_opt = compute_optimal.(ηs)
outs_opt = Gsand.(xs_opt, ηs) + G2.(1 .- xs_opt)

output_plt = plot(ηs, 
    ones(length(ηs)),
    label="Equilibrium",
    lw=4,
    color=:firebrick,
    ylabel="Total Output",
    xlabel=L"Slippage Tolerance $\eta$",
    legend=:topright,
    dpi=300
)
plot!(output_plt, ηs, outs_opt,
    label="Optimal",
    lw=4,
    color=:mediumblue,
)
savefig(output_plt, joinpath(FIGS_PATH, "output-pigou.pdf"))

proportion_plt = plot(
    ηs,
    [xs_eq xs_opt],
    label=[L"Equilibrium $x$" L"Optimal $x$"],
    ylabel=L"Fraction $x$ on $G_1(x)$",
    xlabel=L"Slippage Tolerance $\eta$",
    lw=3,
    ls=:dash,
    color=[:firebrick :mediumblue],
    dpi=300,
)
savefig(proportion_plt, joinpath(FIGS_PATH, "proportion-pigou.pdf"))


function compute_pnl_sand(η, input; R=R, R2=R2)
    @inline Δsand(η, Δ, R) = (-(Δ + 2*R) + sqrt((Δ + 2*R)^2 - 4*(R^2 + R*Δ)*(-η/(1-η))))/2

    Δs = Δsand(η, input, R)
    Δs_out = G1(Δs; R=R, R2=R2)
    Δ_out = G1(input; R=R + Δs, R2=R2 - Δs_out)
    return G1inv(Δs_out; R=R + Δs + input, R2=R2 - Δs_out - Δ_out) - Δs
end

pnl_plt = plot(
    ηs, compute_pnl_sand.(ηs, xs_eq),
    label="Equilibrium",
    ylabel="Sandwicher Profit",
    xlabel=L"Slippage Tolerance $\eta$",
    legend=:topright,
    lw=4,
    color=:firebrick,
    dpi=300,
)

plot!(pnl_plt, ηs, compute_pnl_sand.(ηs, xs_opt),
    label="Optimal",
    lw=4,
    color=:mediumblue,
)

savefig(pnl_plt, joinpath(FIGS_PATH, "pnl-pigou.pdf"))