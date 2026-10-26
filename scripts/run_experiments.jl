# ==============================================================================
# run_experiments.jl
# Computes all numerical results for the AI-compute-token swing-option paper and
# writes them to results/experiments.json (via a minimal JSON encoder).
# ==============================================================================
using Pkg; Pkg.activate(joinpath(@__DIR__, ".."))
include(joinpath(@__DIR__, "..", "src", "AITokenSwing.jl"))
using .AITokenSwing
using Random, Printf, Statistics

# ---- minimal JSON encoder (no external dependency) ----
jstr(x::Bool) = x ? "true" : "false"
jstr(x::Integer) = string(x)
jstr(x::AbstractFloat) = (isnan(x) ? "\"nan\"" : isinf(x) ? (x > 0 ? "\"inf\"" : "\"-inf\"") : string(x))
jstr(x::String) = "\"" * replace(x, "\\" => "\\\\", "\"" => "\\\"") * "\""
jstr(x::Symbol) = jstr(String(x))
function jstr(x::AbstractVector)
    "[" * join((jstr(v) for v in x), ", ") * "]"
end
function jstr(x::AbstractMatrix)
    "[" * join(("[" * join((jstr(x[i, j]) for j in axes(x, 2)), ", ") * "]" for i in axes(x, 1)), ", ") * "]"
end
function jstr(x::AbstractDict)
    "{" * join(("\"" * string(k) * "\": " * jstr(v) for (k, v) in x), ", ") * "}"
end

# ==============================================================================
# Base-case parameters (documented)
# ==============================================================================
p  = ExpOUJumpParams(0.50, 2.0, 0.50, 2.0, 0.0, 0.25)
c  = ContractSpec(1.0, 12, 0.0, 2.0, 6.0, 20.0, 5.0)
ag = AgentParams(0.3, 0.3, 0.5, 0.5, 20.0, 0.5)
x0 = log(7.39)   # S0 = 7.39

res = Dict{String,Any}()

res["base_params"] = Dict(
    "kappa" => p.kappa, "theta" => p.theta, "sigma" => p.sigma,
    "lambda" => p.lambda, "muJ" => p.muJ, "sigmaJ" => p.sigmaJ,
    "T" => c.T, "N" => c.Nsteps, "qmin" => c.qmin, "qmax" => c.qmax,
    "Umin" => c.Umin, "Umax" => c.Umax, "fee" => c.fee,
    "gammaB" => ag.gammaB, "gammaS" => ag.gammaS, "alphaB" => ag.alphaB,
    "cs" => ag.cs, "Kmax" => ag.Kmax, "alphaNash" => ag.alphaNash,
    "S0" => exp(x0), "x0" => x0, "ln_fee" => log(c.fee))

println("== 1. Reference (high-resolution DCT) ==")
ref = reservation_prices(p, c, ag; Nx=1024, Nu=256, nq=256, L=8.0, x0=x0)
res["reference"] = Dict("P_B" => ref.P_B, "P_S" => ref.P_S, "P_star" => ref.P_star)
println("   P_B_ref = ", ref.P_B, "  P_S_ref = ", ref.P_S)

# ==============================================================================
# 2. DCT convergence in Nx
# ==============================================================================
println("== 2. DCT convergence ==")
Nx_list = [64, 128, 256, 512, 1024]
conv = Dict("Nx" => Nx_list, "P_B" => Float64[], "P_S" => Float64[], "P_star" => Float64[])
for Nx in Nx_list
    r = reservation_prices(p, c, ag; Nx=Nx, Nu=256, nq=256, L=8.0, x0=x0)
    push!(conv["P_B"], r.P_B); push!(conv["P_S"], r.P_S); push!(conv["P_star"], r.P_star)
    println("   Nx=", Nx, "  P_B=", round(r.P_B, digits=5), "  P_S=", round(r.P_S, digits=5))
end
res["conv_dct"] = conv

# ==============================================================================
# 3. LSM convergence (5 seeds) and accuracy
# ==============================================================================
println("== 3. LSM convergence ==")
np_list = [10_000, 20_000, 50_000, 100_000]
lsm = Dict("npath" => np_list, "mean" => Float64[], "std" => Float64[], "time" => Float64[])
for np in np_list
    vals = Float64[]; tt = 0.0
    for seed in 1:5
        t0 = time()
        v = lsm_reservation_price(p, c, ag; npath=np, Nu=64, nq=32, deg=5, x0=x0, rng=Xoshiro(seed))
        tt += time() - t0
        push!(vals, v)
    end
    push!(lsm["mean"], mean(vals)); push!(lsm["std"], std(vals)); push!(lsm["time"], tt / 5)
    println("   npath=", np, "  P_B=", round(mean(vals), digits=4), " ± ", round(std(vals), digits=4), "  (", round(tt/5, digits=2), " s)")
end
res["conv_lsm"] = lsm

# ==============================================================================
# 4. Timing + allocations (DCT)
# ==============================================================================
println("== 4. Timing / allocation ==")
timing = Dict("Nx" => Int[], "time" => Float64[], "alloc_step" => Float64[])
for Nx in [128, 256, 512, 1024]
    b = bench_dct(p, c, ag; Nx=Nx, Nu=128, nq=128, L=8.0, x0=x0)
    push!(timing["Nx"], Nx); push!(timing["time"], b.time); push!(timing["alloc_step"], b.per_step_alloc)
    println("   Nx=", Nx, "  time=", round(b.time, digits=4), " s  alloc/step=", b.per_step_alloc, " B")
end
res["timing"] = timing

# ==============================================================================
# 5. Free boundary at t_0 (DCT vs LSM, 3 seeds)
# ==============================================================================
println("== 5. Free boundary ==")
sol = solve_buyer_dct(p, c, ag; Nx=512, Nu=256, nq=256, L=8.0, x0=x0)
ugrid_b = collect(range(0.0, 14.0, length=15))    # u up to 14 (boundary above that is cap-limited)
du256 = c.Umax / 255

function extract_dct_boundary(sol, ugrid, step)
    xb = Float64[]
    for u in ugrid
        j0 = clamp(round(Int, u / (c.Umax / (256 - 1))) + 1, 1, 256)
        xbb = Inf
        for i in 1:512
            if sol.policy[i, j0, step] > 1e-12
                xbb = sol.gridX[i]; break
            end
        end
        push!(xb, xbb)
    end
    return xb
end

xb_dct_t0 = extract_dct_boundary(sol, ugrid_b, 1)   # t_0 boundary (main result)
xb_dct_t1 = extract_dct_boundary(sol, ugrid_b, 2)   # t_1 boundary (for LSM comparison)
res["boundary"] = Dict("u" => ugrid_b, "xbar_dct" => xb_dct_t0, "xbar_dct_t1" => xb_dct_t1)
println("   DCT t_0 boundary: ", round.(xb_dct_t0, digits=3))

xgrid_b = collect(range(-1.0, 3.0, length=400))
lsm_bounds = Dict{String,Any}()
for seed in (7, 21, 99)
    xb = lsm_free_boundary(p, c, ag, xgrid_b, ugrid_b; npath=80_000, Nu=64, nq=32, deg=5, x0=x0, rng=Xoshiro(seed))
    lsm_bounds["seed_$seed"] = xb
    println("   LSM seed $seed boundary: ", round.(xb, digits=3))
end
res["boundary_lsm"] = lsm_bounds
res["boundary_xgrid"] = xgrid_b

# ==============================================================================
# 6. Free-boundary time dependence (DCT) at u = 0 and u = 3
# ==============================================================================
println("== 6. Free-boundary time dependence ==")
steps_plot = [1, 4, 8, 12]
bd_time = Dict("step" => steps_plot, "u0" => Float64[], "u3" => Float64[])
du256 = c.Umax / 255
for st in steps_plot
    # u = 0 -> j=1 ; u = 3 -> j ~ 3/du256 + 1
    for (key, uu) in (("u0", 0.0), ("u3", 3.0))
        jj = clamp(round(Int, uu / du256) + 1, 1, 256)
        xb = Inf
        for i in 1:512
            if sol.policy[i, jj, st] > 1e-12
                xb = sol.gridX[i]; break
            end
        end
        push!(bd_time[key], xb)
    end
end
res["boundary_time"] = bd_time
println("   done; t_0 boundary u=0: ", bd_time["u0"][1], "  u=3: ", bd_time["u3"][1])

# ==============================================================================
# 7. Sensitivity analysis
# ==============================================================================
println("== 7. Sensitivity ==")
sens = Dict{String,Any}()

# fee sweep
fee_list = [3.0, 4.0, 5.0, 6.0, 7.0]
fee_res = Dict("fee" => fee_list, "P_B" => Float64[], "P_S" => Float64[], "P_star" => Float64[])
for f in fee_list
    cc = ContractSpec(c.T, c.Nsteps, c.qmin, c.qmax, c.Umin, c.Umax, f)
    r = reservation_prices(p, cc, ag; Nx=512, Nu=256, nq=256, L=8.0, x0=x0)
    push!(fee_res["P_B"], r.P_B); push!(fee_res["P_S"], r.P_S); push!(fee_res["P_star"], r.P_star)
end
sens["fee"] = fee_res

# gamma_B sweep
gB_list = [0.1, 0.3, 0.5, 0.8]
gB_res = Dict("gammaB" => gB_list, "P_B" => Float64[], "P_star" => Float64[])
for g in gB_list
    aa = AgentParams(g, ag.gammaS, ag.alphaB, ag.cs, ag.Kmax, ag.alphaNash)
    r = reservation_prices(p, c, aa; Nx=512, Nu=256, nq=256, L=8.0, x0=x0)
    push!(gB_res["P_B"], r.P_B); push!(gB_res["P_star"], r.P_star)
end
sens["gammaB"] = gB_res

# jump intensity sweep
lam_list = [0.5, 1.0, 2.0, 4.0]
lam_res = Dict("lambda" => lam_list, "P_B" => Float64[], "P_S" => Float64[])
for l in lam_list
    pp = ExpOUJumpParams(p.kappa, p.theta, p.sigma, l, p.muJ, p.sigmaJ)
    r = reservation_prices(pp, c, ag; Nx=512, Nu=256, nq=256, L=8.0, x0=x0)
    push!(lam_res["P_B"], r.P_B); push!(lam_res["P_S"], r.P_S)
end
sens["lambda"] = lam_res

# jump-size vol sweep
sJ_list = [0.1, 0.25, 0.5, 0.75]
sJ_res = Dict("sigmaJ" => sJ_list, "P_B" => Float64[], "P_S" => Float64[])
for s in sJ_list
    pp = ExpOUJumpParams(p.kappa, p.theta, p.sigma, p.lambda, p.muJ, s)
    r = reservation_prices(pp, c, ag; Nx=512, Nu=256, nq=256, L=8.0, x0=x0)
    push!(sJ_res["P_B"], r.P_B); push!(sJ_res["P_S"], r.P_S)
end
sens["sigmaJ"] = sJ_res

# bargaining power sweep
alpha_list = [0.0, 0.25, 0.5, 0.75, 1.0]
r0 = reservation_prices(p, c, ag; Nx=512, Nu=256, nq=256, L=8.0, x0=x0)
sens["alpha"] = Dict("alpha" => alpha_list,
    "P_star" => [nash_price(r0.P_S, r0.P_B, a) for a in alpha_list],
    "P_B" => r0.P_B, "P_S" => r0.P_S)

res["sensitivity"] = sens

# ==============================================================================
# 8. Value-function surface (for Figure 2) at t_0
# ==============================================================================
println("== 8. Value function surface ==")
# downsampled grid for the figure
idx_x = collect(range(1, 512, length=100)) .|> round .|> Int
idx_u = collect(range(1, 256, length=60))  .|> round .|> Int
Vsurf = [log(max(sol.V[ix, iu], 1e-16)) for ix in idx_x, iu in idx_u]   # log-value for plotting
res["surface"] = Dict(
    "x" => sol.gridX[idx_x],
    "u" => [(j - 1) * du256 for j in idx_u],
    "logV" => Vsurf)

# ==============================================================================
# Write JSON
# ==============================================================================
mkpath("../results")
open("../results/experiments.json", "w") do io
    write(io, jstr(res))
end
println("\nWrote results/experiments.json")
println("SUMMARY:  P_B_ref=", round(ref.P_B, digits=4),
        "  P_S_ref=", round(ref.P_S, digits=4),
        "  P*=", round(ref.P_star, digits=4))
