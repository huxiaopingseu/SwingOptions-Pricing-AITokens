"""
    AITokenSwing.jl

Bilateral utility-indifference pricing of AI-compute-token swing contracts under an
Exp-OU + mean-reverting jump-diffusion (MRJD) spot process, in the physical (P) measure.

Core numerical method: Fourier-Cosine (COS) expansion implemented via the Discrete
Cosine Transform (DCT-II / DCT-III), with a precomputed dense reconstruction matrix that
accounts for the affine (mean-reverting) drift.  Zero-heap-allocation hot loops.

Reference baseline: grid-based Least-Squares Monte Carlo (LSM).
"""
module AITokenSwing

using FFTW
using LinearAlgebra
using Statistics
using Random
using Printf

# Knuth's Poisson sampler (avoids an extra dependency)
@inline function poisson_rand(rng::AbstractRNG, lambda::Float64)
    lambda <= 0.0 && return 0
    L = exp(-lambda)
    k = 0
    p = 1.0
    while true
        p *= rand(rng)
        if p <= L
            return k
        end
        k += 1
    end
end

export ExpOUJumpParams, ContractSpec, AgentParams,
       truncation_domain, log_charfun,
       solve_buyer_dct, solve_seller_dct, solve_spot_dct, seller_ratio,
       reservation_prices, nash_price,
       free_boundary, free_boundary_grid,
       solve_buyer_lsm, lsm_reservation_price, lsm_free_boundary,
       simulate_X, bench_dct

# ---------------------------------------------------------------------------
# Parameter containers
# ---------------------------------------------------------------------------
struct ExpOUJumpParams
    kappa::Float64    # mean-reversion speed
    theta::Float64    # long-run log-spot level
    sigma::Float64    # diffusion volatility
    lambda::Float64   # Poisson jump intensity
    muJ::Float64      # jump-size mean
    sigmaJ::Float64   # jump-size volatility
end

struct ContractSpec
    T::Float64        # maturity (years)
    Nsteps::Int       # number of exercise periods (decision dates t_0..t_{N-1})
    qmin::Float64     # minimum delivery quantum per period (>= 0)
    qmax::Float64     # maximum delivery quantum per period
    Umin::Float64     # take-or-pay minimum cumulative quantity
    Umax::Float64     # global upper bound on cumulative quantity
    fee::Float64      # per-token contract fee  c
end

struct AgentParams
    gammaB::Float64   # buyer  risk aversion
    gammaS::Float64   # seller risk aversion
    alphaB::Float64   # take-or-pay quadratic penalty coefficient (buyer)
    cs::Float64       # seller variable cost per token
    Kmax::Float64     # seller capacity per period
    alphaNash::Float64# buyer bargaining power in [0,1]
end

# ---------------------------------------------------------------------------
# Characteristic function  phi(w) = E_P[ exp(i w X_{t+dt}) | X_t = x ]
# ---------------------------------------------------------------------------
@inline function log_charfun(w::Float64, dt::Float64, p::ExpOUJumpParams, x::Float64)
    emk = exp(-p.kappa * dt)
    re  = -(p.sigma^2 * w^2 / (4.0 * p.kappa)) * (1.0 - exp(-2.0 * p.kappa * dt))
    re += p.lambda * dt * (exp(-0.5 * p.sigmaJ^2 * w^2) * cos(w * p.muJ) - 1.0)
    im_ = w * (x * emk + p.theta * (1.0 - emk))
    im_ += p.lambda * dt * exp(-0.5 * p.sigmaJ^2 * w^2) * sin(w * p.muJ)
    return Complex(re, im_)
end

# ---------------------------------------------------------------------------
# Truncation domain [a,b] from cumulants (Fang-Oosterlee style)
# ---------------------------------------------------------------------------
function truncation_domain(p::ExpOUJumpParams, T::Float64, x0::Float64; L::Float64=8.0)
    emk  = exp(-p.kappa * T)
    c1   = x0 * emk + p.theta * (1.0 - emk) + p.lambda * T * p.muJ
    c2   = (p.sigma^2 / (2.0 * p.kappa)) * (1.0 - exp(-2.0 * p.kappa * T))
    c2  += p.lambda * T * (p.muJ^2 + p.sigmaJ^2)
    c4   = p.lambda * T * (p.muJ^4 + 6.0 * p.muJ^2 * p.sigmaJ^2 + 3.0 * p.sigmaJ^4)
    half = L * sqrt(c2 + sqrt(max(c4, 0.0)))
    return c1 - half, c1 + half
end

# ---------------------------------------------------------------------------
# COS/DCT expectation operator
# ---------------------------------------------------------------------------
#  Reconstruction matrix A[j,m] = (1/N)*(m==0 ? 1/2 : 1) * Re{ phi(w_m; x_j) e^{-i w_m a} }
#  such that  E[V(X_{t+dt}) | X_t = x_j] ~= sum_m A[j,m] * (DCT-II(V))_m.
struct CosExpect
    A::Matrix{Float64}          # N x N reconstruction matrix
    plan::FFTW.r2rFFTWPlan      # DCT-II plan (REDFT10)
    buf::Vector{Float64}        # scratch, length N
    out::Vector{Float64}        # scratch, length N
    N::Int
end

function CosExpect(p::ExpOUJumpParams, dt::Float64, a::Float64, b::Float64, N::Int)
    dx  = (b - a) / N
    A   = zeros(Float64, N, N)
    w0  = pi / (b - a)
    for m in 0:N-1
        w = w0 * m
        for j in 0:N-1
            xj = a + (j + 0.5) * dx
            z  = exp(log_charfun(w, dt, p, xj)) * exp(-im * w * a)
            A[j+1, m+1] = real(z) / N
        end
    end
    A[:, 1] .*= 0.5
    buf = zeros(Float64, N)
    plan = FFTW.plan_r2r(buf, FFTW.REDFT10)
    return CosExpect(A, plan, buf, zeros(Float64, N), N)
end

# apply expectation to a single column vector v (length N), writing into out (length N)
function expect_col!(ce::CosExpect, v::AbstractVector{Float64}, out::AbstractVector{Float64})
    mul!(ce.buf, ce.plan, v)          # DCT-II coefficients
    mul!(out, ce.A, ce.buf)           # dense reconstruction
    return out
end

# ---------------------------------------------------------------------------
# Buyer backward induction (COS/DCT)
# ---------------------------------------------------------------------------
struct BuyerSolution
    V::Matrix{Float64}               # V_tilde at t_0, size (N_x, N_u)
    gridX::Vector{Float64}
    gridU::Vector{Float64}
    policy::Array{Float64,3}         # q* per (x,u,step)  [N_x, N_u, N_steps]
    price::Float64
end

function solve_buyer_dct(p::ExpOUJumpParams, c::ContractSpec, ag::AgentParams;
                         Nx::Int=256, Nu::Int=128, nq::Int=128, L::Float64=8.0,
                         x0::Float64=0.0)
    dt  = c.T / c.Nsteps
    a, b = truncation_domain(p, c.T, x0; L=L)
    ce  = CosExpect(p, dt, a, b, Nx)

    gridX = [a + (j + 0.5) * ((b - a) / Nx) for j in 0:Nx-1]
    du    = c.Umax / (Nu - 1)
    gridU = [(j - 1) * du for j in 1:Nu]
    Sgrid = exp.(gridX)

    # fixed q-grid for the per-period control
    qgrid = range(c.qmin, c.qmax, length=nq) |> collect

    # reward factors: exp(-gammaB * (S - fee) * q)  (precomputed once)
    expfac = zeros(Float64, Nx, nq)
    @inbounds for i in 1:Nx, l in 1:nq
        expfac[i, l] = exp(-ag.gammaB * (Sgrid[i] - c.fee) * qgrid[l])
    end

    # number of feasible q points per u-level (q <= Umax - u)
    nfeas = zeros(Int, Nu)
    @inbounds for j in 1:Nu
        u = gridU[j]
        k = nq
        while k >= 1 && qgrid[k] > c.Umax - u
            k -= 1
        end
        nfeas[j] = max(k, 0)
    end

    # interpolation tables for u_j + q_l (independent of time step)
    j0tab = zeros(Int, Nu, nq)
    ftab  = zeros(Float64, Nu, nq)
    @inbounds for j in 1:Nu, l in 1:nq
        idx = (gridU[j] + qgrid[l]) / du + 1.0
        j0 = clamp(trunc(Int, idx), 1, Nu - 1)
        j0tab[j, l] = j0
        ftab[j, l]  = idx - j0
    end

    # terminal value
    Vnext = zeros(Float64, Nx, Nu)
    @inbounds for j in 1:Nu
        u = gridU[j]
        pen = c.Umin - u
        v = pen > 0.0 ? exp(ag.gammaB * ag.alphaB * pen * pen) : 1.0
        for i in 1:Nx
            Vnext[i, j] = v
        end
    end

    Vcurr  = zeros(Float64, Nx, Nu)
    D      = zeros(Float64, Nx, Nu)   # DCT coefficients (Nx x Nu)
    E      = zeros(Float64, Nx, Nu)   # conditional expectation
    policy = zeros(Float64, Nx, Nu, c.Nsteps)

    for step in c.Nsteps:-1:1
        buyer_step!(Vcurr, Vnext, policy, E, D, ce, expfac, qgrid, nfeas, j0tab, ftab,
                    step, Nx, Nu, nq)
        Vnext, Vcurr = Vcurr, Vnext   # swap buffers (no allocation)
    end

    # V_tilde at t_0, evaluated at (x0, u=0) via linear interpolation in x
    price = price_at_x0(Vnext, gridX, x0, 1)
    return BuyerSolution(Vnext, gridX, gridU, policy, price)
end

# One backward-induction step: E[Vnext] via DCT-II + reconstruction, then the
# Bellman min over the per-period control.  Zero heap allocation (all buffers
# are caller-owned; the two mul! calls are in-place).
function buyer_step!(Vcurr, Vnext, policy, E, D, ce, expfac, qgrid, nfeas, j0tab, ftab,
                     step, Nx, Nu, nq)
    @inbounds for j in 1:Nu
        for i in 1:Nx
            ce.buf[i] = Vnext[i, j]
        end
        mul!(ce.out, ce.plan, ce.buf)     # DCT-II (REDFT10)
        for i in 1:Nx
            D[i, j] = ce.out[i]
        end
    end
    mul!(E, ce.A, D)                       # reconstruction (BLAS gemv/gemm)

    @inbounds for j in 1:Nu
        nf = nfeas[j]
        for i in 1:Nx
            best = Inf
            bestq = 0.0
            for l in 1:nf
                j0 = j0tab[j, l]
                f  = ftab[j, l]
                Ec  = E[i, j0] * (1.0 - f) + E[i, j0 + 1] * f
                val = expfac[i, l] * Ec
                if val < best
                    best = val
                    bestq = qgrid[l]
                end
            end
            Vcurr[i, j] = best
            policy[i, j, step] = bestq
        end
    end
    return nothing
end

# helper: linear interpolation of matrix col j at coordinate x
function price_at_x0(V::Matrix{Float64}, gridX::Vector{Float64}, x0::Float64, j::Int)
    Nx = length(gridX)
    dx = gridX[2] - gridX[1]
    idx = (x0 - gridX[1]) / dx + 1.0
    i0 = clamp(trunc(Int, idx), 1, Nx - 1)
    frac = idx - i0
    return V[i0, j] * (1.0 - frac) + V[i0 + 1, j] * frac
end

# ---------------------------------------------------------------------------
# Seller: spot baseline (1-D) and contract value along buyer policy,
# computed jointly with a common per-step rescaling to avoid underflow.
# Returns the ratio  V~_S(t0,x0,0) / V~_S,spot(t0,x0),  from which P_S follows.
# ---------------------------------------------------------------------------
function seller_ratio(p::ExpOUJumpParams, c::ContractSpec, ag::AgentParams,
                      policy::Array{Float64,3};
                      Nx::Int=256, Nu::Int=128, L::Float64=8.0, x0::Float64=0.0)
    dt  = c.T / c.Nsteps
    a, b = truncation_domain(p, c.T, x0; L=L)
    ce  = CosExpect(p, dt, a, b, Nx)
    gridX = [a + (j + 0.5) * ((b - a) / Nx) for j in 0:Nx-1]
    du    = c.Umax / (Nu - 1)
    gridU = [(j - 1) * du for j in 1:Nu]
    Sgrid = exp.(gridX)

    # per-period seller profit factors (economically sensible: the seller sells
    # on spot only when S > c_s, i.e. no production at a loss)
    #   spot baseline : pi_spot = (S - c_s)_+ * Kmax
    #   with contract : pi_S    = (fee - c_s) q + (S - c_s)_+ (Kmax - q)
    spotfac = zeros(Float64, Nx)
    @inbounds for i in 1:Nx
        m = Sgrid[i] - ag.cs
        spotfac[i] = exp(-ag.gammaS * (m > 0.0 ? m : 0.0) * ag.Kmax)
    end

    Vsp_next = ones(Float64, Nx)
    VS_next  = ones(Float64, Nx, Nu)
    Vsp_curr = zeros(Float64, Nx)
    VS_curr  = zeros(Float64, Nx, Nu)
    D = zeros(Float64, Nx, Nu); E = zeros(Float64, Nx, Nu)

    for step in c.Nsteps:-1:1
        # spot baseline expectation (1-D)
        mul!(ce.buf, ce.plan, Vsp_next)
        mul!(ce.out, ce.A, ce.buf)
        @inbounds for i in 1:Nx
            Vsp_curr[i] = spotfac[i] * ce.out[i]
        end
        # contract value expectation (2-D)
        @inbounds for j in 1:Nu
            for i in 1:Nx
                ce.buf[i] = VS_next[i, j]
            end
            mul!(ce.out, ce.plan, ce.buf)
            for i in 1:Nx
                D[i, j] = ce.out[i]
            end
        end
        mul!(E, ce.A, D)
        @inbounds for j in 1:Nu
            u = gridU[j]
            for i in 1:Nx
                q = policy[i, j, step]
                unext = u + q
                idx = unext / du + 1.0
                j0 = clamp(trunc(Int, idx), 1, Nu - 1)
                frac = idx - j0
                Ec = E[i, j0] * (1.0 - frac) + E[i, j0 + 1] * frac
                m = Sgrid[i] - ag.cs
                profit = (c.fee - ag.cs) * q + (m > 0.0 ? m : 0.0) * (ag.Kmax - q)
                VS_curr[i, j] = exp(-ag.gammaS * profit) * Ec
            end
        end
        Vsp_next, Vsp_curr = Vsp_curr, Vsp_next
        VS_next, VS_curr = VS_curr, VS_next
    end

    VS0  = price_at_x0(VS_next, gridX, x0, 1)
    Vsp0 = price_at_x0(reshape(Vsp_next, Nx, 1), gridX, x0, 1)
    return VS0 / Vsp0
end

solve_spot_dct(p, c, ag; kwargs...) = nothing  # retained for API completeness

function reservation_prices(p::ExpOUJumpParams, c::ContractSpec, ag::AgentParams;
                            Nx::Int=256, Nu::Int=128, nq::Int=128, L::Float64=8.0,
                            x0::Float64=0.0)
    buy = solve_buyer_dct(p, c, ag; Nx=Nx, Nu=Nu, nq=nq, L=L, x0=x0)
    P_B = -log(buy.price) / ag.gammaB
    ratio = seller_ratio(p, c, ag, buy.policy; Nx=Nx, Nu=Nu, L=L, x0=x0)
    P_S = log(ratio) / ag.gammaS
    P_star = nash_price(P_S, P_B, ag.alphaNash)
    return (P_B=P_B, P_S=P_S, P_star=P_star, ratio=ratio, V_B=buy.price)
end

nash_price(P_S::Float64, P_B::Float64, alphaNash::Float64) = alphaNash * P_S + (1.0 - alphaNash) * P_B

# ---------------------------------------------------------------------------
# Free boundary extraction from the buyer policy
# ---------------------------------------------------------------------------
# returns xbar(u) = smallest x on the grid where q*(x,u) > 0  (policy at given step)
function free_boundary(sol::BuyerSolution, step::Int=1)
    Nx, Nu = size(sol.V)
    q = view(sol.policy, :, :, step)
    xb = fill(NaN, Nu)
    @inbounds for j in 1:Nu
        found = false
        for i in 1:Nx
            if q[i, j] > 1e-12
                xb[j] = sol.gridX[i]
                found = true
                break
            end
        end
        if !found
            xb[j] = Inf
        end
    end
    return xb
end

# boundary on a supplied u-grid (for plotting)
function free_boundary_grid(p::ExpOUJumpParams, c::ContractSpec, ag::AgentParams, ugrid;
                            Nx::Int=256, Nu::Int=128, nq::Int=128, L::Float64=8.0,
                            x0::Float64=0.0)
    sol = solve_buyer_dct(p, c, ag; Nx=Nx, Nu=Nu, nq=nq, L=L, x0=x0)
    return sol, free_boundary(sol)
end

# ---------------------------------------------------------------------------
# Path simulation (exact) for the Exp-OU + MRJD process
# ---------------------------------------------------------------------------
# exact simulation: X_{k+1} = theta(1-emk) + emk*X_k + sigma*sqrt((1-e^{-2k dt})/(2k)) Z
#                       + sum_{j=1}^{J} N(muJ, sigmaJ^2),  J ~ Poisson(lambda dt)
function simulate_X(p::ExpOUJumpParams, dt::Float64, npath::Int, nsteps::Int;
                    x0::Float64, rng::AbstractRNG)
    emk = exp(-p.kappa * dt)
    m   = p.theta * (1.0 - emk)
    s   = p.sigma * sqrt((1.0 - exp(-2.0 * p.kappa * dt)) / (2.0 * p.kappa))
    X = zeros(Float64, npath, nsteps + 1)
    X[:, 1] .= x0
    for k in 1:nsteps
        @views X[:, k+1] .= m .+ emk .* X[:, k] .+ s .* randn(rng, npath)
        # jumps
        for i in 1:npath
            J = poisson_rand(rng, p.lambda * dt)
            if J > 0
                X[i, k+1] += p.muJ * J + p.sigmaJ * sum(randn(rng, J))
            end
        end
    end
    return X
end

# ---------------------------------------------------------------------------
# LSM baseline (grid-based, direct least-squares regression in x on a
# polynomial basis, value stored on the u-grid).  This is the canonical
# Longstaff-Schwartz estimator: the continuation value is projected onto the
# basis (unbiased conditional-expectation estimate), and positivity is enforced
# by clipping the fitted continuation at a small floor.
# ---------------------------------------------------------------------------
struct LSMResult
    price::Float64
    beta0::Matrix{Float64}     # regression coefficients at t_0, (deg+1) x Nu
    mu0::Float64
    sd0::Float64
    gridU::Vector{Float64}
    qgrid::Vector{Float64}
    deg::Int
end

function solve_buyer_lsm(p::ExpOUJumpParams, c::ContractSpec, ag::AgentParams;
                         npath::Int=100_000, Nu::Int=64, nq::Int=32,
                         deg::Int=5, x0::Float64=0.0, boundary_step::Int=2,
                         rng::AbstractRNG=Random.Xoshiro(1234))
    dt = c.T / c.Nsteps
    du = c.Umax / (Nu - 1)
    gridU = [(j - 1) * du for j in 1:Nu]

    X = simulate_X(p, dt, npath, c.Nsteps; x0=x0, rng=rng)
    S = exp.(X)

    qgrid = collect(range(c.qmin, c.qmax, length=nq))
    nfeas = [max(count(qgrid .<= c.Umax - gridU[j]), 0) for j in 1:Nu]

    # interpolation tables for u_j + q_l  (independent of path and step)
    j0tab = zeros(Int, Nu, nq)
    ftab  = zeros(Float64, Nu, nq)
    @inbounds for j in 1:Nu, l in 1:nq
        idx = (gridU[j] + qgrid[l]) / du + 1.0
        j0 = clamp(trunc(Int, idx), 1, Nu - 1)
        j0tab[j, l] = j0
        ftab[j, l]  = idx - j0
    end

    # terminal value
    Vnext = zeros(Float64, npath, Nu)
    @inbounds for j in 1:Nu
        pen = c.Umin - gridU[j]
        v = pen > 0.0 ? exp(ag.gammaB * ag.alphaB * pen * pen) : 1.0
        for i in 1:npath
            Vnext[i, j] = v
        end
    end
    Vcurr = zeros(Float64, npath, Nu)

    beta0 = Matrix{Float64}(undef, deg + 1, Nu)
    mu0 = 0.0; sd0 = 1.0

    for step in c.Nsteps:-1:1
        xk = @view X[:, step]
        mu = mean(xk); sd = std(xk)
        sd == 0.0 && (sd = 1.0)

        # design matrix (standardized monomials)
        Psi = ones(Float64, npath, deg + 1)
        @inbounds for i in 1:npath
            z = (xk[i] - mu) / sd
            acc = z
            for d in 1:deg
                Psi[i, d+1] = acc
                acc *= z
            end
        end
        beta = (Psi' * Psi) \ (Psi' * Vnext)     # (deg+1) x Nu
        C = Psi * beta

        # precomputed reward factors exp(-gammaB*(S_i - fee)*q_l)
        reward = zeros(Float64, npath, nq)
        @inbounds for i in 1:npath
            Si = S[i, step]
            for l in 1:nq
                reward[i, l] = exp(-ag.gammaB * (Si - c.fee) * qgrid[l])
            end
        end

        # Bellman min over q
        @inbounds for i in 1:npath
            for j in 1:Nu
                nf = nfeas[j]
                best = Inf
                for l in 1:nf
                    j0 = j0tab[j, l]
                    f  = ftab[j, l]
                    Cc = C[i, j0] * (1.0 - f) + C[i, j0 + 1] * f
                    if Cc < 1e-12
                        Cc = 1e-12
                    end
                    val = reward[i, l] * Cc
                    if val < best
                        best = val
                    end
                end
                Vcurr[i, j] = best
            end
        end
        if step == boundary_step
            beta0 .= beta
            mu0 = mu; sd0 = sd
        end
        Vnext, Vcurr = Vcurr, Vnext
    end

    V0 = mean(@view Vnext[:, 1])
    price = -log(V0) / ag.gammaB
    return LSMResult(price, beta0, mu0, sd0, gridU, qgrid, deg)
end

lsm_reservation_price(p, c, ag; kwargs...) = solve_buyer_lsm(p, c, ag; kwargs...).price

# ---------------------------------------------------------------------------
# LSM free boundary at t_0: for each u, smallest x on the grid with q*(x,u)>0
# ---------------------------------------------------------------------------
function lsm_free_boundary(p::ExpOUJumpParams, c::ContractSpec, ag::AgentParams,
                           xgrid::Vector{Float64}, ugrid::Vector{Float64};
                           npath::Int=100_000, Nu::Int=64, nq::Int=32, deg::Int=5,
                           x0::Float64=0.0, rng::AbstractRNG=Random.Xoshiro(1234))
    r = solve_buyer_lsm(p, c, ag; npath=npath, Nu=Nu, nq=nq, deg=deg, x0=x0, rng=rng)
    du = c.Umax / (Nu - 1)
    Nu_ = length(r.gridU)
    xb = fill(NaN, length(ugrid))
    for (k, u) in enumerate(ugrid)
        # feasible q for this u
        qmax_eff = min(c.qmax, c.Umax - u)
        found = false
        for x in xgrid
            S = exp(x)
            z = (x - r.mu0) / r.sd0
            # basis vector
            psi = Vector{Float64}(undef, r.deg + 1)
            acc = 1.0
            for d in 0:r.deg
                psi[d+1] = acc
                acc *= z
            end
            bestq = 0.0
            bestv = Inf
            for l in 1:length(r.qgrid)
                q = r.qgrid[l]
                q > qmax_eff && break
                unext = u + q
                idx = unext / du + 1.0
                j0 = clamp(trunc(Int, idx), 1, Nu_ - 1)
                frac = idx - j0
                # continuation C(x, unext) = beta0' psi (with u-interpolation)
                c0 = 0.0; c1 = 0.0
                for d in 0:r.deg
                    c0 += r.beta0[d+1, j0] * psi[d+1]
                    c1 += r.beta0[d+1, j0+1] * psi[d+1]
                end
                Cc = c0 * (1.0 - frac) + c1 * frac
                Cc = max(Cc, 1e-12)
                val = exp(-ag.gammaB * (S - c.fee) * q) * Cc
                if val < bestv
                    bestv = val
                    bestq = q
                end
            end
            if bestq > 1e-12
                xb[k] = x
                found = true
                break
            end
        end
        if !found
            xb[k] = Inf
        end
    end
    return xb
end

# ---------------------------------------------------------------------------
# Benchmark (timing + allocations) for the DCT backward induction
# ---------------------------------------------------------------------------
function bench_dct(p::ExpOUJumpParams, c::ContractSpec, ag::AgentParams;
                   Nx::Int=256, Nu::Int=128, nq::Int=128, L::Float64=8.0, x0::Float64=0.0)
    dt  = c.T / c.Nsteps
    a, b = truncation_domain(p, c.T, x0; L=L)
    ce  = CosExpect(p, dt, a, b, Nx)
    gridX = [a + (j + 0.5) * ((b - a) / Nx) for j in 0:Nx-1]
    du    = c.Umax / (Nu - 1)
    gridU = [(j - 1) * du for j in 1:Nu]
    Sgrid = exp.(gridX)
    qgrid = range(c.qmin, c.qmax, length=nq) |> collect
    expfac = zeros(Float64, Nx, nq)
    @inbounds for i in 1:Nx, l in 1:nq
        expfac[i, l] = exp(-ag.gammaB * (Sgrid[i] - c.fee) * qgrid[l])
    end
    nfeas = zeros(Int, Nu)
    @inbounds for j in 1:Nu
        u = gridU[j]; k = nq
        while k >= 1 && qgrid[k] > c.Umax - u; k -= 1; end
        nfeas[j] = max(k, 0)
    end
    j0tab = zeros(Int, Nu, nq); ftab = zeros(Float64, Nu, nq)
    @inbounds for j in 1:Nu, l in 1:nq
        idx = (gridU[j] + qgrid[l]) / du + 1.0
        j0 = clamp(trunc(Int, idx), 1, Nu - 1)
        j0tab[j, l] = j0; ftab[j, l] = idx - j0
    end
    Vnext = zeros(Float64, Nx, Nu)
    @inbounds for j in 1:Nu
        pen = c.Umin - gridU[j]
        v = pen > 0.0 ? exp(ag.gammaB * ag.alphaB * pen * pen) : 1.0
        for i in 1:Nx; Vnext[i, j] = v; end
    end
    Vcurr = zeros(Float64, Nx, Nu); D = zeros(Float64, Nx, Nu); E = zeros(Float64, Nx, Nu)
    policy = zeros(Float64, Nx, Nu, c.Nsteps)

    # warm-up (also compiles the hot path)
    buyer_step!(Vcurr, Vnext, policy, E, D, ce, expfac, qgrid, nfeas, j0tab, ftab, 1, Nx, Nu, nq)
    Vnext, Vcurr = Vcurr, Vnext

    alloc = @allocated for step in 1:c.Nsteps
        buyer_step!(Vcurr, Vnext, policy, E, D, ce, expfac, qgrid, nfeas, j0tab, ftab, step, Nx, Nu, nq)
        Vnext, Vcurr = Vcurr, Vnext
    end
    t = @elapsed for step in 1:c.Nsteps
        buyer_step!(Vcurr, Vnext, policy, E, D, ce, expfac, qgrid, nfeas, j0tab, ftab, step, Nx, Nu, nq)
        Vnext, Vcurr = Vcurr, Vnext
    end
    return (time=t, alloc_bytes=alloc, per_step_alloc=alloc / c.Nsteps, per_step=t / c.Nsteps)
end

end # module
