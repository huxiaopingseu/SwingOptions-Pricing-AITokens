#!/usr/bin/env python3
# make_figures.py — publication figures for the AI-compute-token swing-option paper
# Reads results/experiments.json, writes figures/*.png
import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)

rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 10,
    "axes.labelsize": 10,
    "axes.titlesize": 10,
    "legend.fontsize": 8,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "figure.dpi": 200,
    "savefig.dpi": 200,
    "axes.linewidth": 0.8,
})

D = json.load(open("results/experiments.json"))
bp = D["base_params"]

# ---------------------------------------------------------------- fig 1: paths
def fig1_sample_paths():
    rng = np.random.default_rng(7)
    kappa, theta, sigma = bp["kappa"], bp["theta"], bp["sigma"]
    lam, muJ, sigmaJ = bp["lambda"], bp["muJ"], bp["sigmaJ"]
    T, N = bp["T"], bp["N"]
    dt = T / N
    x0 = bp["x0"]
    emk = np.exp(-kappa * dt)
    s = sigma * np.sqrt((1 - np.exp(-2 * kappa * dt)) / (2 * kappa))
    fig, ax = plt.subplots(figsize=(6.2, 3.4))
    ts = np.linspace(0, T, N + 1)
    for _ in range(5):
        x = np.empty(N + 1); x[0] = x0
        for k in range(N):
            x[k + 1] = theta * (1 - emk) + emk * x[k] + s * rng.standard_normal()
            J = rng.poisson(lam * dt)
            if J:
                x[k + 1] += muJ * J + sigmaJ * rng.standard_normal() * np.sqrt(J)
        ax.plot(ts, np.exp(x), lw=1.0)
    ax.axhline(np.exp(theta), color="0.4", ls="--", lw=0.8)
    ax.axhline(bp["fee"], color="C3", ls=":", lw=0.9)
    ax.annotate(r"long-run level $e^{\theta}$", (0.02, np.exp(theta) * 1.02), fontsize=8)
    ax.annotate("contract fee $c$", (0.02, bp["fee"] * 1.05), color="C3", fontsize=8)
    ax.set_xlabel("time (years)"); ax.set_ylabel(r"spot token value $S_t$")
    ax.set_title("Sample paths of the Exp-OU + jump-diffusion spot process")
    fig.tight_layout(); fig.savefig("figures/fig1_sample_paths.png"); plt.close(fig)

# ---------------------------------------------------------------- fig 2: surface
def fig2_value_surface():
    s = D["surface"]
    x = np.array(s["x"]); u = np.array(s["u"]); logV = np.array(s["logV"]).T  # (x,u)->(u,x) for imshow
    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    im = ax.pcolormesh(x, u, logV, shading="auto", cmap="viridis")
    ax.axhline(bp["Umin"], color="w", ls="--", lw=0.8)
    ax.annotate(r"$U_{\min}$ (take-or-pay)", (x[-1] - 4.6, bp["Umin"] + 0.3), color="w", fontsize=8)
    cb = fig.colorbar(im, ax=ax)
    cb.set_label(r"$\log \widetilde V_B(t_0,x,u)$")
    ax.set_xlabel(r"log-spot $x$"); ax.set_ylabel(r"cumulative consumption $u$")
    ax.set_title("Buyer value function at inception")
    fig.tight_layout(); fig.savefig("figures/fig2_value_surface.png"); plt.close(fig)

# ---------------------------------------------------------------- fig 3: boundary t0
def fig3_free_boundary_t0():
    b = D["boundary"]
    u = np.array(b["u"]); xb = np.array(b["xbar_dct"])
    xb = np.where(np.isinf(xb), np.nan, xb)
    fig, ax = plt.subplots(figsize=(5.6, 3.4))
    ax.plot(u, xb, "o-", color="C0", lw=1.4, ms=4, label=r"DCT boundary $\bar x(u)$")
    ax.axhline(bp["ln_fee"], color="C3", ls="--", lw=1.0, label=r"$\ln c$ (voluntary threshold)")
    ax.set_xlabel(r"cumulative consumption $u$")
    ax.set_ylabel(r"exercise boundary $\bar x(u)$")
    ax.set_title("Free exercise boundary at inception (DCT)")
    ax.legend()
    fig.tight_layout(); fig.savefig("figures/fig3_free_boundary_t0.png"); plt.close(fig)

# ---------------------------------------------------------------- fig 4: DCT vs LSM boundary
def fig4_dct_vs_lsm():
    b = D["boundary"]; lb = D["boundary_lsm"]
    u = np.array(b["u"])
    xb_dct = np.where(np.isinf(np.array(b["xbar_dct_t1"])), np.nan, np.array(b["xbar_dct_t1"]))
    fig, ax = plt.subplots(figsize=(5.6, 3.4))
    for k, seed in enumerate(lb):
        xb = np.array(lb[seed]); xb[xb < -3.0] = np.nan  # -1.0 = "no exercise" sentinel -> drop
        ax.plot(u, xb, "x-", lw=0.9, ms=4, alpha=0.75,
                label="LSM" if k == 0 else None, color="C1")
    ax.plot(u, xb_dct, "o-", color="C0", lw=1.6, ms=4, label="DCT")
    ax.axhline(bp["ln_fee"], color="C3", ls="--", lw=0.9)
    ax.set_xlabel(r"cumulative consumption $u$")
    ax.set_ylabel(r"exercise boundary $\bar x(u)$ at $t_1$")
    ax.set_title("DCT (smooth, deterministic) vs LSM (seed-dependent)")
    ax.legend()
    fig.tight_layout(); fig.savefig("figures/fig4_dct_vs_lsm.png"); plt.close(fig)

# ---------------------------------------------------------------- fig 5: boundary time drift
def fig5_boundary_time():
    bt = D["boundary_time"]
    st = np.array(bt["step"]); u0 = np.array(bt["u0"]); u3 = np.array(bt["u3"])
    t = (st - 1) / bp["N"]
    fig, ax = plt.subplots(figsize=(5.6, 3.4))
    ax.plot(t, u0, "o-", color="C0", lw=1.4, ms=4, label=r"$u=0$ (no consumption yet)")
    ax.plot(t, u3, "s-", color="C2", lw=1.4, ms=4, label=r"$u=3$ (half of $U_{\min}$)")
    ax.axhline(bp["ln_fee"], color="C3", ls="--", lw=0.9, label=r"$\ln c$")
    ax.set_xlabel("time $t$ (years)")
    ax.set_ylabel(r"exercise boundary $\bar x(t,u)$")
    ax.set_title("Take-or-pay deadline pulls the boundary downward")
    ax.legend()
    fig.tight_layout(); fig.savefig("figures/fig5_boundary_time.png"); plt.close(fig)

# ---------------------------------------------------------------- fig 6: convergence
def fig6_convergence():
    ref = D["reference"]["P_B"]
    cd = D["conv_dct"]; cl = D["conv_lsm"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 3.2))
    # DCT
    nx = np.array(cd["Nx"], float); err = np.abs(np.array(cd["P_B"]) - ref)
    ax1.loglog(nx, err, "o-", color="C0", lw=1.2, ms=4)
    ax1.loglog(nx, err[0] * (nx / nx[0]) ** -2, "--", color="0.5", lw=0.9, label=r"$O(N_x^{-2})$")
    ax1.set_xlabel(r"grid size $N_x$"); ax1.set_ylabel(r"$|P_B - P_B^*|$")
    ax1.set_title("DCT convergence")
    ax1.legend()
    # LSM
    np_ = np.array(cl["npath"], float); err_ = np.abs(np.array(cl["mean"]) - ref)
    std_ = np.array(cl["std"])
    ax2.errorbar(np_, err_, yerr=std_, fmt="o-", color="C1", lw=1.2, ms=4, capsize=3)
    ax2.loglog(np_, err_[0] * (np_ / np_[0]) ** -0.5, "--", color="0.5", lw=0.9, label=r"$O(P^{-1/2})$")
    ax2.set_xscale("log"); ax2.set_yscale("log")
    ax2.set_xlabel(r"paths $P$"); ax2.set_ylabel(r"$|P_B - P_B^*|$")
    ax2.set_title("LSM convergence")
    ax2.legend()
    fig.tight_layout(); fig.savefig("figures/fig6_convergence.png"); plt.close(fig)

# ---------------------------------------------------------------- fig 7: sensitivity
def fig7_sensitivity():
    s = D["sensitivity"]
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.2))
    # fee
    ax = axes[0, 0]
    f = s["fee"]; ax.plot(f["fee"], f["P_B"], "o-", lw=1.2, ms=3, label=r"$P_B$")
    ax.plot(f["fee"], f["P_S"], "s-", lw=1.2, ms=3, label=r"$P_S$")
    ax.plot(f["fee"], f["P_star"], "^-", lw=1.2, ms=3, label=r"$P^*$")
    ax.set_xlabel("contract fee $c$"); ax.set_ylabel("price"); ax.legend(); ax.set_title("(a) fee")
    # gammaB
    ax = axes[0, 1]
    g = s["gammaB"]; ax.plot(g["gammaB"], g["P_B"], "o-", lw=1.2, ms=3, label=r"$P_B$")
    ax.plot(g["gammaB"], g["P_star"], "^-", lw=1.2, ms=3, label=r"$P^*$")
    ax.set_xlabel(r"buyer risk aversion $\gamma_B$"); ax.set_ylabel("price"); ax.legend(); ax.set_title(r"(b) risk aversion")
    # lambda
    ax = axes[1, 0]
    l = s["lambda"]; ax.plot(l["lambda"], l["P_B"], "o-", lw=1.2, ms=3, label=r"$P_B$")
    ax.plot(l["lambda"], l["P_S"], "s-", lw=1.2, ms=3, label=r"$P_S$")
    ax.set_xlabel(r"jump intensity $\lambda$"); ax.set_ylabel("price"); ax.legend(); ax.set_title(r"(c) jump intensity")
    # sigmaJ
    ax = axes[1, 1]
    sj = s["sigmaJ"]; ax.plot(sj["sigmaJ"], sj["P_B"], "o-", lw=1.2, ms=3, label=r"$P_B$")
    ax.plot(sj["sigmaJ"], sj["P_S"], "s-", lw=1.2, ms=3, label=r"$P_S$")
    ax.set_xlabel(r"jump-size vol $\sigma_J$"); ax.set_ylabel("price"); ax.legend(); ax.set_title(r"(d) jump-size volatility")
    fig.suptitle("Sensitivity of bilateral reservation prices")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig("figures/fig7_sensitivity.png"); plt.close(fig)

# ---------------------------------------------------------------- fig 8: timing
def fig8_timing():
    t = D["timing"]; cl = D["conv_lsm"]
    fig, ax = plt.subplots(figsize=(5.6, 3.4))
    nx = np.array(t["Nx"], float); tt = np.array(t["time"])
    ax.semilogy(nx, tt, "o-", color="C0", lw=1.2, ms=4, label="DCT (total, 12 steps)")
    np_ = np.array(cl["npath"], float); tl = np.array(cl["time"])
    ax.semilogy(np_, tl, "s-", color="C1", lw=1.2, ms=4, label="LSM (total)")
    ax.set_xlabel("resolution ($N_x$ or paths $P$)")
    ax.set_ylabel("wall-clock time (s)")
    ax.set_title("Runtime: DCT vs LSM")
    ax.legend()
    fig.tight_layout(); fig.savefig("figures/fig8_timing.png"); plt.close(fig)

if __name__ == "__main__":
    import os
    os.makedirs("figures", exist_ok=True)
    fig1_sample_paths(); print("fig1 done")
    fig2_value_surface(); print("fig2 done")
    fig3_free_boundary_t0(); print("fig3 done")
    fig4_dct_vs_lsm(); print("fig4 done")
    fig5_boundary_time(); print("fig5 done")
    fig6_convergence(); print("fig6 done")
    fig7_sensitivity(); print("fig7 done")
    fig8_timing(); print("fig8 done")
    print("ALL FIGURES DONE")
