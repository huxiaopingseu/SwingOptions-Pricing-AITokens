# SwingOptions-Pricing-AITokens

Bilateral utility-indifference pricing of AI-compute-token swing contracts in
incomplete markets. The spot token value follows an exponential Ornstein–Uhlenbeck
process with Poisson jumps (Exp-OU + MRJD); the swing contract carries per-period
rate limits, a global capacity ceiling, and a take-or-pay minimum. Both the buyer
and the seller have constant absolute risk aversion, and the transaction price is
the generalized Nash bargaining solution of their reservation prices.

The numerical engine evaluates the conditional expectations in the Bellman
equations with a Fourier-cosine (COS) expansion realized through the discrete
cosine transform (FFTW DCT-II plus a dense reconstruction matrix), which yields a
deterministic, **zero-heap-allocation** backward induction. A grid-based
least-squares Monte Carlo (LSM) solver is provided as the benchmark.

## Repository layout

```
.
├── src/
│   └── AITokenSwing.jl          # pricing engine (DCT + LSM + free boundary)
├── scripts/
│   ├── run_experiments.jl       # computes all results -> results/experiments.json
│   └── make_figures.py          # renders publication figures -> figures/
├── results/
│   └── experiments.json         # all numerical results
├── figures/                     # 8 publication figures (PNG)
├── Project.toml / Manifest.toml # Julia environment (FFTW)
└── README.md
```

## Key results (base case)

| Quantity | DCT method | LSM benchmark |
|---|---|---|
| Buyer reservation price $P_B$ | 16.17 (ref, $N_x{=}1024$) | 16.46 (100k paths) |
| Error vs reference | 0.1% at $N_x{=}512$ | 1.8% at 100k paths |
| Runtime | 0.47 s | 6.7 s |
| Heap allocation / step | **0 bytes** | — |

Seller reservation price $P_S = 4.76$; Nash transaction price $P^* = 10.46$
($\alpha = 0.5$). The DCT free boundary is smooth and monotone in cumulative
consumption, converging to $\ln c$; the LSM boundary is seed-dependent and
non-monotone.

## Quick start

```bash
# 1. run the experiments (writes results/experiments.json)
julia --project=. scripts/run_experiments.jl

# 2. render the figures (writes figures/*.png)
python3 scripts/make_figures.py
```

## Requirements

- Julia 1.11+ with the `FFTW` package (`julia --project=. -e 'using Pkg; Pkg.instantiate()'`)
- Python 3.10+ with `numpy` and `matplotlib` (figures only)

## Citation

If you use this code, please cite the accompanying manuscript:

> Hu, X. Pricing AI Compute Tokens as Swing Options: A Bilateral
> Utility-Indifference Framework for Incomplete Markets.
