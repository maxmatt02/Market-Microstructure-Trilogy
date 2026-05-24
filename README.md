# Market-Microstructure-Trilogy

**Max Matthews** · Eller College of Management, University of Arizona · `maxmatt2@arizona.edu`

Three working papers on limit order book dynamics, high-frequency microstructure, and market scaling geometry. Built on 12.4 billion discrete tick-level observations from CME Globex ES/NQ futures and Binance BTCUSDT/ETHUSDT perpetual futures (2022–2024).

---

## Papers

> **Methodological Disclaimer:** All Sharpe ratios, Calmar ratios, and return figures reported in this repository are *gross theoretical metrics* derived from walk-forward simulated signal performance on historical order-book data. They do not constitute realized trading results and have not been adjusted for the full cost of real-world execution, including but not limited to: bid-ask spread crossing at stressed microstructure conditions, market impact under the Square Root Law, queue-position fill uncertainty, and latency-dependent adverse selection. Industry practitioners should treat all figures as upper-bound estimates of signal quality prior to a complete transaction cost analysis (TCA). No representation is made that these results are achievable in live trading.

### 1. Constrained Stochasticity in Central Limit Order Books (2025)
`/paper_1_revised.pdf`

**Core claim:** High-frequency price formation is simultaneously stochastic at the nanosecond level and structurally bounded by algorithmic liquidity architecture at meso- and macro-timescales.

**Key constructs:**
- **Discretized Execution Validation Matrix (Φₖ):** A tick-level linear-algebraic operator aggregating localized liquidity tensors (Lᵢₖ ∈ ℝⁿˣⁿ), directional execution vectors (Vᵢₖ ∈ ℝⁿ), and a volumetric mass tensor (Mₖ ∈ ℝⁿˣⁿ) via Hadamard composition:

  `Φₖ = Σᵢ (Lᵢₖ · Vₖᵀ) ⊙ Mₖ`

- **Three-state HMM regime-switching:** Regime 1 (Normal, 68.2% of days), Regime 2 (Stressed, 24.1%), Regime 3 (Chaotic, 7.7%). Baum-Welch estimation; Viterbi decoding on walk-forward basis.
- **Absorbing boundary (Regime 3):** Derived from the Kesten-Goldie theorem. Under ρ(A) ≥ 1, expected time to ruin is O(log W₀) for any fixed leverage φ > 0. Optimal policy: φ* = 0 when P(Sₖ = 3 | ℱₖ) ≥ η.
- **Discretized bivariate Hawkes process:** Common decay parameter β justified by likelihood ratio test χ²(3) = 4.71, p = 0.194.

**Data:** CME Datamine Level-3, ES + NQ, Jan 2022–Dec 2024. ~8.3 billion order-book events. Walk-forward optimization: 12-month training window, 1-month OOS, rolled quarterly.

**Results (out-of-sample):**

| Regime | Sharpe | 95% CI | Max DD |
|--------|--------|--------|--------|
| 1 — Normal | 2.85 | [2.41, 3.29] | 8.4% |
| 2 — Stressed | 1.42 | [0.97, 1.87] | 14.7% |
| 3 — Clamped | 0.00 | — | 0.0% |
| Buy-and-hold | 0.61 | [0.22, 1.00] | 33.8% |

Both active-regime results survive Bonferroni correction (m=4, α/m = 0.0125).

---

### 2. Multifractal Price Delivery in Algorithmic Futures Markets (2026)
`/paper2revision.pdf`

**Core claim:** Institutional order-book liquidity engineering operates as a nested, scale-invariant hierarchy of Accumulation-Distribution (AD) cycles. The AD cycle repeats self-similarly across all accessible temporal scales, with each phase boundary detectable as a statistically significant collapse in the local Hurst exponent.

**Key constructs:**
- **Fractal Law of Price Delivery (FLPD):** Under long-range dependence H > 0.5, Temporal Liquidity Vacuum (TLV) durations follow a power law P(s > x) ~ x^{−α} with α = 1/(H − 0.5). TLVs are multifractally clustered across scales consistent with the MRW log-correlated field (Bacry, Delour & Muzy, 2001).
- **Dynamic Hurst exponent (Hₜ):** Estimated via MF-DFA at order q=2 on overlapping windows of W = 2,000 ticks, stepped by 200 ticks. Generalized spectrum h(q) computed for q ∈ {1, 2, 3}.
- **Terminal Macro-Attractor (Ωₜ):** Phase boundary identified by CUSUM sequential change-point test (Chu, Stinchcombe & White, 1996) on rolling Hₜ sequence. Mean Hₜ drop at Ωₜ events: −0.094 (t = 11.75, p < 0.001).
- **Discretized Multiscale Hierarchical Delivery Matrix (Ψ):**

  `Ψₛₜ = Σⱼ∈[t−δ,t] (Mₛ₊₁,ⱼ · e^{−λ(t−j)}) ⊙ νⱼ`

  Convergence to Ωₜ in L² at rate O(n^{−H}) under Taqqu (1975), restricted to stationary sub-periods.

**Hurst signature equivalence (MF-DFA across scales):**

| Scale | h(1) | h(2) | h(3) | Permutation p |
|-------|------|------|------|----------------|
| 5-min | 0.71 ± 0.04 | 0.67 ± 0.03 | 0.62 ± 0.04 | — |
| 20-min | 0.70 ± 0.04 | 0.66 ± 0.03 | 0.61 ± 0.04 | 0.41 |
| 1-hour | 0.70 ± 0.05 | 0.67 ± 0.04 | 0.62 ± 0.05 | 0.55 |
| 4-hour | 0.69 ± 0.05 | 0.65 ± 0.04 | 0.61 ± 0.05 | 0.62 |
| Daily | 0.71 ± 0.06 | 0.67 ± 0.05 | 0.62 ± 0.06 | 0.48 |

Overall permutation test χ²(8) = 9.14, p = 0.33. *Note: minimum detectable |Δh(2)| ≈ 0.07 at 80% power; moderate scale-dependence below this threshold cannot be ruled out.*

**Data:** Binance BTCUSDT Perpetual Futures Level-3, Jan 2022–Dec 2024. 4.18 billion order-book events. Training: Jan–Dec 2022. OOS: Jan 2023–Dec 2024 (847 Ωₜ events; 576 OOS).

**Results (out-of-sample, Jan 2023–Dec 2024):**

| Metric | FLPD | Buy-and-Hold BTC | Momentum |
|--------|------|-----------------|----------|
| Annualized Return | 63.4% | 48.1% | 31.2% |
| Sharpe Ratio | **2.41** | 0.54 | 0.79 |
| 95% CI | [1.98, 2.84] | — | — |
| Max Drawdown | 11.8% | 77.2% | 44.6% |
| Calmar Ratio | 5.37 | 0.62 | 0.70 |

ETHUSDT replication: Sharpe 2.19 [1.74, 2.64]. Robustness: LUNA/FTX event windows excluded; stationarity-restricted convergence claim; kernel specification, parameter, and training-window sensitivity all checked.

---

### 3. The Network Geometry of Order Flow: Sublinear Scaling in U.S. Futures Markets (2026)
`/Paper_3revision_2.pdf`

**Core claim:** Participant order activity in futures markets scales with capital footprint at the ¾ power — the same exponent as biological metabolic scaling — derived analytically from optimal branching conditions in hierarchical order-routing networks.

**Derivation of α = 3/4 (four closed steps):**

1. **fₖ ∝ Qₖ^{−1/4}** from Almgren-Chriss adverse-selection cost balance: optimal update frequency trades adverse-selection exposure (∝ Qₖ^{1/2}) against fixed message cost.

2. **Capacity exponent 2/3** from Square Root Law slippage minimization: `n · Qₖ₊₁^{2/3} = Qₖ^{2/3}`, giving Qₖ₊₁ = Qₖ · n^{−3/2}.

3. **Geometric-sum correction C(n,N):** Total metabolism Y = Q₀^{−1/4} · Σₖ (n · n^{3/8})^k. The ratio C(n,N) = n^{11/8}/(n^{11/8}−1) · (1 − n^{−11(N+1)/8}) converges to a constant in M at exponential rate in N (< 10⁻⁴ for n ≥ 2, N ≥ 10). Naively this yields Y ∝ M^{2/3}.

4. **Effective Terminal Density correction (+M^{1/12}):** Space-filling coverage requires nᴺ · a₀ · q₀^{1/3} = A · q₀^{1/3} (three degrees of freedom in order placement: price, time, size). This forces n ∝ M^{1/(4N)}, adding precisely M^{1/12} to the geometric sum: nᴺ · C(n,N) ∝ M^{11/12} · M^{1/12} = M. Combined with M^{−1/4} from Step 1: **Y ∝ M^{3/4}**.

**Robustness (Proposition 2):** Heterogeneous/time-varying branching with E[nₖⱼ] = μₙ, Var(nₖⱼ) < ∞. Martingale argument on Xₖ = Q̃ₖ^{−1/4}/E[Q̃ₖ]^{−1/4} establishes E[Y] ∝ M^{3/4} and a.s. convergence.

**Economic corollaries:**
- Mass-specific metabolism: Y/M ∝ M^{−1/4} (10× capital → only 5.6× message traffic)
- Participant lifespan: T ∝ M^{1/4}
- Reconciliation with Square Root Law: ΔP ∝ σ√(Q/V) governs single-execution price impact; allometric scaling governs cross-participant message activity. Both emerge from the same Almgren-Chriss optimization at different levels of description.

**Empirical design (pre-registered; live CME Datamine data acquisition pending):**
- Latency-burst fingerprinting via HDBSCAN (τ = 100µs, ξ = 50 bursts, cosine similarity > 0.85)
- IV correction for co-location conflation: instrument Zᵢ = ln(σQ,i/μQ,i); requires first-stage F > 10 and Kleibergen-Paap rk Wald diagnostic
- Power analysis: N ≥ 280 participant-proxy clusters required (80% power, α = 0.01); 350–500 expected from one year ES+NQ Level-2
- Pre-registered falsification criteria: OLS α̂ ∈ [0.65, 0.85]; IV α̂IV ≥ 0.70; ES and NQ estimates within 0.12 of each other

**Scale-tranche stratification (8 logarithmic notional bands, T1: <$50K retail algo → T8: >$50B sovereign/pension)** provides a tranche-level regression immune to clustering identification error.

---

## Repository Structure
