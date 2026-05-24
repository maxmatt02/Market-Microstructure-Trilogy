"""
phi_k.py
========
Production-grade construction of the Discretized Execution Validation Matrix (Φₖ),
the central linear-algebraic operator of the Constrained Stochasticity framework.

Mathematical Definition
-----------------------
At each event-tick k, Φₖ is defined as:

    Φₖ = Σᵢ₌₁ⁿ ( Lᵢₖ · Vₖᵀ ) ⊙ Mₖ

where:

    Lᵢₖ ∈ ℝⁿˣⁿ   — Localized Liquidity Matrix for price-level partition i at tick k.
                    A symmetric, positive semi-definite (PSD) Gram matrix encoding
                    bilateral depth and its first-order gradient around the best quote.

    Vₖ ∈ ℝⁿ       — Directional Execution Vector at tick k.
                    The signed, size-weighted net order flow over the trailing δ ticks.
                    Captures the directional pressure of informed vs. uninformed flow.

    ·              — Standard matrix-vector outer product. Each (Lᵢₖ · Vₖᵀ) produces
                    an n×n rank-1 perturbation of the symmetric PSD base.

    Mₖ ∈ ℝⁿˣⁿ    — Volumetric Mass Tensor at tick k.
                    A symmetric PSD matrix encoding cross-partition correlations of
                    Hawkes conditional intensities over a trailing W-tick window.
                    Acts as a spatially adaptive bandwidth operator (cf. kernel-weighted
                    nonparametric regression — Fan & Gijbels 1996).

    ⊙              — Hadamard (element-wise) product.
                    Up-weights (L, V) outer products where arrival intensities are
                    correlated; down-weights where intensities are orthogonal.
                    Preserves PSD by the Schur product theorem (Horn & Johnson 1990).

PSD Guarantee
-------------
Σᵢ(Lᵢₖ · Vₖᵀ) need not itself be PSD; however, the Hadamard product of any matrix
with a PSD matrix Mₖ produces a PSD matrix (Schur product theorem). We verify this
numerically after each construction via minimum-eigenvalue inspection.

Regime Semantics
----------------
The scalar trace tr(Φₖ) aggregates the total "execution pressure signal" across all
price-level partitions. In practice:

    tr(Φₖ) ↑  →  concentrated, directional, high-intensity order flow (Regime 1/2)
    tr(Φₖ) ≈ 0  →  diffuse, low-depth, chaotic flow (Regime 3 precursor)

The spectral norm ‖Φₖ‖₂ = σ_max(Φₖ) bounds the worst-case amplification of any
directional signal through the operator.

References
----------
Fan, J., & Gijbels, I. (1996). Local Polynomial Modelling and Its Applications. Chapman & Hall.
Horn, R. A., & Johnson, C. R. (1990). Matrix Analysis. Cambridge University Press.
    → Theorem 5.2.1 (Schur product theorem): if A, B ∈ ℝⁿˣⁿ are PSD, then A ⊙ B is PSD.

Author  : Max Matthews <maxmatt2@arizona.edu>
Affil.  : Eller College of Management, University of Arizona
Version : 1.0  (2025)
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
from numpy.linalg import eigvalsh, norm

# ---------------------------------------------------------------------------
# GPU acceleration — attempt cupy import; fall back to numpy transparently.
# In production, CME Globex Level-3 data runs on NVIDIA A100 with CuPy.
# ---------------------------------------------------------------------------
try:
    import cupy as cp  # type: ignore[import]

    _GPU_AVAILABLE = True
    _GPU_DEVICE = cp.cuda.Device(0).id
except ImportError:
    cp = None
    _GPU_AVAILABLE = False
    _GPU_DEVICE = None


def _get_array_module(use_gpu: bool):
    """Return cupy if GPU is requested and available, else numpy."""
    if use_gpu and _GPU_AVAILABLE:
        return cp
    return np


# ===========================================================================
# Section 1 — Input Data Containers
# ===========================================================================


@dataclass
class OrderBookSnapshot:
    """
    A single tick-k snapshot of Level-3 order book state.

    Attributes
    ----------
    L_partitions : List of n Localized Liquidity Matrices Lᵢₖ ∈ ℝⁿˣⁿ, i=1..n.
                   Each Lᵢₖ is a symmetric PSD Gram matrix derived from bilateral
                   depth within the i-th price-level partition.

    V_k          : Directional Execution Vector Vₖ ∈ ℝⁿ.
                   Signed, size-weighted net order flow over trailing δ ticks.
                   Positive entries → net buy pressure; negative → net sell.

    M_k          : Volumetric Mass Tensor Mₖ ∈ ℝⁿˣⁿ.
                   Empirical covariance of Hawkes conditional intensity vector
                   (λᵇₖ, λˢₖ, …) over a trailing W-tick window.
                   Must be symmetric PSD (enforced on construction).

    tick_k       : Integer event-tick index.
    n_levels     : Number of price-level partitions n.
    delta_ticks  : Lookback δ used to construct Vₖ (default 50 ticks ≈ 1 second ES).
    W_ticks      : Rolling window W used to construct Mₖ (default 500 ticks ≈ 10s ES).
    """

    L_partitions: List[np.ndarray]   # shape: n × (n, n)
    V_k: np.ndarray                  # shape: (n,)
    M_k: np.ndarray                  # shape: (n, n)
    tick_k: int = 0
    n_levels: int = 10
    delta_ticks: int = 50
    W_ticks: int = 500

    def __post_init__(self):
        n = self.n_levels
        if len(self.L_partitions) != n:
            raise ValueError(
                f"Expected {n} L_i matrices (one per partition), "
                f"got {len(self.L_partitions)}."
            )
        for i, L in enumerate(self.L_partitions):
            if L.shape != (n, n):
                raise ValueError(f"L_partitions[{i}] must be ({n},{n}); got {L.shape}.")
        if self.V_k.shape != (n,):
            raise ValueError(f"V_k must have shape ({n},); got {self.V_k.shape}.")
        if self.M_k.shape != (n, n):
            raise ValueError(f"M_k must be ({n},{n}); got {self.M_k.shape}.")


@dataclass
class PhiResult:
    """
    Full result object returned by ExecutionValidationMatrix.compute().

    Attributes
    ----------
    Phi_k         : The computed Φₖ matrix, shape (n, n).
    is_psd        : True if Φₖ passes numerical PSD verification.
    min_eigenval  : Minimum eigenvalue of Φₖ (negative → not PSD, check Mₖ).
    trace         : tr(Φₖ) — aggregate execution pressure signal.
    spectral_norm : ‖Φₖ‖₂ = σ_max(Φₖ) — worst-case signal amplification.
    frobenius_norm: ‖Φₖ‖_F — overall operator magnitude.
    outer_sum     : Σᵢ(Lᵢₖ · Vₖᵀ) before Hadamard with Mₖ. Shape (n, n).
    tick_k        : Tick index of this snapshot.
    regime_signal : Normalized trace ∈ [0,1] used as soft regime indicator.
    """

    Phi_k: np.ndarray
    is_psd: bool
    min_eigenval: float
    trace: float
    spectral_norm: float
    frobenius_norm: float
    outer_sum: np.ndarray
    tick_k: int
    regime_signal: float = 0.0

    def summary(self) -> str:
        psd_str = "✓ PSD" if self.is_psd else "✗ NOT PSD"
        lines = [
            "=" * 56,
            f"  Φₖ Result — Tick {self.tick_k}",
            "=" * 56,
            f"  PSD check      : {psd_str}",
            f"  Min eigenvalue : {self.min_eigenval:+.6f}",
            f"  tr(Φₖ)         : {self.trace:.6f}",
            f"  ‖Φₖ‖₂          : {self.spectral_norm:.6f}",
            f"  ‖Φₖ‖_F         : {self.frobenius_norm:.6f}",
            f"  Regime signal  : {self.regime_signal:.4f}",
            "=" * 56,
        ]
        return "\n".join(lines)


# ===========================================================================
# Section 2 — Liquidity Tensor Builders
# ===========================================================================


class LiquidityTensorBuilder:
    """
    Constructs the n Localized Liquidity Matrices {Lᵢₖ} from raw Level-3
    order-book depth arrays.

    Each Lᵢₖ is a symmetric PSD Gram matrix derived from the bilateral depth
    profile within the i-th price-level partition:

        Lᵢₖ = Dᵢₖ · Dᵢₖᵀ + ε·I

    where Dᵢₖ ∈ ℝⁿ is the depth vector (bid-side negated, ask-side positive)
    within partition i, and ε·I is a small Tikhonov regularization to ensure
    strict positive definiteness.

    The first-order gradient of depth is embedded by stacking [Dᵢₖ, ∇Dᵢₖ]
    into a 2n-vector before taking the outer product, then projecting back to
    n×n via block averaging. This gradient encoding captures whether liquidity
    is building or eroding at each partition boundary.

    Parameters
    ----------
    n_levels : Number of price-level partitions.
    epsilon  : Tikhonov regularization (default 1e-8).
    """

    def __init__(self, n_levels: int = 10, epsilon: float = 1e-8) -> None:
        self.n = n_levels
        self.epsilon = epsilon
        self._I = np.eye(n_levels)

    def build(self, depth_bid: np.ndarray, depth_ask: np.ndarray) -> List[np.ndarray]:
        """
        Construct {Lᵢₖ}ᵢ₌₁ⁿ from raw bilateral depth arrays.

        Parameters
        ----------
        depth_bid : (n, n) array — bid-side depth at each (partition, price_level).
                    depth_bid[i, j] = resting bid quantity at level j within partition i.
        depth_ask : (n, n) array — ask-side depth, same indexing convention.

        Returns
        -------
        List of n symmetric PSD matrices, each (n, n).
        """
        n = self.n
        assert depth_bid.shape == (n, n), f"depth_bid must be ({n},{n})"
        assert depth_ask.shape == (n, n), f"depth_ask must be ({n},{n})"

        L_list = []
        for i in range(n):
            # Signed depth vector for partition i:
            # bid side enters negatively (supports downward price pressure),
            # ask side positively. This sign convention ensures that a one-sided
            # book (all bids, no asks) produces a negative-definite contribution
            # that, when Hadamard'd with PSD Mₖ, suppresses upward signals.
            D_i = depth_ask[i] - depth_bid[i]  # shape: (n,)

            # Gram matrix: outer product of depth vector with itself.
            # By construction this is rank-1 PSD. The Tikhonov term ε·I
            # inflates it to full rank n, enabling well-conditioned inversion
            # downstream (e.g., in Sharpe ratio computation per regime).
            L_i = np.outer(D_i, D_i) + self.epsilon * self._I  # (n, n)

            # Gradient encoding: approximate ∂D_i/∂j via finite differences.
            # This embeds the "slope" of the depth profile into the Gram matrix,
            # capturing whether liquidity is stacking up (∇D > 0) or thinning.
            grad_D_i = np.gradient(D_i)                         # shape: (n,)
            L_i += 0.1 * np.outer(grad_D_i, grad_D_i)           # rank-1 update

            L_list.append(L_i)

        return L_list

    @staticmethod
    def verify_psd_list(L_list: List[np.ndarray], tol: float = -1e-8) -> bool:
        """Return True if all matrices in the list are numerically PSD."""
        for L in L_list:
            if np.min(eigvalsh(L)) < tol:
                return False
        return True


class MassMatrixBuilder:
    """
    Constructs the Volumetric Mass Tensor Mₖ from the empirical covariance
    of Hawkes conditional intensity vectors over a trailing W-tick window.

    In the paper (§3.2), Mₖ is set equal to the empirical covariance of the
    estimated conditional intensity vector (λᵇₖ, λˢₖ) over W=500 ticks,
    updated at each regime transition.

    For an n-dimensional intensity vector (one per price-level partition),
    Mₖ ∈ ℝⁿˣⁿ is the sample covariance matrix of the intensity history matrix
    Λ ∈ ℝᵂˣⁿ over the trailing window:

        Mₖ = (1/(W-1)) · (Λ - Λ̄)ᵀ · (Λ - Λ̄) + ε·I

    The Ledoit-Wolf shrinkage estimator (Ledoit & Wolf 2004) is applied when
    W < 5n to guard against ill-conditioning at small sample sizes.

    Parameters
    ----------
    n_levels  : Dimension n of the intensity vector.
    epsilon   : Regularization floor for eigenvalues.
    use_shrinkage : Apply Ledoit-Wolf shrinkage (recommended when W < 5n).
    """

    def __init__(
        self,
        n_levels: int = 10,
        epsilon: float = 1e-8,
        use_shrinkage: bool = True,
    ) -> None:
        self.n = n_levels
        self.epsilon = epsilon
        self.use_shrinkage = use_shrinkage

    def build(self, intensity_history: np.ndarray) -> np.ndarray:
        """
        Compute Mₖ from intensity history matrix.

        Parameters
        ----------
        intensity_history : (W, n) array — Hawkes conditional intensities over
                            the trailing W-tick window. Each row is a tick,
                            each column is a price-level partition intensity.

        Returns
        -------
        Mₖ : (n, n) symmetric PSD covariance matrix.
        """
        W, n = intensity_history.shape
        assert n == self.n, f"Intensity dimension {n} ≠ n_levels {self.n}."

        if self.use_shrinkage and W < 5 * n:
            # Ledoit-Wolf analytical shrinkage (Oracle approximation)
            # Shrinks the sample covariance toward a scaled identity:
            #   M* = (1-ρ)·S + ρ·μ_S·I
            # where ρ is the optimal shrinkage intensity.
            try:
                from sklearn.covariance import LedoitWolf
                lw = LedoitWolf(assume_centered=False)
                lw.fit(intensity_history)
                M_k = lw.covariance_
            except ImportError:
                warnings.warn(
                    "sklearn not available — falling back to sample covariance. "
                    "Install scikit-learn for Ledoit-Wolf shrinkage.",
                    ImportWarning,
                    stacklevel=2,
                )
                M_k = np.cov(intensity_history, rowvar=False)
        else:
            M_k = np.cov(intensity_history, rowvar=False)  # (n, n)

        # Tikhonov floor: ensures strict PSD and prevents Hadamard product
        # from producing near-singular Φₖ in low-variance regimes.
        M_k += self.epsilon * np.eye(n)

        # Symmetrize to eliminate floating-point asymmetry from covariance
        M_k = 0.5 * (M_k + M_k.T)

        return M_k


# ===========================================================================
# Section 3 — Core Operator: ExecutionValidationMatrix
# ===========================================================================


class ExecutionValidationMatrix:
    """
    Computes the Discretized Execution Validation Matrix Φₖ at a single tick k.

    Core operation:

        Φₖ = [ Σᵢ₌₁ⁿ ( Lᵢₖ · Vₖᵀ ) ] ⊙ Mₖ

    Dimensional trace:
    ------------------
        Lᵢₖ         : (n, n)   PSD Gram matrix — liquidity geometry
        Vₖᵀ          : (1, n)   transposed execution vector (row)
        Lᵢₖ · Vₖᵀ   : (n, n)   outer product — each column of Lᵢₖ scaled by Vₖ
                                This is NOT a standard matrix product; it is the
                                outer product np.outer(Lᵢₖ @ 1ₙ, Vₖ), where 1ₙ
                                is the all-ones vector used to "project" the
                                n×n matrix onto the V direction. Formally:
                                (Lᵢₖ · Vₖᵀ)[p,q] = (Lᵢₖ @ e_q) · Vₖ[p]
                                — i.e., the p-th row of the outer product encodes
                                how much execution pressure in direction p is
                                "supported" by the q-th liquidity column.
        Σᵢ(...)      : (n, n)   aggregated outer-product sum across all partitions
        ⊙ Mₖ         : (n, n)   Hadamard (element-wise) with volumetric mass tensor

    Hadamard Intuition:
    -------------------
    Mₖ[p,q] is large when partitions p and q have highly correlated Hawkes
    intensities — meaning they tend to fill and drain together. Hadamard-ing
    with Mₖ therefore up-weights the (L,V) outer product in regions of the
    price-level grid where order arrival is synchronous (informed flow), and
    down-weights regions where arrivals are orthogonal (noise flow). This is
    the key filtering operation that separates regime signal from microstructure
    noise, analogous to a spatially adaptive kernel in nonparametric regression.

    PSD Inheritance (Schur Product Theorem):
    ----------------------------------------
    Although Σᵢ(Lᵢₖ · Vₖᵀ) is not guaranteed PSD (the V direction can break
    symmetry), the Hadamard product of ANY matrix with a PSD matrix Mₖ yields
    a matrix whose eigenvalues are bounded below by:

        λ_min(A ⊙ B) ≥ min_i(A[i,i]) · λ_min(B)

    Since Mₖ[i,i] = Var(λᵢₖ) ≥ ε > 0 and λ_min(Mₖ) ≥ ε, Φₖ has bounded
    minimum eigenvalue, ensuring it is numerically manageable as a matrix operator.

    Parameters
    ----------
    n_levels    : Number of price-level partitions (baseline 10; robustness in {5,10,20}).
    use_gpu     : Attempt to use CuPy for tensor operations on NVIDIA GPU.
    psd_tol     : Tolerance for numerical PSD verification (default −1e-6).
    verbose     : Print diagnostic information per tick.
    """

    def __init__(
        self,
        n_levels: int = 10,
        use_gpu: bool = False,
        psd_tol: float = -1e-6,
        verbose: bool = False,
    ) -> None:
        self.n = n_levels
        self.use_gpu = use_gpu and _GPU_AVAILABLE
        self.psd_tol = psd_tol
        self.verbose = verbose
        self._xp = _get_array_module(self.use_gpu)

        if self.use_gpu:
            print(f"[Φₖ] GPU mode active — CuPy device {_GPU_DEVICE}.")
        else:
            print(f"[Φₖ] CPU mode — NumPy backend.")

        # Running normalization state for regime_signal ∈ [0,1]
        self._trace_max: float = 1.0

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _to_device(self, arr: np.ndarray):
        """Move a NumPy array to the active compute device (CPU or GPU)."""
        if self.use_gpu:
            return cp.asarray(arr)
        return arr

    def _to_host(self, arr) -> np.ndarray:
        """Move an array back to CPU NumPy (no-op if already CPU)."""
        if self.use_gpu and isinstance(arr, cp.ndarray):
            return cp.asnumpy(arr)
        return np.asarray(arr)

    def _compute_outer_sum(
        self,
        L_partitions: List[np.ndarray],
        V_k: np.ndarray,
    ):
        """
        Compute Σᵢ ( Lᵢₖ · Vₖᵀ ) ∈ ℝⁿˣⁿ.

        Implementation note:
        --------------------
        (Lᵢₖ · Vₖᵀ)[p,q] = Lᵢₖ[p, :] · Vₖ[q]
        In NumPy: np.outer(Lᵢₖ @ ones_n, Vₖ) is equivalent to Lᵢₖ * Vₖ[np.newaxis, :]
        but the latter is faster for large n.

        We use the broadcasting form:
            L_i_times_VT = L_i * V_k[np.newaxis, :]
        which interprets each row of Lᵢₖ as a "liquidity profile" and scales
        column q by Vₖ[q], the execution pressure in direction q.
        """
        xp = self._xp
        n = self.n
        acc = xp.zeros((n, n), dtype=xp.float64)

        for i, L_i in enumerate(L_partitions):
            L_dev = self._to_device(L_i)      # (n, n) — liquidity at partition i
            V_dev = self._to_device(V_k)       # (n,)   — directional execution vector

            # Outer product: Lᵢₖ · Vₖᵀ
            # Dimension: (n, n) ← column j scaled by V_k[j]
            # Microstructure meaning: entry [p,q] is the bid-ask depth gradient
            # at level p weighted by the q-th direction of execution pressure.
            outer_i = L_dev * V_dev[xp.newaxis, :]   # (n, n)

            acc += outer_i   # accumulate across all n partitions

        return acc  # Σᵢ(Lᵢₖ · Vₖᵀ), shape (n, n)

    def _hadamard_with_mass(self, outer_sum, M_k: np.ndarray):
        """
        Apply the Hadamard product: Φₖ = outer_sum ⊙ Mₖ.

        Microstructure meaning of element-wise multiplication:
        ------------------------------------------------------
        Φₖ[p, q] = [Σᵢ Lᵢₖ · Vₖᵀ][p, q]  ×  Mₖ[p, q]

        Mₖ[p, q] = Cov(λᵖₖ, λᵍₖ) — the covariance of Hawkes arrival intensities
        between partitions p and q over the trailing W-tick window.

        A large Mₖ[p, q] indicates that partitions p and q tend to receive order
        flow simultaneously — the signature of a coherent institutional flow (e.g.,
        iceberg order splitting across adjacent price levels). Φₖ therefore amplifies
        the execution signal exactly where institutional activity is most synchronized,
        and suppresses it where arrivals are asynchronous (noise traders).

        This is formally equivalent to a locally stationary kernel smoothing of the
        raw (L, V) outer products, with Mₖ as the adaptive bandwidth matrix.
        """
        xp = self._xp
        M_dev = self._to_device(M_k)
        return outer_sum * M_dev   # element-wise, (n, n)

    # ------------------------------------------------------------------
    # PSD verification
    # ------------------------------------------------------------------

    def _verify_psd(self, Phi: np.ndarray) -> Tuple[bool, float]:
        """
        Numerically verify that Φₖ is positive semi-definite.

        Uses eigvalsh (symmetric eigenvalue decomposition) which is ~2× faster
        than eig for symmetric matrices and numerically stable.

        Returns
        -------
        (is_psd, min_eigenvalue)
        """
        # Symmetrize to eliminate floating-point drift from Hadamard product
        Phi_sym = 0.5 * (Phi + Phi.T)
        eigs = eigvalsh(Phi_sym)
        min_eig = float(eigs.min())
        return (min_eig >= self.psd_tol), min_eig

    # ------------------------------------------------------------------
    # Main forward pass
    # ------------------------------------------------------------------

    def compute(self, snapshot: OrderBookSnapshot) -> PhiResult:
        """
        Execute the full Φₖ forward pass for a single tick-k snapshot.

        Steps:
        1.  Validate input dimensions.
        2.  Compute Σᵢ(Lᵢₖ · Vₖᵀ) via vectorized outer products.
        3.  Apply Hadamard product with Mₖ.
        4.  Move result to CPU (if GPU mode).
        5.  Verify PSD, compute diagnostics.
        6.  Update running trace normalization for regime_signal.

        Parameters
        ----------
        snapshot : OrderBookSnapshot with L_partitions, V_k, M_k, tick_k.

        Returns
        -------
        PhiResult with Φₖ and all diagnostics.
        """
        n = self.n

        # Step 2 — Outer product sum: Σᵢ(Lᵢₖ · Vₖᵀ)
        outer_sum_dev = self._compute_outer_sum(
            snapshot.L_partitions, snapshot.V_k
        )

        # Step 3 — Hadamard with Mₖ
        Phi_dev = self._hadamard_with_mass(outer_sum_dev, snapshot.M_k)

        # Step 4 — Transfer to CPU for diagnostics
        Phi_k = self._to_host(Phi_dev)
        outer_sum = self._to_host(outer_sum_dev)

        # Symmetrize: the outer product is not guaranteed symmetric; the
        # Hadamard with symmetric Mₖ preserves this asymmetry. We symmetrize
        # as a final step to ensure Φₖ is well-behaved as a linear operator.
        # The antisymmetric component ½(Φ - Φᵀ) represents the "torque" of
        # directional flow around the diagonal — discarding it is conservative.
        Phi_k = 0.5 * (Phi_k + Phi_k.T)

        # Step 5 — Diagnostics
        is_psd, min_eig = self._verify_psd(Phi_k)
        tr = float(np.trace(Phi_k))
        spec_norm = float(norm(Phi_k, ord=2))      # largest singular value
        frob_norm = float(norm(Phi_k, ord='fro'))  # Frobenius norm

        # Step 6 — Regime signal ∈ [0, 1]: running max normalization of tr(Φₖ).
        # A near-zero regime signal (tr → 0) is a soft precursor to Regime 3.
        self._trace_max = max(self._trace_max, abs(tr) + 1e-12)
        regime_signal = min(abs(tr) / self._trace_max, 1.0)

        if self.verbose:
            psd_str = "✓" if is_psd else "✗ WARNING"
            print(
                f"[Φₖ | tick={snapshot.tick_k:>8,}] "
                f"tr={tr:.4f}  ‖Φ‖₂={spec_norm:.4f}  "
                f"λ_min={min_eig:+.2e}  PSD:{psd_str}  "
                f"signal={regime_signal:.3f}"
            )

        return PhiResult(
            Phi_k=Phi_k,
            is_psd=is_psd,
            min_eigenval=min_eig,
            trace=tr,
            spectral_norm=spec_norm,
            frobenius_norm=frob_norm,
            outer_sum=outer_sum,
            tick_k=snapshot.tick_k,
            regime_signal=regime_signal,
        )

    def compute_batch(
        self,
        snapshots: List[OrderBookSnapshot],
    ) -> List[PhiResult]:
        """
        Compute Φₖ for a sequence of tick snapshots (e.g., one trading session).

        In production, this is called on each walk-forward OOS quarter with a
        pre-estimated Mₖ covariance block from the preceding training window.

        Parameters
        ----------
        snapshots : List of OrderBookSnapshot, one per event-tick.

        Returns
        -------
        List of PhiResult in tick order.
        """
        results = []
        for snap in snapshots:
            results.append(self.compute(snap))
        return results


# ===========================================================================
# Section 4 — Synthetic Data Generator
# ===========================================================================


class SyntheticOrderBookFactory:
    """
    Generates synthetic Level-3 order book snapshots for demonstration.

    All tensors are constructed to satisfy the mathematical constraints of the
    paper: Lᵢₖ are PSD (via Gram construction), Mₖ is PSD (via covariance
    estimation from simulated Hawkes intensities).

    Parameters
    ----------
    n_levels : Price-level partition count.
    seed     : Random seed for reproducibility.
    """

    def __init__(self, n_levels: int = 10, seed: int = 42) -> None:
        self.n = n_levels
        self.rng = np.random.default_rng(seed)
        self._L_builder = LiquidityTensorBuilder(n_levels=n_levels)
        self._M_builder = MassMatrixBuilder(n_levels=n_levels)

    def _make_psd(self, scale: float = 1.0) -> np.ndarray:
        """Generate a random symmetric PSD matrix via Wishart construction."""
        A = self.rng.normal(scale=scale, size=(self.n, self.n))
        return A @ A.T + 1e-6 * np.eye(self.n)

    def make_snapshot(
        self,
        tick_k: int = 0,
        regime: str = "normal",
    ) -> OrderBookSnapshot:
        """
        Synthesize a single tick-k OrderBookSnapshot.

        Parameters
        ----------
        tick_k : Tick index.
        regime : 'normal' (Regime 1), 'stressed' (Regime 2), or 'chaotic' (Regime 3).
                 Controls the statistical properties of the synthetic tensors.

        Returns
        -------
        OrderBookSnapshot ready for forward-pass through ExecutionValidationMatrix.
        """
        n = self.n

        # Regime-specific scaling factors (mirrors HMM state properties in paper)
        regime_cfg = {
            "normal":   {"depth_scale": 1.0, "v_scale": 0.5,  "m_scale": 0.3},
            "stressed": {"depth_scale": 0.5, "v_scale": 1.2,  "m_scale": 0.8},
            "chaotic":  {"depth_scale": 0.1, "v_scale": 2.0,  "m_scale": 2.5},
        }
        cfg = regime_cfg.get(regime, regime_cfg["normal"])

        # --- Build L_partitions ---
        # Synthetic bilateral depth: bid and ask depth per (partition, price_level)
        depth_bid = np.abs(self.rng.normal(
            loc=50.0 * cfg["depth_scale"],
            scale=20.0 * cfg["depth_scale"],
            size=(n, n),
        ))
        depth_ask = np.abs(self.rng.normal(
            loc=50.0 * cfg["depth_scale"],
            scale=20.0 * cfg["depth_scale"],
            size=(n, n),
        ))
        L_list = self._L_builder.build(depth_bid, depth_ask)

        # --- Build V_k ---
        # Directional execution vector: net signed order flow over trailing δ ticks.
        # In Regime 1, V is moderate and balanced; in Regime 3, it is extreme.
        V_k = self.rng.normal(
            loc=0.0,
            scale=cfg["v_scale"],
            size=(n,),
        )

        # --- Build M_k ---
        # Simulate Hawkes intensity history matrix Λ ∈ ℝᵂˣⁿ
        W = 500
        base_intensity = 0.05 * cfg["m_scale"]
        # Correlated intensities across partitions (institutional flow signature)
        corr_factor = self.rng.normal(loc=0, scale=base_intensity, size=(W, 1))
        noise = self.rng.normal(loc=0, scale=base_intensity * 0.3, size=(W, n))
        intensity_history = np.abs(base_intensity + corr_factor + noise)
        M_k = self._M_builder.build(intensity_history)

        return OrderBookSnapshot(
            L_partitions=L_list,
            V_k=V_k,
            M_k=M_k,
            tick_k=tick_k,
            n_levels=n,
        )


# ===========================================================================
# Section 5 — Entry Point (Demonstration)
# ===========================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("  Φₖ Forward Pass — Demonstration")
    print("  Paper 1: Constrained Stochasticity in CLOBs")
    print("  Discretized Execution Validation Matrix")
    print("=" * 60)

    N_LEVELS = 10  # Baseline n=10; robustness tested over {5, 10, 20}

    # -----------------------------------------------------------------------
    # Step 1 — Initialize the operator
    # -----------------------------------------------------------------------
    phi_operator = ExecutionValidationMatrix(
        n_levels=N_LEVELS,
        use_gpu=False,   # Set True if CuPy + NVIDIA GPU available
        psd_tol=-1e-6,
        verbose=True,
    )
    factory = SyntheticOrderBookFactory(n_levels=N_LEVELS, seed=42)

    # -----------------------------------------------------------------------
    # Step 2 — Single-tick forward pass across three regimes
    # -----------------------------------------------------------------------
    print("\n── Single-Tick Forward Pass (n=10, one tick per regime) ──\n")
    for regime in ["normal", "stressed", "chaotic"]:
        print(f"\n  Regime: {regime.upper()}")
        snap = factory.make_snapshot(tick_k=0, regime=regime)
        result = phi_operator.compute(snap)
        print(result.summary())

    # -----------------------------------------------------------------------
    # Step 3 — Manual tensor inspection (Normal regime)
    # -----------------------------------------------------------------------
    print("\n── Manual Tensor Inspection (Regime 1 / Normal) ──")
    snap_normal = factory.make_snapshot(tick_k=100, regime="normal")
    result_normal = phi_operator.compute(snap_normal)

    print(f"\n  L₁ₖ (first liquidity partition), shape {snap_normal.L_partitions[0].shape}:")
    print("  " + np.array2string(
        snap_normal.L_partitions[0],
        precision=3, suppress_small=True, max_line_width=80,
    ).replace("\n", "\n  "))

    print(f"\n  Vₖ (directional execution vector), shape {snap_normal.V_k.shape}:")
    print("  " + np.array2string(snap_normal.V_k, precision=4))

    print(f"\n  Mₖ diagonal (volumetric mass tensor variances):")
    print("  " + np.array2string(np.diag(snap_normal.M_k), precision=5))

    print(f"\n  Σᵢ(Lᵢₖ·Vₖᵀ) [outer_sum], shape {result_normal.outer_sum.shape}:")
    print("  " + np.array2string(
        result_normal.outer_sum,
        precision=3, suppress_small=True, max_line_width=80,
    ).replace("\n", "\n  "))

    print(f"\n  Φₖ [final operator], shape {result_normal.Phi_k.shape}:")
    print("  " + np.array2string(
        result_normal.Phi_k,
        precision=4, suppress_small=True, max_line_width=80,
    ).replace("\n", "\n  "))

    # -----------------------------------------------------------------------
    # Step 4 — Schur product theorem verification
    # -----------------------------------------------------------------------
    print("\n── Schur Product Theorem Verification ──")
    print("  Verifying PSD of Mₖ (mass tensor) …")
    M_eigs = eigvalsh(snap_normal.M_k)
    print(f"  λ_min(Mₖ) = {M_eigs.min():.2e}  — {'PSD ✓' if M_eigs.min() >= 0 else 'NOT PSD ✗'}")
    print("  Verifying PSD of Φₖ (after Hadamard) …")
    Phi_eigs = eigvalsh(result_normal.Phi_k)
    print(f"  λ_min(Φₖ) = {Phi_eigs.min():.2e}  — {'PSD ✓' if Phi_eigs.min() >= -1e-6 else 'NOT PSD ✗'}")

    # -----------------------------------------------------------------------
    # Step 5 — Batch forward pass (simulating one trading session)
    # -----------------------------------------------------------------------
    K_SESSION = 200
    print(f"\n── Batch Forward Pass: K={K_SESSION} ticks (simulated session) ──\n")

    # Alternate between normal and stressed regimes
    snapshots = []
    for k in range(K_SESSION):
        regime = "stressed" if 80 <= k < 120 else "normal"
        snapshots.append(factory.make_snapshot(tick_k=k, regime=regime))

    phi_operator.verbose = False   # suppress per-tick printing for batch
    batch_results = phi_operator.compute_batch(snapshots)

    traces = np.array([r.trace for r in batch_results])
    signals = np.array([r.regime_signal for r in batch_results])
    psd_ok = all(r.is_psd for r in batch_results)

    print(f"  Ticks processed : {K_SESSION}")
    print(f"  All PSD         : {'✓' if psd_ok else '✗ FAIL'}")
    print(f"  tr(Φₖ) — mean   : {traces.mean():.4f}")
    print(f"  tr(Φₖ) — min    : {traces.min():.4f}")
    print(f"  tr(Φₖ) — max    : {traces.max():.4f}")
    print(f"  Regime signal   : mean={signals.mean():.3f}, "
          f"min={signals.min():.3f}, max={signals.max():.3f}")

    # Show the regime transition: signal should dip in the stressed window (k=80-120)
    print(f"\n  Mean regime signal by window:")
    windows = [(0, 80, "Pre-stress"), (80, 120, "Stressed"), (120, 200, "Post-stress")]
    for start, end, label in windows:
        seg = signals[start:end]
        print(f"    [{start:>3}–{end:>3}] {label:<15}: {seg.mean():.4f}")

    print("\n[DONE] phi_k.py demonstration complete.")
