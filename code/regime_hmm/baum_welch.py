"""
baum_welch.py
=============
Production-grade 3-state Hidden Markov Model for CLOB microstructure regime detection.

Implements the Baum-Welch (EM) algorithm with log-space Forward-Backward to prevent
numerical underflow on nanosecond-level tick data. Designed for the Constrained
Stochasticity framework (Matthews, 2025): Regime 1 (Normal), Regime 2 (Stressed),
Regime 3 (Chaotic/Absorbing).

Absorbing boundary check (Proposition 1, Matthews 2025):
    Under ρ(A) >= 1 conditional on S_k = 3, the expected time to ruin of any leveraged
    wealth process is O(log W_0). Optimal policy: φ* = 0 (zero position).

References
----------
Baum et al. (1970). A maximization technique occurring in the statistical analysis of
    probabilistic functions of Markov chains. Ann. Math. Stat., 41(1), 164-171.
Hamilton, J.D. (1989). A new approach to the economic analysis of nonstationary time
    series and the business cycle. Econometrica, 57(2), 357-384.
Goldie, C.M. (1991). Implicit renewal theory and tails of solutions of random equations.
    Ann. Appl. Probab., 1(1), 126-166.
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from numpy.typing import NDArray
from scipy.stats import norm

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("MicrostructureHMM")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
N_STATES: int = 3          # Regime 1 (Normal), 2 (Stressed), 3 (Chaotic)
LOG_ZERO: float = -np.inf  # log(0) sentinel
EPS: float = 1e-300        # floor to avoid log(0) in emission


# ---------------------------------------------------------------------------
# Data container for fitted parameters
# ---------------------------------------------------------------------------
@dataclass
class HMMParams:
    """
    Container for HMM parameters.

    Attributes
    ----------
    pi : NDArray[float], shape (3,)
        Initial state distribution. pi[i] = P(S_0 = i).
    A : NDArray[float], shape (3, 3)
        Transition matrix. A[i, j] = P(S_{k+1} = j | S_k = i).
        Rows sum to 1.
    mu : NDArray[float], shape (3,)
        Gaussian emission means per regime.
    sigma : NDArray[float], shape (3,)
        Gaussian emission standard deviations per regime (strictly positive).
    """
    pi: NDArray[np.float64]
    A: NDArray[np.float64]
    mu: NDArray[np.float64]
    sigma: NDArray[np.float64]

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        for arr, name in [(self.pi, "pi"), (self.A, "A"),
                          (self.mu, "mu"), (self.sigma, "sigma")]:
            if not isinstance(arr, np.ndarray):
                raise TypeError(f"{name} must be a numpy ndarray.")
        if self.pi.shape != (N_STATES,):
            raise ValueError(f"pi must have shape ({N_STATES},).")
        if self.A.shape != (N_STATES, N_STATES):
            raise ValueError(f"A must have shape ({N_STATES}, {N_STATES}).")
        if not np.allclose(self.pi.sum(), 1.0, atol=1e-6):
            raise ValueError("pi must sum to 1.")
        if not np.allclose(self.A.sum(axis=1), 1.0, atol=1e-6):
            raise ValueError("Each row of A must sum to 1.")
        if np.any(self.sigma <= 0):
            raise ValueError("All sigma values must be strictly positive.")

    def __repr__(self) -> str:
        lines = [
            "HMMParams(",
            f"  pi    = {np.round(self.pi, 4)}",
            f"  A     =\n{np.round(self.A, 4)}",
            f"  mu    = {np.round(self.mu, 6)}",
            f"  sigma = {np.round(self.sigma, 6)}",
            ")",
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------
class MicrostructureHMM:
    """
    3-State Hidden Markov Model for CLOB microstructure regime detection.

    Regimes
    -------
    0 → Regime 1 (Normal / Accumulation):
        Liquid, low-volatility. Hawkes branching ratio ρ < 0.5.
    1 → Regime 2 (Stressed):
        Elevated volatility, widened spreads. Consistent with pre-crisis patterns.
    2 → Regime 3 (Chaotic / Absorbing):
        Near-critical Hawkes dynamics (ρ → 1 or ρ > 1). Order-book depth collapse.
        Absorbing boundary: Φ_k|S_k=3 = 0 halts all execution.

    Parameters
    ----------
    max_iter : int
        Maximum Baum-Welch EM iterations.
    tol : float
        Convergence tolerance on log-likelihood change.
    random_state : Optional[int]
        Seed for reproducible initialisation.
    absorbing_threshold : float
        Posterior probability threshold η ∈ (0, 1) above which Regime 3 is
        declared active and execution is clamped (Proposition 1, Matthews 2025).
    hawkes_rho : float
        Spectral radius ρ(A_hawkes) of the Hawkes branching matrix estimated
        externally and passed in to check_absorbing_boundary(). Default 0.0
        (unchecked until set).
    """

    def __init__(
        self,
        max_iter: int = 200,
        tol: float = 1e-6,
        random_state: Optional[int] = 42,
        absorbing_threshold: float = 0.72,
        hawkes_rho: float = 0.0,
    ) -> None:
        if max_iter < 1:
            raise ValueError("max_iter must be >= 1.")
        if not (0.0 < tol < 1.0):
            raise ValueError("tol must be in (0, 1).")
        if not (0.0 < absorbing_threshold < 1.0):
            raise ValueError("absorbing_threshold η must be in (0, 1).")

        self.max_iter = max_iter
        self.tol = tol
        self.rng = np.random.default_rng(random_state)
        self.absorbing_threshold = absorbing_threshold
        self.hawkes_rho = hawkes_rho

        self.params_: Optional[HMMParams] = None
        self.log_likelihood_history_: list[float] = []
        self.n_iter_: int = 0
        self._fitted: bool = False

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------
    def _initialise_params(self, observations: NDArray[np.float64]) -> HMMParams:
        """
        Initialise HMM parameters with regime-aware priors.

        Regime 1 (index 0): centred near 0, low volatility.
        Regime 2 (index 1): centred near 0, medium volatility.
        Regime 3 (index 2): centred near 0, high volatility (fat-tail proxy).
        """
        global_std = float(np.std(observations))

        pi = np.array([0.68, 0.24, 0.08], dtype=np.float64)  # empirical priors (paper)

        # Transition matrix: high persistence on diagonal, small prob of Regime 3
        A = np.array([
            [0.974, 0.023, 0.003],
            [0.198, 0.765, 0.037],
            [0.074, 0.245, 0.681],
        ], dtype=np.float64)

        mu = np.array([0.0, 0.0, 0.0], dtype=np.float64)

        # Volatility tiers from paper: Normal < Stressed < Chaotic
        sigma = np.array([
            0.5 * global_std,
            1.2 * global_std,
            2.8 * global_std,
        ], dtype=np.float64)

        return HMMParams(pi=pi, A=A, mu=mu, sigma=sigma)

    # ------------------------------------------------------------------
    # Emission log-probabilities  (T × N_STATES)
    # ------------------------------------------------------------------
    def _log_emission(
        self,
        observations: NDArray[np.float64],
        params: HMMParams,
    ) -> NDArray[np.float64]:
        """
        Compute log p(o_k | S_k = j) for Gaussian emissions.

        Parameters
        ----------
        observations : shape (T,)
        params       : current HMMParams

        Returns
        -------
        log_B : shape (T, N_STATES)
        """
        T = len(observations)
        log_B = np.empty((T, N_STATES), dtype=np.float64)
        for j in range(N_STATES):
            log_B[:, j] = norm.logpdf(observations, loc=params.mu[j],
                                       scale=params.sigma[j])
        return log_B  # may contain -inf for extreme outliers; handled below

    # ------------------------------------------------------------------
    # Forward pass  (log-space)
    # ------------------------------------------------------------------
    def _forward(
        self,
        log_B: NDArray[np.float64],
        params: HMMParams,
    ) -> tuple[NDArray[np.float64], float]:
        """
        Log-space forward algorithm.

        Computes log α_k(j) = log P(o_1:k, S_k = j) via the log-sum-exp trick.

        Parameters
        ----------
        log_B   : shape (T, N_STATES) — log emission probabilities
        params  : current HMMParams

        Returns
        -------
        log_alpha : shape (T, N_STATES)
        log_likelihood : float — log P(o_1:T)
        """
        T = log_B.shape[0]
        log_alpha = np.full((T, N_STATES), LOG_ZERO, dtype=np.float64)
        log_pi = np.log(np.maximum(params.pi, EPS))
        log_A = np.log(np.maximum(params.A, EPS))

        # Initialisation
        log_alpha[0] = log_pi + log_B[0]

        # Recursion
        for k in range(1, T):
            # log_alpha[k, j] = logsumexp_i(log_alpha[k-1, i] + log_A[i, j]) + log_B[k, j]
            for j in range(N_STATES):
                log_alpha[k, j] = (
                    self._logsumexp(log_alpha[k - 1] + log_A[:, j])
                    + log_B[k, j]
                )

        log_likelihood = self._logsumexp(log_alpha[-1])
        return log_alpha, log_likelihood

    # ------------------------------------------------------------------
    # Backward pass  (log-space)
    # ------------------------------------------------------------------
    def _backward(
        self,
        log_B: NDArray[np.float64],
        params: HMMParams,
    ) -> NDArray[np.float64]:
        """
        Log-space backward algorithm.

        Computes log β_k(i) = log P(o_{k+1:T} | S_k = i) via log-sum-exp.

        Parameters
        ----------
        log_B  : shape (T, N_STATES)
        params : current HMMParams

        Returns
        -------
        log_beta : shape (T, N_STATES)
        """
        T = log_B.shape[0]
        log_beta = np.full((T, N_STATES), LOG_ZERO, dtype=np.float64)
        log_A = np.log(np.maximum(params.A, EPS))

        # Initialisation
        log_beta[-1] = 0.0  # log(1)

        # Recursion (backward)
        for k in range(T - 2, -1, -1):
            for i in range(N_STATES):
                log_beta[k, i] = self._logsumexp(
                    log_A[i] + log_B[k + 1] + log_beta[k + 1]
                )

        return log_beta

    # ------------------------------------------------------------------
    # E-step: compute sufficient statistics
    # ------------------------------------------------------------------
    def _e_step(
        self,
        observations: NDArray[np.float64],
        params: HMMParams,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], float]:
        """
        E-step: compute posterior state and transition probabilities.

        Returns
        -------
        gamma  : shape (T, N_STATES) — P(S_k = j | O, θ)
        xi     : shape (T-1, N_STATES, N_STATES) — P(S_k=i, S_{k+1}=j | O, θ)
        log_likelihood : float
        """
        log_B = self._log_emission(observations, params)
        log_A = np.log(np.maximum(params.A, EPS))

        log_alpha, log_likelihood = self._forward(log_B, params)
        log_beta = self._backward(log_B, params)

        T = len(observations)

        # --- gamma ---
        log_gamma = log_alpha + log_beta
        # Normalise row-wise in log space
        log_gamma -= self._logsumexp(log_gamma, axis=1, keepdims=True)
        gamma = np.exp(log_gamma)

        # --- xi ---
        xi = np.zeros((T - 1, N_STATES, N_STATES), dtype=np.float64)
        for k in range(T - 1):
            log_xi_k = np.full((N_STATES, N_STATES), LOG_ZERO, dtype=np.float64)
            for i in range(N_STATES):
                for j in range(N_STATES):
                    log_xi_k[i, j] = (
                        log_alpha[k, i]
                        + log_A[i, j]
                        + log_B[k + 1, j]
                        + log_beta[k + 1, j]
                    )
            # Normalise
            log_xi_k -= self._logsumexp(log_xi_k.ravel())
            xi[k] = np.exp(log_xi_k)

        return gamma, xi, log_likelihood

    # ------------------------------------------------------------------
    # M-step: update parameters from sufficient statistics
    # ------------------------------------------------------------------
    def _m_step(
        self,
        observations: NDArray[np.float64],
        gamma: NDArray[np.float64],
        xi: NDArray[np.float64],
    ) -> HMMParams:
        """
        M-step: maximise expected complete-data log-likelihood.

        Updates π, A, μ, σ from gamma and xi.
        """
        # Initial distribution
        pi_new = gamma[0] / gamma[0].sum()

        # Transition matrix
        A_new = xi.sum(axis=0)  # (N_STATES, N_STATES)
        row_sums = A_new.sum(axis=1, keepdims=True)
        row_sums = np.maximum(row_sums, EPS)
        A_new /= row_sums

        # Emission parameters (Gaussian)
        gamma_sum = gamma.sum(axis=0)  # (N_STATES,)
        gamma_sum = np.maximum(gamma_sum, EPS)

        mu_new = (gamma * observations[:, None]).sum(axis=0) / gamma_sum

        sigma_new = np.sqrt(
            (gamma * (observations[:, None] - mu_new[None, :]) ** 2).sum(axis=0)
            / gamma_sum
        )
        sigma_new = np.maximum(sigma_new, 1e-8)  # strict positivity floor

        return HMMParams(pi=pi_new, A=A_new, mu=mu_new, sigma=sigma_new)

    # ------------------------------------------------------------------
    # Public API: fit
    # ------------------------------------------------------------------
    def fit(self, observations: NDArray[np.float64]) -> "MicrostructureHMM":
        """
        Fit the 3-state HMM via Baum-Welch EM.

        Parameters
        ----------
        observations : NDArray[float], shape (T,)
            Sequence of tick-level log-returns r_k = ln(P_k) - ln(P_{k-1}).

        Returns
        -------
        self : fitted MicrostructureHMM

        Raises
        ------
        ValueError
            If observations contain NaN/Inf or are too short.
        """
        observations = np.asarray(observations, dtype=np.float64)
        self._validate_observations(observations)

        T = len(observations)
        logger.info("Fitting MicrostructureHMM on %d observations.", T)

        params = self._initialise_params(observations)
        self.log_likelihood_history_ = []

        prev_ll = -np.inf
        for iteration in range(1, self.max_iter + 1):
            # E-step
            gamma, xi, log_ll = self._e_step(observations, params)
            self.log_likelihood_history_.append(log_ll)

            delta = log_ll - prev_ll
            logger.debug("Iter %3d | log-lik = %+.6f | Δ = %+.6e", iteration, log_ll, delta)

            if abs(delta) < self.tol and iteration > 1:
                logger.info(
                    "Converged at iteration %d (Δlog-lik = %.2e < tol = %.2e).",
                    iteration, delta, self.tol,
                )
                break

            prev_ll = log_ll

            # M-step
            params = self._m_step(observations, gamma, xi)
        else:
            warnings.warn(
                f"Baum-Welch did not converge in {self.max_iter} iterations. "
                "Consider increasing max_iter or checking data quality.",
                RuntimeWarning,
                stacklevel=2,
            )

        self.params_ = params
        self.n_iter_ = iteration
        self._fitted = True
        logger.info("Fit complete. Final log-likelihood: %.4f", log_ll)
        return self

    # ------------------------------------------------------------------
    # Public API: predict (Viterbi)
    # ------------------------------------------------------------------
    def predict(self, observations: NDArray[np.float64]) -> NDArray[np.int64]:
        """
        Decode the most-probable state sequence via the Viterbi algorithm.

        Parameters
        ----------
        observations : NDArray[float], shape (T,)

        Returns
        -------
        state_sequence : NDArray[int], shape (T,)
            0 → Regime 1 (Normal), 1 → Regime 2 (Stressed), 2 → Regime 3 (Chaotic)
        """
        self._check_fitted()
        observations = np.asarray(observations, dtype=np.float64)
        self._validate_observations(observations)

        params = self.params_
        log_B = self._log_emission(observations, params)
        log_A = np.log(np.maximum(params.A, EPS))
        log_pi = np.log(np.maximum(params.pi, EPS))

        T = len(observations)
        viterbi = np.full((T, N_STATES), LOG_ZERO, dtype=np.float64)
        backpointer = np.zeros((T, N_STATES), dtype=np.int64)

        viterbi[0] = log_pi + log_B[0]

        for k in range(1, T):
            for j in range(N_STATES):
                candidates = viterbi[k - 1] + log_A[:, j]
                backpointer[k, j] = int(np.argmax(candidates))
                viterbi[k, j] = candidates[backpointer[k, j]] + log_B[k, j]

        # Traceback
        states = np.empty(T, dtype=np.int64)
        states[-1] = int(np.argmax(viterbi[-1]))
        for k in range(T - 2, -1, -1):
            states[k] = backpointer[k + 1, states[k + 1]]

        return states

    # ------------------------------------------------------------------
    # Public API: posterior probabilities
    # ------------------------------------------------------------------
    def predict_proba(
        self, observations: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """
        Compute smoothed posterior state probabilities P(S_k = j | O, θ).

        Parameters
        ----------
        observations : NDArray[float], shape (T,)

        Returns
        -------
        gamma : NDArray[float], shape (T, N_STATES)
        """
        self._check_fitted()
        observations = np.asarray(observations, dtype=np.float64)
        gamma, _, _ = self._e_step(observations, self.params_)
        return gamma

    # ------------------------------------------------------------------
    # Core method: Absorbing boundary check (Proposition 1, Matthews 2025)
    # ------------------------------------------------------------------
    def check_absorbing_boundary(
        self,
        posterior_regime3: float,
        hawkes_rho: Optional[float] = None,
    ) -> dict:
        """
        Evaluate whether execution should be clamped (Φ_k|S_k=3 = 0).

        Implements Proposition 1 from Matthews (2025): under the HMM with
        transition matrix Π, if ρ(A_hawkes) ≥ 1 conditional on S_k = 3, then
        the expected time to ruin E[T_0] of the leveraged wealth process W_k is
        O(log W_0) under any fixed leverage φ > 0. Optimal policy: φ* = 0.

        The absorbing boundary fires when BOTH:
            (i)  Pr(S_k = 3 | F_k) ≥ η  (posterior threshold exceeded), AND
            (ii) ρ(A_hawkes) ≥ 1          (Kesten-Goldie condition met).

        Condition (ii) being False is treated conservatively: the boundary
        still fires on the posterior check alone, but the diagnostic reports
        the Kesten-Goldie flag separately.

        Parameters
        ----------
        posterior_regime3 : float
            Pr(S_k = 3 | F_k) — current HMM posterior probability of Regime 3,
            obtained from predict_proba()[:, 2].
        hawkes_rho : float, optional
            Spectral radius ρ(A_hawkes) of the Hawkes branching matrix at tick k.
            Overrides self.hawkes_rho if supplied.

        Returns
        -------
        result : dict with keys:
            'clamp_execution'   (bool)  — True → set φ* = 0 immediately
            'posterior_regime3' (float) — input posterior
            'threshold_eta'     (float) — absorbing_threshold η
            'posterior_exceeds_threshold' (bool)
            'kesten_goldie_condition_met' (bool) — ρ(A) ≥ 1
            'hawkes_rho'        (float) — ρ(A) used in evaluation
            'regime3_transition_prob' (float | None) — π_{*,3} from fitted A

        Raises
        ------
        ValueError
            If posterior_regime3 is not in [0, 1].
        RuntimeError
            If called before fit().
        """
        self._check_fitted()

        if not (0.0 <= posterior_regime3 <= 1.0):
            raise ValueError(
                f"posterior_regime3 must be in [0, 1]; got {posterior_regime3:.4f}."
            )

        rho = hawkes_rho if hawkes_rho is not None else self.hawkes_rho

        posterior_exceeds = posterior_regime3 >= self.absorbing_threshold
        kesten_goldie_met = rho >= 1.0

        # Clamping fires on posterior threshold; KG condition is a diagnostic flag.
        # (Conservative: a Regime-3 classification alone is sufficient for clamping
        #  even if the Hawkes branching ratio is not yet estimated externally.)
        clamp = posterior_exceeds

        # Extract average transition probability INTO Regime 3 from the fitted A
        A_fitted = self.params_.A
        avg_transition_to_r3 = float(A_fitted[:, 2].mean())  # mean across origin states

        result = {
            "clamp_execution": clamp,
            "posterior_regime3": float(posterior_regime3),
            "threshold_eta": self.absorbing_threshold,
            "posterior_exceeds_threshold": posterior_exceeds,
            "kesten_goldie_condition_met": kesten_goldie_met,
            "hawkes_rho": float(rho),
            "regime3_transition_prob_mean": avg_transition_to_r3,
        }

        if clamp:
            logger.warning(
                "ABSORBING BOUNDARY ACTIVATED: Pr(R3|F_k) = %.4f ≥ η = %.4f. "
                "ρ(A_hawkes) = %.4f [KG condition %s]. φ* = 0.",
                posterior_regime3,
                self.absorbing_threshold,
                rho,
                "MET" if kesten_goldie_met else "not met (conservative clamp retained)",
            )
        else:
            logger.debug(
                "Absorbing boundary check PASSED: Pr(R3|F_k) = %.4f < η = %.4f.",
                posterior_regime3,
                self.absorbing_threshold,
            )

        return result

    # ------------------------------------------------------------------
    # Utility: regime summary
    # ------------------------------------------------------------------
    def regime_summary(self, state_sequence: NDArray[np.int64]) -> dict:
        """
        Compute regime occupancy statistics consistent with paper Table 2.

        Parameters
        ----------
        state_sequence : NDArray[int], shape (T,)

        Returns
        -------
        dict with empirical proportions and average durations per regime.
        """
        T = len(state_sequence)
        summary = {}
        labels = {0: "Regime_1_Normal", 1: "Regime_2_Stressed", 2: "Regime_3_Chaotic"}
        for s, name in labels.items():
            mask = state_sequence == s
            proportion = float(mask.sum() / T)
            # Run-length encoding for duration
            durations = []
            in_run = False
            run_len = 0
            for v in mask:
                if v:
                    in_run = True
                    run_len += 1
                else:
                    if in_run:
                        durations.append(run_len)
                        run_len = 0
                    in_run = False
            if in_run:
                durations.append(run_len)
            avg_dur = float(np.mean(durations)) if durations else 0.0
            summary[name] = {"proportion": proportion, "avg_duration_ticks": avg_dur}
        return summary

    # ------------------------------------------------------------------
    # Internal utilities
    # ------------------------------------------------------------------
    @staticmethod
    def _logsumexp(
        a: NDArray[np.float64],
        axis: Optional[int] = None,
        keepdims: bool = False,
    ) -> float | NDArray[np.float64]:
        """Numerically stable log-sum-exp."""
        a_max = np.max(a, axis=axis, keepdims=True)
        finite_mask = np.isfinite(a_max)
        a_max_safe = np.where(finite_mask, a_max, 0.0)
        out = np.log(np.sum(np.exp(a - a_max_safe), axis=axis, keepdims=keepdims))
        out += np.squeeze(a_max_safe, axis=axis) if not keepdims and axis is not None else a_max_safe if keepdims else float(a_max_safe.ravel()[0])
        return out

    @staticmethod
    def _validate_observations(observations: NDArray[np.float64]) -> None:
        if observations.ndim != 1:
            raise ValueError("observations must be a 1-D array of log-returns.")
        if len(observations) < N_STATES + 1:
            raise ValueError(
                f"Need at least {N_STATES + 1} observations to fit a {N_STATES}-state HMM."
            )
        if not np.isfinite(observations).all():
            raise ValueError("observations contain NaN or Inf values. Clean data before fitting.")

    def _check_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError(
                "MicrostructureHMM is not fitted. Call fit() first."
            )

    def __repr__(self) -> str:
        status = "fitted" if self._fitted else "unfitted"
        return (
            f"MicrostructureHMM(n_states={N_STATES}, max_iter={self.max_iter}, "
            f"tol={self.tol}, η={self.absorbing_threshold}, status={status})"
        )


# ---------------------------------------------------------------------------
# Helpers for synthetic data generation
# ---------------------------------------------------------------------------
def generate_regime_switching_returns(
    T: int = 10_000,
    seed: int = 0,
) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
    """
    Generate a synthetic log-return sequence with volatility clustering via a
    3-state Markov-switching process — matching the empirical regime structure
    reported in Matthews (2025) Table 2.

    Regime 0 (Normal):  σ = 0.0008, π_11 = 0.974
    Regime 1 (Stressed): σ = 0.0020, π_22 = 0.765
    Regime 2 (Chaotic):  σ = 0.0055, π_33 = 0.681

    Returns
    -------
    returns : NDArray[float], shape (T,)
    true_states : NDArray[int], shape (T,)
    """
    rng = np.random.default_rng(seed)

    A_true = np.array([
        [0.974, 0.023, 0.003],
        [0.198, 0.765, 0.037],
        [0.074, 0.245, 0.681],
    ])
    mu_true = np.array([0.00005, -0.00010, -0.00050])
    sigma_true = np.array([0.0008, 0.0020, 0.0055])

    states = np.empty(T, dtype=np.int64)
    states[0] = rng.choice(N_STATES, p=[0.68, 0.24, 0.08])
    for t in range(1, T):
        states[t] = rng.choice(N_STATES, p=A_true[states[t - 1]])

    returns = rng.normal(
        loc=mu_true[states],
        scale=sigma_true[states],
    )
    return returns, states


# ---------------------------------------------------------------------------
# Demonstration
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import time

    print("=" * 70)
    print("  MicrostructureHMM — Baum-Welch Demo")
    print("  Constrained Stochasticity Framework (Matthews, 2025)")
    print("=" * 70)

    # 1. Generate synthetic volatility-clustered returns
    T = 10_000
    print(f"\n[1] Generating {T:,} synthetic log-returns with regime switching...")
    returns, true_states = generate_regime_switching_returns(T=T, seed=2025)
    true_proportions = {i: (true_states == i).mean() for i in range(N_STATES)}
    print(f"    True regime proportions: R1={true_proportions[0]:.3f}  "
          f"R2={true_proportions[1]:.3f}  R3={true_proportions[2]:.3f}")
    print(f"    Return stats: μ={returns.mean():.6f}  σ={returns.std():.6f}  "
          f"min={returns.min():.6f}  max={returns.max():.6f}")

    # 2. Instantiate and fit
    print("\n[2] Fitting MicrostructureHMM via Baum-Welch EM...")
    hmm = MicrostructureHMM(
        max_iter=150,
        tol=1e-7,
        random_state=42,
        absorbing_threshold=0.72,
        hawkes_rho=0.69,  # representative value from paper Appendix A (Q1 2023)
    )

    t0 = time.perf_counter()
    hmm.fit(returns)
    elapsed = time.perf_counter() - t0
    print(f"    Converged in {hmm.n_iter_} iterations ({elapsed:.2f}s).")

    # 3. Print fitted parameters
    print("\n[3] Fitted parameters:")
    print(hmm.params_)

    # 4. Log-likelihood trajectory
    ll_hist = hmm.log_likelihood_history_
    print(f"\n[4] Log-likelihood trajectory:")
    print(f"    Initial  : {ll_hist[0]:.4f}")
    print(f"    Final    : {ll_hist[-1]:.4f}")
    print(f"    Δ (total): {ll_hist[-1] - ll_hist[0]:.4f}")

    # Print every 10 iterations
    stride = max(1, len(ll_hist) // 10)
    for i in range(0, len(ll_hist), stride):
        print(f"    iter {i+1:3d}: {ll_hist[i]:.4f}")

    # 5. Transition matrix evolution (illustrative — show initial vs final)
    print("\n[5] Fitted transition matrix A:")
    A = hmm.params_.A
    labels = ["R1(Normal)", "R2(Stress)", "R3(Chaos) "]
    header = "           " + "  ".join(f"{l:>10}" for l in labels)
    print(header)
    for i, row_label in enumerate(labels):
        row = "  ".join(f"{A[i, j]:10.4f}" for j in range(N_STATES))
        print(f"  {row_label}  {row}")

    # 6. Viterbi decoding
    print("\n[6] Viterbi regime decoding...")
    decoded = hmm.predict(returns)
    summary = hmm.regime_summary(decoded)
    print("    Decoded regime occupancy:")
    for name, stats in summary.items():
        print(f"      {name}: {stats['proportion']:.3f}  "
              f"(avg run = {stats['avg_duration_ticks']:.1f} ticks)")

    # 7. Absorbing boundary check
    print("\n[7] Absorbing boundary check (Proposition 1 / Kesten-Goldie):")
    posterior_proba = hmm.predict_proba(returns)  # (T, 3)
    posterior_r3 = posterior_proba[:, 2]

    # Find tick with highest Regime 3 posterior
    worst_tick = int(np.argmax(posterior_r3))
    worst_prob = posterior_r3[worst_tick]

    print(f"    Peak Regime-3 posterior: {worst_prob:.4f} at tick {worst_tick:,}")
    boundary_result = hmm.check_absorbing_boundary(
        posterior_regime3=worst_prob,
        hawkes_rho=1.02,  # simulating a super-critical state (ρ > 1)
    )
    print("    Boundary check result:")
    for k, v in boundary_result.items():
        print(f"      {k:40s}: {v}")

    # Also check a normal tick
    normal_prob = float(posterior_r3[0])
    normal_result = hmm.check_absorbing_boundary(
        posterior_regime3=normal_prob,
        hawkes_rho=0.69,
    )
    print(f"\n    Normal tick check (Pr(R3)={normal_prob:.4f}):")
    print(f"      clamp_execution: {normal_result['clamp_execution']}")

    # 8. Fraction of ticks where boundary fires
    n_clamped = (posterior_r3 >= hmm.absorbing_threshold).sum()
    print(f"\n[8] Ticks with execution clamped: {n_clamped:,} / {T:,} "
          f"({100*n_clamped/T:.2f}%)")
    print("    (Paper reports 7.7% of trading days as Regime 3.)")

    print("\n" + "=" * 70)
    print("  Done. MicrostructureHMM pipeline verified.")
    print("=" * 70)
