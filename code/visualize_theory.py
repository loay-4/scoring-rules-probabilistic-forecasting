"""Visualizations of the theory behind proper scoring rules.

Companion code to the seminar talk "Scoring Rules for Probabilistic
Forecasting" (after Gneiting & Raftery, 2007). Each figure demonstrates one
of the central theoretical objects on a concrete example:

1. ``properness.png``
   Properness in action: for a binary event with true probability q, the
   *expected* Brier and logarithmic scores are uniquely maximized by
   reporting p = q (strict properness), while the zero-one score is flat on
   either side of 1/2 (proper, but not strictly).

2. ``bregman_divergence.png``
   The geometry behind d(P, Q) = S(Q, Q) - S(P, Q): the divergence of a
   proper score is the gap between the convex entropy G and its tangent —
   for the log score this is exactly the Kullback-Leibler divergence.

3. ``crps.png``
   The Continuous Ranked Probability Score as the squared area between the
   forecast CDF and the empirical step function of the observation, and its
   sensitivity to distance for sharp vs. vague forecasts.

4. ``honesty_experiment.png``
   Monte Carlo confirmation: forecasters who mis-state their uncertainty
   (over- or underconfident) are punished in expectation by both the log
   score and the CRPS — the honest forecaster wins.

All scores follow the talk's *reward* orientation: higher is better.

Run ``python visualize_theory.py`` to write all figures to ``figures/``.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

SCRIPT_DIR = Path(__file__).resolve().parent
FIGURES_DIR = SCRIPT_DIR / "figures"

# Validated colorblind-safe categorical palette (fixed assignment order).
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, MUTED = "#0b0b0b", "#52514e"

# These figures use matplotlib's mathtext (never external LaTeX), so they
# render identically on machines with and without a TeX installation.
plt.rcParams.update(
    {
        "text.usetex": False,
        "font.family": "serif",
        "mathtext.fontset": "cm",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.linestyle": "--",
        "grid.alpha": 0.35,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    }
)


# ----------------------------------------------------------------------------
# Figure 1: properness — honesty maximizes the expected score
# ----------------------------------------------------------------------------

def figure_properness(q: float = 0.7) -> None:
    """Expected score of reporting p when the true probability is q.

    Binary sample space, reward orientation. Brier and log score peak
    exactly at p = q; the zero-one score only cares which side of 1/2 the
    report falls on, so any p on the correct side is equally good.
    """
    _mathtext()
    p = np.linspace(0.001, 0.999, 1000)

    # S(p, Q) = E_Q[S(p, omega)] for each rule.
    brier = -(q * (1 - p) ** 2 + (1 - q) * p**2)
    log_score = q * np.log(p) + (1 - q) * np.log(1 - p)
    zero_one = np.where(p > 0.5, q, np.where(p < 0.5, 1 - q, 0.5))

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), sharex=True)
    curves = [
        ("Brier score", brier, BLUE, True),
        ("Logarithmic score", log_score, ORANGE, True),
        ("Zero-one score", zero_one, AQUA, False),
    ]
    for ax, (name, y, color, strictly_proper) in zip(axes, curves):
        ax.plot(p, y, color=color, linewidth=2)
        ax.axvline(q, color=MUTED, linestyle=":", linewidth=1)
        if strictly_proper:
            # Unique maximum at the truth: honesty is the only optimum.
            y_at_q = np.interp(q, p, y)
            ax.plot(q, y_at_q, "o", color=color, markersize=8)
            ax.annotate(
                "unique max at $p = q$",
                xy=(q, y_at_q),
                xytext=(0.15, 0.9),
                textcoords="axes fraction",
                fontsize=10,
                color=MUTED,
                arrowprops=dict(arrowstyle="->", color=MUTED, lw=1),
            )
        else:
            ax.annotate(
                "flat: proper, not strictly",
                xy=(0.75, q),
                xytext=(0.08, 0.55),
                textcoords="axes fraction",
                fontsize=10,
                color=MUTED,
                arrowprops=dict(arrowstyle="->", color=MUTED, lw=1),
            )
        ax.set_title(name, fontsize=12, color=INK)
        ax.set_xlabel("reported probability $p$")
    axes[1].set_ylim(-2.5, 0)  # log score diverges at the boundary
    axes[0].set_ylabel(r"expected score $S(p, Q)$")
    fig.suptitle(
        rf"Proper scoring rules reward honesty (true probability $q = {q}$)",
        fontsize=14,
    )
    fig.tight_layout()
    _save(fig, "properness.png")


# ----------------------------------------------------------------------------
# Figure 2: Bregman divergence — the gap between entropy and its tangent
# ----------------------------------------------------------------------------

def figure_bregman(p_forecast: float = 0.25, q_truth: float = 0.7) -> None:
    """Divergence of the log score as a Bregman divergence.

    On the binary simplex the entropy of the log score is the negative
    Shannon entropy G(p) = p log p + (1-p) log(1-p). The divergence
    d(p, q) = G(q) - [G(p) + G'(p)(q - p)] is the vertical gap between G
    and its tangent at the forecast p, evaluated at the truth q — and it
    equals KL(q || p).
    """
    _mathtext()
    G = lambda p: p * np.log(p) + (1 - p) * np.log(1 - p)
    dG = lambda p: np.log(p) - np.log(1 - p)  # derivative of G

    x = np.linspace(0.01, 0.99, 500)
    tangent = G(p_forecast) + dG(p_forecast) * (x - p_forecast)
    gap_lo = G(p_forecast) + dG(p_forecast) * (q_truth - p_forecast)
    gap_hi = G(q_truth)

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(x, G(x), color=BLUE, linewidth=2.5, label=r"entropy $G$")
    ax.plot(x, tangent, color=ORANGE, linewidth=1.8, linestyle="--",
            label=rf"tangent at forecast $p = {p_forecast}$")

    # The divergence: vertical gap between curve and tangent at the truth q.
    ax.plot([q_truth, q_truth], [gap_lo, gap_hi], color=AQUA, linewidth=3,
            solid_capstyle="butt")
    ax.plot(p_forecast, G(p_forecast), "o", color=ORANGE, markersize=8)
    ax.plot(q_truth, G(q_truth), "o", color=BLUE, markersize=8)

    kl = gap_hi - gap_lo
    ax.annotate(
        rf"$d(p, q) = \mathrm{{KL}}(q \,\|\, p) \approx {kl:.2f}$",
        xy=(q_truth, (gap_lo + gap_hi) / 2),
        xytext=(0.28, 0.14),
        textcoords="axes fraction",
        fontsize=12,
        color=INK,
        arrowprops=dict(arrowstyle="->", color=MUTED, lw=1),
    )
    ax.annotate(rf"truth $q = {q_truth}$", xy=(q_truth, G(q_truth)),
                xytext=(q_truth - 0.06, G(q_truth) + 0.06), fontsize=10,
                color=MUTED)

    ax.set_xlabel(r"probability of event, $p$")
    ax.set_ylabel(r"$G(p) = p\log p + (1-p)\log(1-p)$")
    ax.set_title("Bregman divergence: entropy minus its tangent", fontsize=14)
    ax.legend(loc="lower center", frameon=False)
    _save(fig, "bregman_divergence.png")


# ----------------------------------------------------------------------------
# Figure 3: CRPS — area between forecast CDF and observation step function
# ----------------------------------------------------------------------------

def crps_normal(mu: float, sigma: float, x: np.ndarray) -> np.ndarray:
    """Closed-form CRPS of a N(mu, sigma^2) forecast at observation x.

    Positive orientation (a loss); the talk's reward version is its negative.
    See Gneiting & Raftery (2007), eq. (21).
    """
    z = (x - mu) / sigma
    return sigma * (z * (2 * norm.cdf(z) - 1) + 2 * norm.pdf(z) - 1 / np.sqrt(np.pi))


def figure_crps(mu: float = 0.0, sigma: float = 1.0, obs: float = 1.5) -> None:
    """Left: CRPS as the shaded squared area between forecast CDF and the
    step function of the observation. Right: CRPS as a function of where the
    observation lands, for a sharp and a vague forecast — unlike the log
    score, the CRPS grows with the *distance* of the miss.
    """
    _mathtext()
    y = np.linspace(-5, 6, 1000)
    F = norm.cdf(y, mu, sigma)
    step = (y >= obs).astype(float)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # Left panel: the integrand of CRPS(F, x) = -∫ (F(y) - 1{y >= x})^2 dy.
    ax1.plot(y, F, color=BLUE, linewidth=2, label=r"forecast CDF $F$")
    ax1.plot(y, step, color=INK, linewidth=2,
             label=r"observation $\mathbb{1}\{y \geq x\}$")
    ax1.fill_between(y, F, step, color=AQUA, alpha=0.35, linewidth=0)
    crps_val = crps_normal(mu, sigma, np.array(obs))
    ax1.annotate(
        rf"$\mathrm{{CRPS}}(F, x) = -\int (F - \mathbb{{1}})^2\,dy"
        rf"\approx {-crps_val:.2f}$",
        xy=(0.98, 0.30), xycoords="axes fraction", ha="right", fontsize=11,
        color=INK,
    )
    ax1.axvline(obs, color=MUTED, linestyle=":", linewidth=1)
    ax1.text(obs + 0.15, 0.55, rf"$x = {obs}$", color=MUTED, fontsize=10)
    ax1.set_title("CRPS: squared area between CDFs", fontsize=13)
    ax1.set_xlabel("$y$")
    ax1.set_ylabel("cumulative probability")
    ax1.legend(loc="upper left", frameon=False)

    # Right panel: score as a function of the realized observation.
    obs_grid = np.linspace(-6, 6, 500)
    for s, color, label in [(1.0, BLUE, r"sharp: $\mathcal{N}(0, 1)$"),
                            (2.5, ORANGE, r"vague: $\mathcal{N}(0, 2.5^2)$")]:
        ax2.plot(obs_grid, -crps_normal(0.0, s, obs_grid), color=color,
                 linewidth=2, label=label)
    ax2.set_title("Reward by observation: sharpness pays off\nnear the center, "
                  "costs in the tails", fontsize=13)
    ax2.set_xlabel("observation $x$")
    ax2.set_ylabel(r"$\mathrm{CRPS}(F, x)$ (reward)")
    ax2.legend(loc="lower center", frameon=False)

    fig.tight_layout()
    _save(fig, "crps.png")


# ----------------------------------------------------------------------------
# Figure 4: Monte Carlo — the honest forecaster wins in expectation
# ----------------------------------------------------------------------------

def figure_honesty_experiment(n_draws: int = 50_000, seed: int = 42) -> None:
    """Empirical check of properness for continuous forecasts.

    Nature draws from N(0, 1). Forecasters all center correctly but report
    different sharpness sigma. Averaging the log score and the CRPS over many
    draws shows both are maximized at the honest sigma = 1: neither
    overconfidence (sigma < 1) nor underconfidence (sigma > 1) pays.
    """
    _mathtext()
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(n_draws)  # observations from the true N(0, 1)
    sigmas = np.linspace(0.4, 2.5, 60)

    mean_log = np.array([norm.logpdf(x, 0.0, s).mean() for s in sigmas])
    mean_crps = np.array([-crps_normal(0.0, s, x).mean() for s in sigmas])

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), sharex=True)
    for ax, y, color, name in [
        (axes[0], mean_log, BLUE, "mean log score"),
        (axes[1], mean_crps, ORANGE, "mean CRPS"),
    ]:
        ax.plot(sigmas, y, color=color, linewidth=2)
        best = sigmas[np.argmax(y)]
        ax.axvline(1.0, color=MUTED, linestyle=":", linewidth=1)
        ax.plot(best, y.max(), "o", color=color, markersize=8)
        ax.annotate(
            rf"empirical optimum $\sigma \approx {best:.2f}$",
            xy=(best, y.max()),
            xytext=(0.35, 0.2),
            textcoords="axes fraction",
            fontsize=10,
            color=MUTED,
            arrowprops=dict(arrowstyle="->", color=MUTED, lw=1),
        )
        ax.set_title(name, fontsize=13)
        ax.set_xlabel(r"reported sharpness $\sigma$")
        ax.set_ylabel("average score (higher = better)")
    fig.suptitle(
        rf"Truth is $\mathcal{{N}}(0,1)$: proper scores are maximized by "
        rf"honest uncertainty  ({n_draws:,} draws)",
        fontsize=13,
    )
    fig.tight_layout()
    _save(fig, "honesty_experiment.png")


def _mathtext() -> None:
    """Force mathtext rendering (other scripts may have enabled usetex)."""
    plt.rcParams["text.usetex"] = False


def _save(fig: plt.Figure, name: str) -> None:
    FIGURES_DIR.mkdir(exist_ok=True)
    out = FIGURES_DIR / name
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


if __name__ == "__main__":
    figure_properness()
    figure_bregman()
    figure_crps()
    figure_honesty_experiment()
