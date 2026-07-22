"""Motivating example for the seminar talk: three forecasters predict tomorrow's
temperature as a full probability distribution.

Hilbert is honest but vague, Riemann is sharp but slightly off, and Bolzano is
confidently wrong. Scoring the realized observation under each forecast density
(the logarithmic score S(P, omega) = log p(omega)) motivates why we need
*proper* scoring rules.

Generates the build-up sequence ``forecast-distribution-intro{0,1,2}.png`` used
on the "Motivation" slides.
"""

import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

SCRIPT_DIR = Path(__file__).resolve().parent
FIGURES_DIR = SCRIPT_DIR / "figures"

# The cast of forecasters. The "true" distribution is what nature actually
# draws the temperature from; nobody observes it directly.
DISTRIBUTIONS = [
    {"label": "Hilbert", "mean": 21, "std": 4, "color": "green"},
    {"label": "True Distribution", "mean": 23, "std": 1, "color": "blue"},
    {"label": "Riemann", "mean": 24.5, "std": 0.5, "color": "orange"},
    {"label": "Bolzano", "mean": 4, "std": 0.25, "color": "red"},
]

# Slide build-up: each stage adds one forecaster to the picture.
# Stage 0/1: Hilbert, truth and Riemann. Stage 2: Bolzano joins.
STAGES = {
    0: ["Hilbert", "True Distribution", "Riemann"],
    1: ["Hilbert", "True Distribution", "Riemann"],
    2: ["Hilbert", "True Distribution", "Riemann", "Bolzano"],
}


def plot_stage(stage: int, labels: list[str], save: bool = True) -> None:
    """Plot the forecast densities for one build-up stage of the intro slides."""
    # Render labels with real LaTeX when available, mathtext otherwise.
    plt.rcParams["text.usetex"] = shutil.which("latex") is not None
    x = np.linspace(0, 40, 2000)

    plt.figure(figsize=(10, 6))
    for dist in DISTRIBUTIONS:
        if dist["label"] not in labels:
            continue
        y = norm.pdf(x, dist["mean"], dist["std"])
        plt.plot(x, y, label=dist["label"], color=dist["color"], linewidth=2.5)

        # Direct label above each peak, e.g. N(23, 1^2).
        annotation = rf"$\mathcal{{N}}({dist['mean']}, {dist['std']}^2)$"
        plt.text(
            x=dist["mean"] - 1,
            y=y.max() + 0.03,
            s=annotation,
            color=dist["color"],
            ha="center",
            fontweight="bold",
            fontsize=10,
        )

    plt.title("Forecast Distributions", fontsize=16)
    plt.xlabel(r"Temperature ($^\circ$C)", fontsize=12)
    plt.ylabel(r"Probability Density $p(x)$", fontsize=12)
    plt.legend(loc="upper right", fontsize=10)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.xlim(0, 36)
    plt.ylim(0, 1.8 if "Bolzano" in labels else 0.9)
    plt.gca().spines["top"].set_visible(False)
    plt.gca().spines["right"].set_visible(False)
    plt.tight_layout()

    if save:
        FIGURES_DIR.mkdir(exist_ok=True)
        out = FIGURES_DIR / f"forecast-distribution-intro{stage}.png"
        plt.savefig(out, dpi=300, bbox_inches="tight", transparent=True)
        print(f"Saved {out}")
    else:
        plt.show()
    plt.close()


def main(save: bool = True) -> None:
    """Generate all build-up stages of the intro figure."""
    for stage, labels in STAGES.items():
        plot_stage(stage, labels, save=save)


if __name__ == "__main__":
    main()
