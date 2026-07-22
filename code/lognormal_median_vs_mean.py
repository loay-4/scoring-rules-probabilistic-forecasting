"""Counterexample: the "linear score" S(P, omega) = |omega - median(P)| is improper.

For a right-skewed log-normal truth Q the median (22) and mean (25) differ.
A forecaster judged by absolute distance to the median is rewarded for
reporting a distribution centered elsewhere instead of the truth — i.e.
S(P, Q) < S(Q, Q) for some P != Q, violating properness.

Generates ``lognormal_distribution.png`` for the "Improper Scoring Rule" slide.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import lognorm

SCRIPT_DIR = Path(__file__).resolve().parent
FIGURES_DIR = SCRIPT_DIR / "figures"

# Target moments of the true distribution Q.
TARGET_MEAN = 25
TARGET_MEDIAN = 22

# Recover the underlying normal parameters (mu, sigma) of the log-normal:
#   median = exp(mu)              =>  mu    = ln(median)
#   mean   = exp(mu + sigma^2/2)  =>  sigma = sqrt(2 ln(mean / median))
mu = np.log(TARGET_MEDIAN)
sigma = np.sqrt(2 * np.log(TARGET_MEAN / TARGET_MEDIAN))


def main(save: bool = True) -> None:
    x = np.linspace(0.1, 60, 2000)  # start above 0: log-normal support is (0, inf)
    y = lognorm.pdf(x, s=sigma, scale=np.exp(mu))

    plt.figure(figsize=(10, 6))
    plt.plot(x, y, label="Log-Normal Distribution", color="purple", linewidth=2.5)

    # Mark median vs mean — the gap between them is exactly what the improper
    # linear score lets a dishonest forecaster exploit.
    for value, color in [(TARGET_MEDIAN, "black"), (TARGET_MEAN, "red")]:
        height = lognorm.pdf(value, s=sigma, scale=np.exp(mu))
        plt.vlines(value, 0, height, colors=color, linestyles="--", alpha=0.5)

    # Show the two special values as bold ticks on the x-axis.
    plt.xticks([0, 10, 20, 22, 25, 30, 40, 50, 60])
    for label in plt.gca().get_xticklabels():
        if label.get_text() == str(TARGET_MEDIAN):
            label.set_fontweight("bold")
        elif label.get_text() == str(TARGET_MEAN):
            label.set_color("red")
            label.set_fontweight("bold")

    plt.title("Log-Normal Distribution", fontsize=16)
    plt.xlabel(r"Temperature ($^\circ$C)", fontsize=12)
    plt.ylabel(r"Probability Density $q(x)$", fontsize=12)
    plt.legend(loc="upper right", fontsize=10)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.xlim(0, 60)
    plt.ylim(0, 0.045)
    plt.gca().spines["top"].set_visible(False)
    plt.gca().spines["right"].set_visible(False)
    plt.tight_layout()

    if save:
        FIGURES_DIR.mkdir(exist_ok=True)
        out = FIGURES_DIR / "lognormal_distribution.png"
        plt.savefig(out, dpi=300, bbox_inches="tight", transparent=True)
        print(f"Saved {out}")
    else:
        plt.show()
    plt.close()


if __name__ == "__main__":
    main()
