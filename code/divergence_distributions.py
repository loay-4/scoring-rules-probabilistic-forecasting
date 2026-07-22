"""Three well-separated normal densities used on the "Bregman Divergence" slide.

The figure illustrates that a divergence d(P, Q) = S(Q, Q) - S(P, Q) measures
how far a forecast P is from the truth Q — here shown with two forecasts on
opposite sides of the true distribution.

Generates ``divergences.png``.
"""

import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

SCRIPT_DIR = Path(__file__).resolve().parent
FIGURES_DIR = SCRIPT_DIR / "figures"

DISTRIBUTIONS = [
    {"label": "Hilbert", "mean": 5, "std": 1, "color": "green"},
    {"label": "True Distribution", "mean": 15, "std": 1, "color": "blue"},
    {"label": "Riemann", "mean": 27, "std": 1, "color": "orange"},
]


def main(save: bool = True) -> None:
    # Render labels with real LaTeX when available, mathtext otherwise.
    plt.rcParams["text.usetex"] = shutil.which("latex") is not None
    x = np.linspace(0, 40, 2000)

    plt.figure(figsize=(10, 6))
    for dist in DISTRIBUTIONS:
        y = norm.pdf(x, dist["mean"], dist["std"])
        plt.plot(x, y, label=dist["label"], color=dist["color"], linewidth=2.5)

    plt.grid(True, linestyle="--", alpha=0.5)
    plt.xlim(0, 36)
    plt.ylim(0, 0.5)
    plt.gca().spines["top"].set_visible(False)
    plt.gca().spines["right"].set_visible(False)
    plt.tight_layout()

    if save:
        FIGURES_DIR.mkdir(exist_ok=True)
        out = FIGURES_DIR / "divergences.png"
        plt.savefig(out, dpi=300, bbox_inches="tight", transparent=True)
        print(f"Saved {out}")
    else:
        plt.show()
    plt.close()


if __name__ == "__main__":
    main()
