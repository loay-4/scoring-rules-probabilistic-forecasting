"""Subdifferential of a convex function at a kink.

The piecewise function G(x) = x^2/2 for x < 0 and G(x) = x for x >= 0 is convex
but not differentiable at 0. Every line through (0, 0) with slope v in [0, 1]
stays below the graph, so the subdifferential is the whole interval
partial G(0) = [0, 1] — the picture behind the McCarthy/Savage
characterization of proper scoring rules via subgradients of the entropy.

Generates ``subgradient_visualization.png``.
"""

import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["text.usetex"] = shutil.which("latex") is not None

FIGURES_DIR = Path(__file__).parent / "figures"


def G(x: np.ndarray) -> np.ndarray:
    """Convex function with a kink at 0: quadratic left branch, linear right."""
    return np.where(x < 0, 0.5 * x**2, x)


def main(save: bool = True) -> None:
    x = np.linspace(-3, 3, 400)

    # The kink where the subdifferential is a whole interval rather than a point.
    p_x, p_y = 0.0, G(np.array(0.0))

    plt.figure(figsize=(10, 6))
    ax = plt.gca()

    # A few supporting lines G(p) + v (x - p) with subgradients v in [0, 1].
    for v, color in zip([0.2, 0.5, 0.8], ["orange", "red", "purple"]):
        plt.plot(x, p_y + v * (x - p_x), linestyle="--", color=color,
                 alpha=0.6, linewidth=1.5)

    plt.plot(x, G(x), color="green", linewidth=3)
    plt.plot(p_x, p_y, "o", color="black", markersize=8, zorder=5)

    # Center the axes on the origin so the supporting lines are easy to read.
    ax.spines["left"].set_position("zero")
    ax.spines["bottom"].set_position("zero")
    ax.spines["right"].set_color("none")
    ax.spines["top"].set_color("none")
    ax.xaxis.set_ticks_position("bottom")
    ax.yaxis.set_ticks_position("left")

    plt.title(r"Visualization of Subdifferential $\partial G(0) = [0, 1]$",
              fontsize=16, pad=20)
    ax.set_xlabel(r"$x$", loc="right", fontsize=14)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.xlim(-3, 3.01)
    plt.ylim(-1.01, 3)
    plt.tight_layout()

    if save:
        FIGURES_DIR.mkdir(exist_ok=True)
        out = FIGURES_DIR / "subgradient_visualization.png"
        plt.savefig(out, dpi=300, bbox_inches="tight", transparent=True)
        print(f"Saved {out}")
    else:
        plt.show()
    plt.close()


if __name__ == "__main__":
    main()
