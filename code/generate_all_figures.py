"""Generate every figure of the project in one non-interactive run.

Usage::

    python code/generate_all_figures.py

Calls the figure functions of all sibling scripts, prints each file as it is
written, and exits with a non-zero status if any figure fails.
"""

import sys
import traceback

import matplotlib

matplotlib.use("Agg")  # never require a display

import divergence_distributions
import forecast_distributions
import lognormal_median_vs_mean
import subdifferential
import visualize_theory
from visualize_theory import (
    figure_bregman,
    figure_crps,
    figure_honesty_experiment,
    figure_properness,
)

# (description, callable) — every entry writes one or more PNGs to figures/.
TASKS = [
    ("intro forecast distributions", forecast_distributions.main),
    ("log-normal median vs. mean", lognormal_median_vs_mean.main),
    ("divergence distributions", divergence_distributions.main),
    ("subdifferential at a kink", subdifferential.main),
    ("properness of Brier/log/zero-one", figure_properness),
    ("Bregman divergence geometry", figure_bregman),
    ("CRPS area and distance sensitivity", figure_crps),
    ("Monte Carlo honesty experiment", figure_honesty_experiment),
]


def main() -> int:
    print(f"Writing figures to {visualize_theory.FIGURES_DIR}\n")
    failures = []
    for description, task in TASKS:
        print(f"-> {description}")
        try:
            task()
        except Exception:  # keep going; report everything at the end
            failures.append(description)
            traceback.print_exc()
    if failures:
        print(f"\nFAILED: {len(failures)} of {len(TASKS)} figure tasks:",
              ", ".join(failures), file=sys.stderr)
        return 1
    print(f"\nAll {len(TASKS)} figure tasks completed successfully.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
