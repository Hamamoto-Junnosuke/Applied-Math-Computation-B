"""
Experiment 2
------------
Fix the target (house price) and vary the feature set.

Latitude alone and longitude alone are both nearly useless, yet the two
together are far more informative than the sum of their individual values:

    I_Y({lat, lon})  >>  I_Y({lat}) + I_Y({lon})

Information therefore cannot be distributed over the individual features.
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from info import load_data, information

HERE = os.path.dirname(os.path.abspath(__file__))
FIGDIR = os.path.join(HERE, os.pardir, "figures")
OUTPUT = os.path.join(FIGDIR, "fig2_combination.png")


def main():
    features, price = load_data()

    sets = [
        ("Latitude\nonly", ["Latitude"]),
        ("Longitude\nonly", ["Longitude"]),
        ("sum of\nthe two", None),
        ("Latitude +\nLongitude", ["Latitude", "Longitude"]),
    ]

    values = []
    for label, subset in sets:
        if subset is None:
            values.append(values[0] + values[1])
            continue
        value, _ = information(subset, price, features)
        values.append(value)

    for (label, _), value in zip(sets, values):
        print(f"  {label.replace(chr(10), ' '):<22} I = {value:7.4f}")
    print(f"\n  ratio (joint / sum) = {values[3] / values[2]:.1f}")

    labels = [label for label, _ in sets]
    colours = ["#8fa9cf", "#8fa9cf", "#b8b8b8", "#3f6fb5"]

    fig, ax = plt.subplots(figsize=(7, 4.4))
    bars = ax.bar(labels, values, color=colours)
    bars[2].set_hatch("//")
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.008,
                f"{value:.3f}", ha="center", fontsize=10)

    ax.axhline(values[2], color="grey", linestyle="--", linewidth=1)
    ax.set_ylabel("information  $I_Y(S)$   (Y = house price)")
    ax.set_ylim(0, max(values) * 1.18)
    ax.set_title("Two features that are useless alone but strong together")
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    os.makedirs(FIGDIR, exist_ok=True)
    fig.savefig(OUTPUT, dpi=150)
    print(f"saved: {os.path.normpath(OUTPUT)}")


if __name__ == "__main__":
    main()
