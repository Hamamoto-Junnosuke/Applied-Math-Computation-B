"""
Experiment 1
------------
Keep the feature set fixed and change the prediction target Y.

Y_1 = median house value of the district
Y_2 = latitude of the district

For each single feature we compute I_Y({feature}) and rank the features.
If information were a property of the data alone, the ranking would not
depend on Y.
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from info import load_data, normalized_information

HERE = os.path.dirname(os.path.abspath(__file__))
FIGDIR = os.path.join(HERE, os.pardir, "figures")
OUTPUT = os.path.join(FIGDIR, "fig1_two_targets.png")


def rank_features(target, table, pool):
    scores = [(name, normalized_information([name], target, table)) for name in pool]
    return sorted(scores, key=lambda item: item[1])


def main():
    features, price = load_data()
    latitude = features["Latitude"].copy()

    # Latitude is excluded from the second pool: it is the target there.
    pool_price = list(features.columns)
    pool_latitude = [c for c in features.columns if c != "Latitude"]

    ranked_price = rank_features(price, features, pool_price)
    ranked_latitude = rank_features(latitude, features, pool_latitude)

    for title, ranking in [("Y = house price", ranked_price),
                           ("Y = latitude", ranked_latitude)]:
        print(f"\n{title}")
        for name, value in reversed(ranking):
            print(f"  {name:<12} {value:7.4f}")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    panels = [
        (axes[0], ranked_price, "Y = house price", "#3f6fb5"),
        (axes[1], ranked_latitude, "Y = latitude", "#c26a3d"),
    ]
    for ax, ranking, title, colour in panels:
        names = [n for n, _ in ranking]
        values = [v for _, v in ranking]
        bars = ax.barh(names, values, color=colour)
        # Highlight the two features whose ranking reverses.
        for bar, name in zip(bars, names):
            if name in ("MedInc", "Longitude"):
                bar.set_edgecolor("black")
                bar.set_linewidth(1.6)
        ax.set_xlim(0, 1)
        ax.set_xlabel("normalised information  $I_Y(S)\\,/\\,L(f_\\emptyset)$")
        ax.set_title(title)
        ax.grid(axis="x", alpha=0.3)

    fig.suptitle(
        "The same features carry different information "
        "depending on what is predicted",
        fontsize=12,
    )
    fig.tight_layout()
    os.makedirs(FIGDIR, exist_ok=True)
    fig.savefig(OUTPUT, dpi=150)
    print(f"\nsaved: {os.path.normpath(OUTPUT)}")


if __name__ == "__main__":
    main()
