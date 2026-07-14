"""
Morse decomposition of the 1D dynamical system

    x' = F(x) = x - x^3

Equilibria: -1, 0, 1.  F'(x) = 1 - 3x^2, so F'(-1) = F'(1) = -2 < 0 (stable)
and F'(0) = 1 > 0 (unstable).

Global attractor:      A = [-1, 1]
Morse decomposition:   M = { M_-={-1}, M_0={0}, M_+={1} }
Morse graph:           M_0 --> M_-,  M_0 --> M_+
Lyapunov function:     V(x) = -x^2/2 + x^4/4   (dV/dt = -F(x)^2 <= 0)

The script produces morse_decomposition.png with four panels:
  (a) vector field F(x), equilibria, and the global attractor A = [-1,1]
  (b) the phase line with the flow direction
  (c) trajectories x(t) from several initial conditions
  (d) the Morse graph, drawn at the Lyapunov levels V(M_i)

Run:  python morse_decomposition.py
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# --------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------
def F(x):
    """Right-hand side of the ODE."""
    return x - x**3


def dF(x):
    """Derivative of F (the 1x1 Jacobian)."""
    return 1.0 - 3.0 * x**2


def V(x):
    """Lyapunov function: F = -V', hence dV/dt = V'(x) x' = -F(x)^2 <= 0."""
    return -0.5 * x**2 + 0.25 * x**4


EQUILIBRIA = np.array([-1.0, 0.0, 1.0])
LABELS = [r"$M_-=\{-1\}$", r"$M_0=\{0\}$", r"$M_+=\{1\}$"]
# stable (attracting) iff F'(x*) < 0
STABLE = dF(EQUILIBRIA) < 0

ATTRACTOR = (-1.0, 1.0)  # global attractor A = [-1, 1]

STABLE_COLOR = "#2a9d8f"
UNSTABLE_COLOR = "#e76f51"
ATTRACTOR_COLOR = "#bcd7e8"


def eq_color(i):
    return STABLE_COLOR if STABLE[i] else UNSTABLE_COLOR


# --------------------------------------------------------------------------
# Panel (a): the vector field F(x) and the global attractor
# --------------------------------------------------------------------------
def panel_vector_field(ax):
    x = np.linspace(-1.8, 1.8, 400)
    ax.axhline(0.0, color="k", lw=0.8)
    ax.axvspan(*ATTRACTOR, color=ATTRACTOR_COLOR, alpha=0.6,
               label=r"global attractor $\mathcal{A}=[-1,1]$")
    ax.plot(x, F(x), color="#264653", lw=2, label=r"$F(x)=x-x^3$")

    for i, xe in enumerate(EQUILIBRIA):
        ax.plot(xe, 0.0, "o", ms=10, mfc=eq_color(i), mec="k", zorder=5)
        ax.annotate("stable" if STABLE[i] else "unstable",
                    xy=(xe, 0.0), xytext=(0, -22), textcoords="offset points",
                    ha="center", fontsize=9, color=eq_color(i))

    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$\dot{x}=F(x)$")
    ax.set_title("(a) vector field and global attractor")
    ax.set_ylim(-1.1, 1.1)
    ax.legend(loc="upper center", fontsize=8)


# --------------------------------------------------------------------------
# Panel (b): the phase line (Morse sets + connecting orbits)
# --------------------------------------------------------------------------
def panel_phase_line(ax):
    ax.axhline(0.0, color="k", lw=1.2)
    ax.axvspan(*ATTRACTOR, ymin=0.35, ymax=0.65,
               color=ATTRACTOR_COLOR, alpha=0.6)

    # arrows show the sign of F: the connecting orbits inside the attractor
    for x0 in [-1.55, -0.65, -0.35, 0.35, 0.65, 1.55]:
        ax.annotate("", xy=(x0 + 0.22 * np.sign(F(x0)), 0.0), xytext=(x0, 0.0),
                    arrowprops=dict(arrowstyle="-|>", lw=1.6, color="#264653"))

    for i, xe in enumerate(EQUILIBRIA):
        ax.plot(xe, 0.0, "o", ms=12, mfc=eq_color(i), mec="k", zorder=5)
        ax.annotate(LABELS[i], xy=(xe, 0.0), xytext=(0, 16),
                    textcoords="offset points", ha="center", fontsize=10)

    ax.text(0.0, -0.55, r"Morse sets $\{-1\},\{0\},\{1\}$ + connecting orbits"
                        "\n" r"$=\;$ global attractor",
            ha="center", fontsize=9)
    ax.set_xlim(-1.85, 1.85)
    ax.set_ylim(-0.8, 0.8)
    ax.set_yticks([])
    ax.set_xlabel(r"$x$")
    ax.set_title("(b) phase line: invariant pieces and the flow between them")


# --------------------------------------------------------------------------
# Panel (c): trajectories -- every bounded orbit ends in a Morse set
# --------------------------------------------------------------------------
def panel_trajectories(ax):
    t_span = (0.0, 6.0)
    t_eval = np.linspace(*t_span, 400)
    for x0 in np.linspace(-1.6, 1.6, 17):
        sol = solve_ivp(lambda t, y: F(y), t_span, [x0], t_eval=t_eval,
                        rtol=1e-8, atol=1e-10)
        ax.plot(sol.t, sol.y[0], color="#4a7fa5", lw=1.0, alpha=0.8)

    # the unstable equilibrium itself is a complete orbit
    ax.plot(t_eval, np.zeros_like(t_eval), color=UNSTABLE_COLOR, lw=2.2,
            label=r"$M_0=\{0\}$ (unstable)")
    for xe in (-1.0, 1.0):
        ax.axhline(xe, color=STABLE_COLOR, lw=2.2, ls="--")
    ax.axhline(np.nan, color=STABLE_COLOR, lw=2.2, ls="--",
               label=r"$M_\pm=\{\pm 1\}$ (stable)")

    ax.set_xlabel(r"time $t$")
    ax.set_ylabel(r"$x(t)$")
    ax.set_title("(c) orbits are attracted to the Morse sets")
    ax.set_ylim(-1.8, 1.8)
    ax.legend(loc="center right", fontsize=8)


# --------------------------------------------------------------------------
# Panel (d): the Morse graph, drawn at the levels of the Lyapunov function
# --------------------------------------------------------------------------
def panel_morse_graph(ax):
    # vertical position = value of the Lyapunov function on each Morse set
    pos = {"M0": (0.0, V(0.0)), "Mm": (-1.0, V(-1.0)), "Mp": (1.0, V(1.0))}
    txt = {"M0": r"$M_0=\{0\}$" "\n" "unstable",
           "Mm": r"$M_-=\{-1\}$" "\n" "stable",
           "Mp": r"$M_+=\{1\}$" "\n" "stable"}
    col = {"M0": UNSTABLE_COLOR, "Mm": STABLE_COLOR, "Mp": STABLE_COLOR}

    # edges M_0 -> M_-  and  M_0 -> M_+ : the connecting complete orbits
    for target in ("Mm", "Mp"):
        x0, y0 = pos["M0"]
        x1, y1 = pos[target]
        ax.annotate("", xy=(0.78 * x1, y1 + 0.035), xytext=(0.55 * x1, y0 - 0.02),
                    arrowprops=dict(arrowstyle="-|>", lw=2.0, color="k"))

    for key, (x, y) in pos.items():
        ax.text(x, y, txt[key], ha="center", va="center", fontsize=9,
                bbox=dict(boxstyle="round,pad=0.45", fc=col[key], ec="k",
                          alpha=0.75))

    for y, lab in [(V(0.0), r"higher $V$"), (V(1.0), r"lower $V$")]:
        ax.axhline(y, color="#7aa6c2", ls=":", lw=1.2, zorder=0)
        ax.text(1.75, y, lab, va="center", fontsize=9, color="#4a7fa5")

    ax.set_xlim(-1.7, 2.3)
    ax.set_ylim(V(1.0) - 0.12, V(0.0) + 0.12)
    ax.set_xticks([])
    ax.set_ylabel(r"Lyapunov function $V(x)=-\frac{x^2}{2}+\frac{x^4}{4}$")
    ax.set_title(r"(d) Morse graph: $M_0\to M_-$, $M_0\to M_+$")


# --------------------------------------------------------------------------
def main():
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    panel_vector_field(axes[0, 0])
    panel_phase_line(axes[0, 1])
    panel_trajectories(axes[1, 0])
    panel_morse_graph(axes[1, 1])

    fig.suptitle(r"Morse decomposition of $\dot{x}=x-x^3$"
                 r"  ($\mathcal{A}=[-1,1]$,  $\mathcal{M}=\{M_-,M_0,M_+\}$)",
                 fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig("morse_decomposition.png", dpi=150)
    print("Equilibria and stability:")
    for i, xe in enumerate(EQUILIBRIA):
        kind = "stable" if STABLE[i] else "unstable"
        print(f"  x* = {xe:+.1f}   F'(x*) = {dF(xe):+.1f}   {kind:8s}"
              f"   V(x*) = {V(xe):+.3f}")
    print("Morse graph edges: M_0 -> M_-,  M_0 -> M_+")
    print("Saved morse_decomposition.png")


if __name__ == "__main__":
    main()