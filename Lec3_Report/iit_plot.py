#!/usr/bin/env python3
"""
IIT Example: Phi for a 2-node COPY system.

System: Node A is random; Node B copies Node A with probability p.
  - p = 0.5 : B is random (no integration) -> Phi = 0
  - p = 1.0 : B perfectly copies A (maximum integration) -> Phi = 1 bit

This example is NOT shown in the lecture note.
"""

import numpy as np
import matplotlib.pyplot as plt


def compute_phi(p):
    """
    Compute Phi for the 2-node COPY system.

    States: (A, B) in {(0,0), (0,1), (1,0), (1,1)} = {0, 1, 2, 3}
    Transition: A_{t+1} is always random (0.5/0.5).
                B_{t+1} copies A_t with probability p.
    Prior: uniform over 4 states.
    """
    states = [(0, 0), (0, 1), (1, 0), (1, 1)]

    # Build TPM (4 x 4)
    tpm = np.zeros((4, 4))
    for i, (a, b) in enumerate(states):
        for j, (a1, b1) in enumerate(states):
            p_a1 = 0.5
            p_b1 = p if b1 == a else (1 - p)
            tpm[i, j] = p_a1 * p_b1

    # Joint distribution P(s_t, s_{t+1})
    prior = np.ones(4) / 4
    joint = prior[:, None] * tpm  # shape (4, 4)

    def mi(j):
        px, py = j.sum(axis=1), j.sum(axis=0)
        val = 0.0
        for x in range(len(px)):
            for y in range(len(py)):
                if j[x, y] > 0 and px[x] > 0 and py[y] > 0:
                    val += j[x, y] * np.log2(j[x, y] / (px[x] * py[y]))
        return val

    mi_whole = mi(joint)

    # Partition: marginal for A alone and B alone
    jA = np.zeros((2, 2))
    jB = np.zeros((2, 2))
    for i, (a, b) in enumerate(states):
        for j2, (a1, b1) in enumerate(states):
            jA[a, a1] += joint[i, j2]
            jB[b, b1] += joint[i, j2]

    phi = max(0.0, mi_whole - mi(jA) - mi(jB))
    return phi


p_values = np.linspace(0.5, 1.0, 100)
phi_values = [compute_phi(p) for p in p_values]

plt.figure(figsize=(7, 4))
plt.plot(p_values, phi_values, color="steelblue", linewidth=2.5)
plt.fill_between(p_values, phi_values, alpha=0.15, color="steelblue")
plt.xlabel("Copy probability  p", fontsize=12)
plt.ylabel("Φ  (bits)", fontsize=12)
plt.title("IIT: Φ for a 2-node COPY system\n"
          "B copies A with probability p;  A is always random", fontsize=11)
plt.annotate("p=0.5\n(no integration)\nΦ=0",
             xy=(0.5, 0), xytext=(0.55, 0.2),
             arrowprops=dict(arrowstyle="->"), fontsize=9)
plt.annotate("p=1.0\n(perfect copy)\nΦ=1 bit",
             xy=(1.0, 1.0), xytext=(0.82, 0.75),
             arrowprops=dict(arrowstyle="->"), fontsize=9)
plt.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig("iit_copy_phi.png", dpi=150, bbox_inches="tight")
plt.show()
print("Plot saved to iit_copy_phi.png")
