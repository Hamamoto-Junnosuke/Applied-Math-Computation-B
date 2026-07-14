# Morse Decomposition of $\dot{x} = x - x^3$

Report assignment, Lecture 5 (Introduction to Applied Mathematics and Computation B, 2026 Q2).
Topic 1: *a Morse decomposition of a one- or two-dimensional dynamical system*.

## Files

| file | content |
|---|---|
| `morse_decomposition.py` | generates `morse_decomposition.png` |
| `morse_decomposition.png` | the figure (produced by running the script) |

Requirements: `numpy`, `scipy`, `matplotlib`.

```bash
python morse_decomposition.py
```

## The system

We take the scalar ODE from the lecture,

$$\dot{x} = F(x) = x - x^3 .$$

Its equilibria are $-1, 0, 1$. Since $F'(x) = 1 - 3x^2$ we get $F'(\pm 1) = -2 < 0$ and
$F'(0) = +1 > 0$, so $\pm 1$ are stable and $0$ is unstable. The global attractor is

$$\mathcal{A} = [-1, 1],$$

and it is *not* just a list of equilibria: it also contains the connecting orbits from $0$ to $-1$
and from $0$ to $+1$.

## How the figure shows the Morse decomposition

The Morse decomposition used here is

$$\mathcal{M} = \{\, M_- = \{-1\},\; M_0 = \{0\},\; M_+ = \{1\} \,\},$$

with Morse graph $M_0 \to M_-$ and $M_0 \to M_+$. Each panel shows one part of the definition.

**(a) Vector field and global attractor.**
The curve $F(x)$ crosses zero at the three equilibria, and the shaded band is $\mathcal{A} = [-1,1]$.
Outside the band the arrows point inward, which is the statement
$\operatorname{dist}_H(S(t)B, \mathcal{A}) \to 0$ for every bounded $B$.

**(b) Phase line.**
The three Morse sets are pairwise disjoint, compact and invariant, and each is isolated (a small
neighbourhood of it contains no other invariant set). The arrows between them are the connecting
complete orbits. Every complete orbit inside $\mathcal{A}$ that is not contained in a single Morse
set runs from $M_0$ backwards in time to a stable equilibrium forwards in time — exactly condition
(iii) of Definition 3.2.

**(c) Trajectories.**
Numerical solutions from many initial conditions all converge to $\pm 1$; only the orbit starting
exactly at $0$ stays at $0$. This is the stable/unstable set picture: $W^s(M_0) = \{0\}$ is thin,
while $W^u(M_0) = (-1,1)$ is the whole interior of the attractor. The connections are read off as

$$M_0 \to M_\pm \iff W^u(M_0) \cap W^s(M_\pm) \neq \varnothing .$$

**(d) Morse graph with a Lyapunov function.**
The system is gradient-like: with $V(x) = -\tfrac{x^2}{2} + \tfrac{x^4}{4}$ we have $F = -V'$, so

$$\frac{d}{dt} V(x(t)) = V'(x)\,\dot{x} = -F(x)^2 \le 0,$$

with equality only at the equilibria. Hence $V$ is non-increasing along orbits, constant on each
Morse set, and strictly decreasing elsewhere — a Lyapunov function in the sense of Definition 3.4.
The nodes are drawn at their levels $V(0) = 0$ and $V(\pm 1) = -1/4$, so the arrows literally point
downhill. This is the "recurrent pieces plus downhill motion between them" reading of the dynamics,
and it realises the partial order $M_0 \succ M_\pm$.

## Relation to integrated information

The Morse graph is the continuous-time replacement for a state-transition graph. Its nodes $M_i$ are
the recurrent possibilities of the system and its arrows say which possibility can lead to which
other, so it plays the role that a TPM plays in discrete IIT.

Reading the graph in the two time directions gives the two repertoires of IIT:

* **Effect (future).** From $M_0$ the flow can reach either $M_-$ or $M_+$, so the effect repertoire
  of $M_0$ is spread over two nodes. From $M_\pm$ nothing can be left, so their effect repertoire is
  concentrated on themselves.
* **Cause (past).** Backwards in time, a state at $M_+$ may have come from $M_0$ or from $M_+$
  itself, so the cause repertoire is where the branching structure appears for the stable nodes.

The asymmetry between (c) — every orbit ends at $\pm 1$ — and the backward picture is exactly why the
lecture insists on *complete* orbits (Definition 2.2): the semiflow is forward-unique but not
backward-unique, and the past is where the informational content of a state lives.

Weights could be attached to the edges using the eigenvalues of the Jacobian, $\mathrm{Att}(\pm 1) =
e^{2}$ and $\mathrm{Rep}(0) = e^{1}$, turning this Morse graph into an informational structure
$\mathcal{I} = (\mathcal{M}, E, w)$ and then into a TPM. This 1D example has integrated information
$\phi = 0$ in a trivial sense — with a single node there is no bipartition of the mechanism to cut —
so it illustrates the *geometry* side of the lecture (Sections 2–3) rather than the integration side.
A Lotka–Volterra mechanism with $N \ge 2$ nodes would be the next step.

## References

Lecture 5, "Morse Decomposition and Integrated Information", Motoya Ohnishi, 2026 Q2
(Definitions 2.5, 3.1, 3.2, 3.4, Examples 2.1 and 3.1).