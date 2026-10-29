# Magnitude for Conversational Memory Selection

I tested whether metric-space magnitude could help choose useful memories when only a limited number of tokens fit in context.

See the [README](https://github.com/Seydi89/magnitude-research/blob/main/README.md) for the findings.

## Theory

For a nonempty finite set of memory embeddings $X$, with distance $d(i,j)$, define

$$
Z_{ij}(t) = e^{-t d(i,j)}, \qquad t > 0.
$$

When $Z(t)$ is invertible, its magnitude is

$$
\mathrm{Mag}_t(X) = \mathbf{1}^{T} Z(t)^{-1} \mathbf{1}.
$$

Here, $\mathbf{1}$ is the column vector of all ones, $\mathbf{1}^{T}$ is its transpose, and $Z(t)^{-1}$ is the inverse matrix.

The scale $t$ determines which distances matter. For distinct points with Euclidean distance, magnitude approaches one as $t$ approaches zero from above, and the number of points as $t$ approaches infinity. Values across scales form a magnitude profile.

The experiments used regularized kernels for numerical stability. These produce a regularized magnitude score rather than the exact quantity above.

Two properties matter for memory selection:

- **Magnitude ignores order.** Reordering the same embedded memories leaves it unchanged.
- **Geometric difference does not establish usefulness.** Increasing magnitude does not necessarily improve the information available to answer a question.

To illustrate the second property, consider a singleton set. The unregularized gain from adding a second point at distance $d > 0$ is

$$
\Delta\mathrm{Mag}_t = \frac{1 - e^{-td}}{1 + e^{-td}} = \tanh\left(\frac{td}{2}\right).
$$

It increases with distance. With equal costs and no relevance term, this step selects the farthest candidate.

Magnitude is computed entirely from pairwise distances. Its value here would therefore be as a useful summary of geometry, rather than a source of additional semantic information.
