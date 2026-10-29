# Magnitude for Conversational Memory Selection

I tested whether metric-space magnitude could help choose useful memories when only a limited number of tokens fit in context.

See the [README](README.md) for the findings.

## Theory

For a nonempty finite set of memory embeddings $X$, with distance $d(i,j)$, define

$$
Z_{ij}(t)=e^{-t\,d(i,j)}, \qquad t>0.
$$

When $Z(t)$ is invertible, its magnitude is

$$
\mathrm{Mag}_t(X)=\mathbf{1}^{T}Z(t)^{-1}\mathbf{1}.
$$

The scale $t$ determines which distances matter. For distinct points with Euclidean distance, magnitude approaches one as $t\to0^+$ and the number of points as $t\to\infty$. Values across scales form a magnitude profile.

The experiments used regularized kernels for numerical stability. These produce a regularized magnitude score rather than the exact quantity above.

Two properties matter for memory selection:

- **Magnitude ignores order.** Reordering the same embedded memories leaves it unchanged.
- **Geometric difference does not establish usefulness.** From a singleton set, the unregularized gain from adding a second point at distance $d>0$ is

  $$
  \Delta\mathrm{Mag}_t=\frac{1-e^{-td}}{1+e^{-td}}=\tanh(td/2).
  $$

  It increases with distance. With equal costs and no relevance term, this step selects the farthest candidate.

Magnitude is computed entirely from pairwise distances. Its value here would therefore be as a useful summary of geometry, rather than a source of additional semantic information.
