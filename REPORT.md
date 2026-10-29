# Magnitude for Conversational Memory Selection

I tested whether metric-space magnitude could help choose useful memories when only a limited number of tokens fit in context.

See the [README](README.md) for the findings.

## Theory

For a nonempty finite set of memory embeddings `X`, with distance `d(i,j)`, define

```text
Z_ij(t) = exp(-t * d(i,j)),    t > 0.
```

When `Z(t)` is invertible, its magnitude is

```text
Mag_t(X) = 1^T * Z(t)^(-1) * 1.
```

Here, `1` is the column vector of all ones, `1^T` is its transpose, and `Z(t)^(-1)` is the inverse matrix.

The scale `t` determines which distances matter. For distinct points with Euclidean distance, magnitude approaches one as `t` approaches zero from above, and the number of points as `t` approaches infinity. Values across scales form a magnitude profile.

The experiments used regularized kernels for numerical stability. These produce a regularized magnitude score rather than the exact quantity above.

Two properties matter for memory selection:

- **Magnitude ignores order.** Reordering the same embedded memories leaves it unchanged.
- **Geometric difference does not establish usefulness.** From a singleton set, the unregularized gain from adding a second point at distance `d > 0` is

  ```text
  Delta Mag_t = (1 - exp(-t*d)) / (1 + exp(-t*d))
              = tanh(t*d / 2).
  ```

  It increases with distance. With equal costs and no relevance term, this step selects the farthest candidate.

Magnitude is computed entirely from pairwise distances. Its value here would therefore be as a useful summary of geometry, rather than a source of additional semantic information.
