# Research decisions and extension path

## What this pilot can establish

1. Whether global, task-conditional, or class-conditional magnitude changes during sequential fine-tuning.
2. Whether such drift leads old-task degradation rather than merely correlating contemporaneously.
3. Whether the signal survives comparison with displacement, pairwise-distance moments, effective rank and classifier alignment.
4. Which layer and which scale range carries any predictive signal.

It cannot establish usefulness from one synthetic seed. Treat the included run as a pipeline validation. A research result requires repeated seeds, at least one standard continual-learning benchmark, frozen analysis choices, uncertainty intervals, and correction for trying many scales/layers.

## Next experiments, in order

1. Run 20 synthetic seeds and estimate out-of-seed predictive performance.
2. Add Split-MNIST and Split-CIFAR-10 with fixed reference sets.
3. Fit simple predictors of future forgetting using training seeds only: standard baselines; magnitude only; baselines plus magnitude.
4. Freeze layer/scale/features, then evaluate held-out seeds and benchmark.
5. Only if magnitude adds predictive value, test a replay controller triggered by the frozen signal.

## Avoiding leakage and false discovery

- The scale grid is normalized from each group's median distance and the absolute scales are stored.
- Select scales/layers on validation seeds, never on test seeds.
- Compare predictive performance out of seed, not checkpoint-level random splits.
- Report a no-magnitude baseline with accuracy/loss, prediction entropy, displacement, effective rank, centroid separation and pairwise moments.
- Analyse profile-valued signals with preregistered summaries or regularized models; do not select the best-looking scale after seeing test outcomes.

