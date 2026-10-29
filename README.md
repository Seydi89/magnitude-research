# Metric-Space Magnitude Research

Experiments testing whether metric space magnitude can help select useful, diverse conversational memories under a token budget. The idea was to use the geometry of the selected memories together, across different scales, to reduce redundancy while preserving relevance.

- Memory selection: Magnitude did not consistently outperform relevance ranking or MMR across the tested datasets and budgets.
- Conversation history and trajectory: Observing conversation history's magnitude change as well as magnitude trajectory did not provide any measurable information and provided no benefit to memory selection. 
- Limitations: The experiments used limited number of curated benchmarks and controlled setups. Evaluation focused mainly on evidence retrieval rather than final answer quality.
- Still open: More realistic memory stores with naturally occurring repetition, updates and conflicting information remain untested.

[Report and theory](REPORT.md)
