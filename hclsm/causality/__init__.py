"""Causal reasoning modules for HCLSM.

Sprint 4: Full causal discovery with DAG-constrained graph learning,
data-dependent adjacency, Gumbel-softmax edge sampling, do-calculus
counterfactual simulation, and augmented Lagrangian optimization.
"""

from hclsm.causality.causal_graph import CausalGraph
from hclsm.causality.intervention import CounterfactualSimulator, InterventionResult

__all__ = [
    "CausalGraph",
    "CounterfactualSimulator",
    "InterventionResult",
]
