"""HCLSM: Hierarchical Causal Latent State Machine.

A novel world model architecture combining latent prediction (JEPA),
object-centric decomposition, hierarchical temporal dynamics, causal reasoning,
and continual learning.
"""

__version__ = "0.1.0"

from hclsm.config import HCLSMConfig

__all__ = ["HCLSMConfig", "__version__"]
