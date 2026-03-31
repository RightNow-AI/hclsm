"""Sleep-phase consolidation — re-export from semantic.py.

This module provides backward compatibility. The actual implementation
is in semantic.py (ConsolidationLoop class).
"""

from __future__ import annotations

from hclsm.memory.semantic import ConsolidationLoop

__all__ = ["ConsolidationLoop"]
