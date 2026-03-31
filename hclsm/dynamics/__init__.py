"""Hierarchical dynamics engine for HCLSM.

Layer 3: Three-level temporal dynamics:
- Level 0 (SSM): Fast continuous physics at every timestep
- Level 1 (Sparse Transformer): Discrete event dynamics at event boundaries
- Level 2 (Compressed Transformer): Goal/plan dynamics over longer horizons
"""

from hclsm.dynamics.level0_ssm import Level0SSM
from hclsm.dynamics.level1_event import Level1EventDynamics
from hclsm.dynamics.level2_goal import Level2GoalDynamics
from hclsm.dynamics.event_detector import EventDetector
from hclsm.dynamics.hierarchy_manager import HierarchyManager

__all__ = [
    "Level0SSM",
    "Level1EventDynamics",
    "Level2GoalDynamics",
    "EventDetector",
    "HierarchyManager",
]
