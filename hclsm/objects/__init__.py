"""Object decomposition layer for HCLSM.

Layer 2: Decompose scene representations into variable-count object slots
with temporal tracking and relational reasoning.
"""

from hclsm.objects.object_state import ObjectState
from hclsm.objects.dynamic_slots import DynamicSlotAttention
from hclsm.objects.slot_tracker import SlotTracker
from hclsm.objects.relation_graph import RelationGraph

__all__ = [
    "ObjectState",
    "DynamicSlotAttention",
    "SlotTracker",
    "RelationGraph",
]
