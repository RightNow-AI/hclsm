"""Custom CUDA/Triton kernels for HCLSM.

Sprint 3: Fused kernels with auto-dispatch. Each kernel has:
- Triton implementation (fastest, requires triton + CUDA GPU)
- torch.compile / optimized PyTorch (middle ground)
- Naive PyTorch reference (always works, used for correctness testing)
"""

from hclsm.kernels.slot_attention_kernel import (
    slot_attention_fn,
    naive_slot_attention,
    compiled_slot_attention,
    TRITON_AVAILABLE,
)
from hclsm.kernels.ssm_scan_kernel import (
    ssm_scan_fn,
    sequential_ssm_scan,
    parallel_ssm_scan_pytorch,
)
from hclsm.kernels.sparse_event_attn import (
    sparse_event_attn_fn,
    naive_sparse_event_attention,
    sparse_event_attention,
)
from hclsm.kernels.gnn_message_pass import (
    gnn_message_pass_fn,
    naive_gnn_message_pass,
    optimized_gnn_message_pass,
    chunked_gnn_message_pass,
)
from hclsm.kernels.hierarchical_state import (
    hierarchical_state_fn,
    hierarchical_state_naive,
    hierarchical_state_fused,
)

__all__ = [
    # Slot attention
    "slot_attention_fn",
    "naive_slot_attention",
    "compiled_slot_attention",
    # SSM scan
    "ssm_scan_fn",
    "sequential_ssm_scan",
    "parallel_ssm_scan_pytorch",
    # Sparse event attention
    "sparse_event_attn_fn",
    "naive_sparse_event_attention",
    "sparse_event_attention",
    # GNN message passing
    "gnn_message_pass_fn",
    "naive_gnn_message_pass",
    "optimized_gnn_message_pass",
    "chunked_gnn_message_pass",
    # Hierarchical state
    "hierarchical_state_fn",
    "hierarchical_state_naive",
    "hierarchical_state_fused",
    # Meta
    "TRITON_AVAILABLE",
]
