"""
Fusion Engine — discovers and merges fusable operations.

The core insight: most GPU kernels are memory-bound, not compute-bound.
Fusing operations eliminates intermediate memory reads/writes.
A chain of RMSNorm → RoPE → Attention saves 2 global memory round-trips.

Fusion rules:
1. Elementwise → Elementwise: always fusable
2. Elementwise → Reduction: fusable (element feeds reduction)
3. Reduction → Elementwise: fusable if reduction output is broadcast
4. Reduction → Reduction: fusable only if same reduction dimension
5. MatMul → Elementwise: epilogue fusion (bias add, activation)
6. MatMul → MatMul: not fusable (different tiling requirements)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from molten.ir import DataflowGraph, Op, OpType


@dataclass
class FusionGroup:
    """A group of operations that will become a single kernel."""
    group_id: int
    op_ids: list[int] = field(default_factory=list)
    has_matmul: bool = False
    has_reduction: bool = False
    reduction_dim: Optional[int] = None

    @property
    def size(self) -> int:
        return len(self.op_ids)

    def can_add(self, op: Op, graph: DataflowGraph) -> bool:
        """Check if an op can be added to this fusion group."""
        is_elem = graph.is_elementwise(op.id)
        is_red = graph.is_reduction(op.id)
        is_matmul = op.op_type == OpType.MATMUL

        # Rule 6: No two matmuls in one group
        if is_matmul and self.has_matmul:
            return False

        # Rule 5: MatMul + elementwise epilogue is OK
        if is_matmul and not self.has_matmul:
            return True

        # Rule 1: Elementwise chains always fuse
        if is_elem and not self.has_reduction:
            return True

        # Rule 2: Elementwise feeding a reduction
        if is_elem and self.has_reduction:
            return True

        # Rule 3: Reduction → Elementwise
        if is_red and not self.has_reduction:
            return True

        # Rule 4: Reduction → Reduction only if same dim
        if is_red and self.has_reduction:
            return (op.reduction_dim is not None and
                    op.reduction_dim == self.reduction_dim)

        return True  # default: allow


class FusionEngine:
    """
    Analyzes a DataflowGraph and partitions it into FusionGroups.

    Each FusionGroup becomes a single GPU kernel. The engine maximizes
    fusion (minimize memory traffic) while respecting fusion constraints.
    """

    def __init__(self):
        self._group_counter = 0

    def fuse(self, graph: DataflowGraph) -> list[FusionGroup]:
        """
        Partition the graph into fusion groups.

        Algorithm:
        1. Topological sort
        2. Greedy forward pass: try to extend current group with each op
        3. If can't extend, start a new group
        4. Post-process: merge small adjacent groups if compatible
        """
        topo_order = graph.topological_order()
        op_to_group: dict[int, int] = {}
        groups: dict[int, FusionGroup] = {}

        for op_id in topo_order:
            op = graph.ops[op_id]

            # Skip input/output/constant nodes
            if op.op_type in {OpType.INPUT, OpType.OUTPUT, OpType.CONSTANT}:
                continue

            # Try to join the group of a producer
            best_group = None
            for inp_id in op.inputs:
                if inp_id in op_to_group:
                    candidate = groups[op_to_group[inp_id]]
                    if candidate.can_add(op, graph):
                        if best_group is None or candidate.size > best_group.size:
                            best_group = candidate

            if best_group is not None:
                best_group.op_ids.append(op_id)
                if graph.is_reduction(op_id):
                    best_group.has_reduction = True
                    best_group.reduction_dim = op.reduction_dim
                if op.op_type == OpType.MATMUL:
                    best_group.has_matmul = True
                op_to_group[op_id] = best_group.group_id
            else:
                # Start new group
                group = FusionGroup(group_id=self._group_counter)
                self._group_counter += 1
                group.op_ids.append(op_id)
                if graph.is_reduction(op_id):
                    group.has_reduction = True
                    group.reduction_dim = op.reduction_dim
                if op.op_type == OpType.MATMUL:
                    group.has_matmul = True
                groups[group.group_id] = group
                op_to_group[op_id] = group.group_id

        # Post-process: merge single-op groups with neighbors
        group_list = list(groups.values())
        group_list = self._merge_small_groups(group_list, graph)

        return group_list

    def _merge_small_groups(self, groups: list[FusionGroup],
                            graph: DataflowGraph) -> list[FusionGroup]:
        """Merge single-op groups into adjacent groups when possible."""
        if len(groups) <= 1:
            return groups

        merged = [groups[0]]
        for group in groups[1:]:
            prev = merged[-1]
            if group.size == 1 and not group.has_matmul:
                op = graph.ops[group.op_ids[0]]
                if prev.can_add(op, graph):
                    prev.op_ids.extend(group.op_ids)
                    if group.has_reduction:
                        prev.has_reduction = True
                    continue
            merged.append(group)

        return merged

    def report(self, graph: DataflowGraph,
               groups: list[FusionGroup]) -> str:
        """Generate a human-readable fusion report."""
        lines = [f"Fusion Report for '{graph.name}':",
                 f"  Total ops: {len(graph.ops)}",
                 f"  Fusion groups (kernels): {len(groups)}",
                 f"  Fusion ratio: {len(graph.ops) / max(len(groups), 1):.1f}x",
                 ""]

        for group in groups:
            op_names = []
            for op_id in group.op_ids:
                op = graph.ops[op_id]
                op_names.append(f"{op.op_type.name}")
            lines.append(f"  Kernel {group.group_id}: [{' → '.join(op_names)}]"
                         f" ({'matmul' if group.has_matmul else 'elementwise'}"
                         f"{'+reduction' if group.has_reduction else ''})")

        return "\n".join(lines)
