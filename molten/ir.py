"""
Dataflow IR — the intermediate representation.

Mathematical expressions are parsed into a dataflow graph where:
- Nodes are operations (add, mul, matmul, softmax, etc.)
- Edges are data dependencies
- The graph is the unit of fusion analysis

The IR is hardware-independent. The codegen backend lowers it
to GPU-specific code.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional


class OpType(Enum):
    # Elementwise
    ADD = auto()
    SUB = auto()
    MUL = auto()
    DIV = auto()
    NEG = auto()
    SQRT = auto()
    RSQRT = auto()
    EXP = auto()
    LOG = auto()
    TANH = auto()
    SIGMOID = auto()
    GELU = auto()
    SILU = auto()
    RELU = auto()
    ABS = auto()
    CLAMP = auto()

    # Reduction
    SUM = auto()
    MEAN = auto()
    MAX = auto()
    MIN = auto()
    NORM = auto()      # L2 norm
    RMS = auto()       # root mean square

    # Matrix ops
    MATMUL = auto()
    DOT = auto()
    OUTER = auto()

    # Attention primitives
    SOFTMAX = auto()
    ROTATE = auto()    # RoPE rotation

    # Data movement
    TRANSPOSE = auto()
    RESHAPE = auto()
    SLICE = auto()
    CONCAT = auto()
    BROADCAST = auto()

    # Special
    INPUT = auto()
    OUTPUT = auto()
    CONSTANT = auto()


@dataclass
class TensorShape:
    """Shape descriptor with symbolic dimensions."""
    dims: list[int | str]  # int for known dims, str for symbolic

    def __repr__(self):
        return f"({', '.join(str(d) for d in self.dims)})"

    @property
    def ndim(self) -> int:
        return len(self.dims)

    @property
    def is_fully_known(self) -> bool:
        return all(isinstance(d, int) for d in self.dims)


@dataclass
class Op:
    """A single operation in the dataflow graph."""
    id: int
    op_type: OpType
    name: str = ""
    inputs: list[int] = field(default_factory=list)   # Op IDs of inputs
    shape: Optional[TensorShape] = None
    dtype: str = "float16"
    attrs: dict = field(default_factory=dict)  # op-specific attributes

    # Fusion metadata
    fusable: bool = True                # can this be fused with neighbors?
    memory_bound: bool = False          # is this op memory-bound?
    reduction_dim: Optional[int] = None # for reduction ops

    def __repr__(self):
        return f"Op({self.id}: {self.op_type.name} '{self.name}' {self.shape})"


class DataflowGraph:
    """
    Dataflow graph — the core IR for Molten.

    Represents a computation as a directed acyclic graph of operations.
    This is the unit that gets fused into a single GPU kernel.
    """

    def __init__(self, name: str = "unnamed"):
        self.name = name
        self.ops: dict[int, Op] = {}
        self._next_id = 0
        self._outputs: list[int] = []

    def add_op(self, op_type: OpType, inputs: list[int] = None,
               name: str = "", shape: Optional[TensorShape] = None,
               dtype: str = "float16", **attrs) -> int:
        """Add an operation and return its ID."""
        op_id = self._next_id
        self._next_id += 1
        self.ops[op_id] = Op(
            id=op_id,
            op_type=op_type,
            name=name,
            inputs=inputs or [],
            shape=shape,
            dtype=dtype,
            attrs=attrs,
        )
        return op_id

    def add_input(self, name: str, shape: TensorShape,
                  dtype: str = "float16") -> int:
        return self.add_op(OpType.INPUT, name=name, shape=shape, dtype=dtype)

    def add_output(self, input_id: int, name: str = "output") -> int:
        out_id = self.add_op(OpType.OUTPUT, inputs=[input_id], name=name,
                             shape=self.ops[input_id].shape)
        self._outputs.append(out_id)
        return out_id

    def add_constant(self, name: str, value: float) -> int:
        return self.add_op(OpType.CONSTANT, name=name, value=value)

    # --- Convenience builders ---

    def add(self, a: int, b: int, name: str = "") -> int:
        return self.add_op(OpType.ADD, [a, b], name)

    def mul(self, a: int, b: int, name: str = "") -> int:
        return self.add_op(OpType.MUL, [a, b], name)

    def div(self, a: int, b: int, name: str = "") -> int:
        return self.add_op(OpType.DIV, [a, b], name)

    def matmul(self, a: int, b: int, name: str = "") -> int:
        return self.add_op(OpType.MATMUL, [a, b], name)

    def softmax(self, x: int, dim: int = -1, name: str = "") -> int:
        op_id = self.add_op(OpType.SOFTMAX, [x], name)
        self.ops[op_id].reduction_dim = dim
        return op_id

    def rms_norm(self, x: int, weight: int, name: str = "") -> int:
        """RMSNorm: x / rms(x) * weight"""
        rms_id = self.add_op(OpType.RMS, [x], f"{name}_rms")
        div_id = self.div(x, rms_id, f"{name}_div")
        return self.mul(div_id, weight, f"{name}_scale")

    def rope(self, x: int, freqs: int, name: str = "") -> int:
        return self.add_op(OpType.ROTATE, [x, freqs], name)

    # --- Analysis ---

    def inputs(self) -> list[Op]:
        return [op for op in self.ops.values() if op.op_type == OpType.INPUT]

    def outputs(self) -> list[Op]:
        return [op for op in self.ops.values() if op.op_type == OpType.OUTPUT]

    def consumers(self, op_id: int) -> list[int]:
        """Get IDs of ops that consume this op's output."""
        return [
            other_id for other_id, op in self.ops.items()
            if op_id in op.inputs
        ]

    def producers(self, op_id: int) -> list[int]:
        """Get IDs of ops that this op consumes."""
        return self.ops[op_id].inputs

    def topological_order(self) -> list[int]:
        """Return op IDs in topological order."""
        visited = set()
        order = []

        def visit(op_id):
            if op_id in visited:
                return
            visited.add(op_id)
            for inp in self.ops[op_id].inputs:
                visit(inp)
            order.append(op_id)

        for op_id in self.ops:
            visit(op_id)

        return order

    def is_elementwise(self, op_id: int) -> bool:
        """Check if an op is purely elementwise (fusable without shared memory)."""
        op = self.ops[op_id]
        return op.op_type in {
            OpType.ADD, OpType.SUB, OpType.MUL, OpType.DIV,
            OpType.NEG, OpType.SQRT, OpType.RSQRT, OpType.EXP,
            OpType.LOG, OpType.TANH, OpType.SIGMOID, OpType.GELU,
            OpType.SILU, OpType.RELU, OpType.ABS, OpType.CLAMP,
        }

    def is_reduction(self, op_id: int) -> bool:
        op = self.ops[op_id]
        return op.op_type in {
            OpType.SUM, OpType.MEAN, OpType.MAX, OpType.MIN,
            OpType.NORM, OpType.RMS, OpType.SOFTMAX,
        }

    def __repr__(self):
        lines = [f"DataflowGraph '{self.name}' ({len(self.ops)} ops):"]
        for op_id in self.topological_order():
            lines.append(f"  {self.ops[op_id]}")
        return "\n".join(lines)
