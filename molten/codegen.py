"""
Code Generator — lowers fused IR to CUDA kernel source.

Takes a FusionGroup and generates a complete CUDA kernel:
- Thread/block dimensions based on tensor shapes
- Shared memory allocation for reductions
- Register-level computation for elementwise ops
- Vectorized loads/stores for memory efficiency

The generated kernels target compute capability 8.0+ (Ampere, Ada, Blackwell).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

from molten.ir import DataflowGraph, Op, OpType, TensorShape
from molten.fusion import FusionGroup


@dataclass
class KernelConfig:
    block_size_x: int = 256
    block_size_y: int = 1
    vectorize: int = 4          # float4 loads
    shared_mem_bytes: int = 0
    registers_per_thread: int = 32
    dtype: str = "float"        # "float" or "half"
    use_half2: bool = True      # fused half-precision pairs


@dataclass
class GeneratedKernel:
    name: str
    source: str
    config: KernelConfig
    num_args: int
    estimated_flops: int = 0
    estimated_memory_bytes: int = 0

    @property
    def arithmetic_intensity(self) -> float:
        if self.estimated_memory_bytes == 0:
            return 0.0
        return self.estimated_flops / self.estimated_memory_bytes


# Map OpType to CUDA expression templates
_ELEMENTWISE_TEMPLATES = {
    OpType.ADD: "{a} + {b}",
    OpType.SUB: "{a} - {b}",
    OpType.MUL: "{a} * {b}",
    OpType.DIV: "{a} / {b}",
    OpType.NEG: "-{a}",
    OpType.SQRT: "sqrtf({a})",
    OpType.RSQRT: "rsqrtf({a})",
    OpType.EXP: "expf({a})",
    OpType.LOG: "logf({a})",
    OpType.TANH: "tanhf({a})",
    OpType.SIGMOID: "1.0f / (1.0f + expf(-{a}))",
    OpType.GELU: "{a} * 0.5f * (1.0f + tanhf(0.7978845608f * ({a} + 0.044715f * {a} * {a} * {a})))",
    OpType.SILU: "{a} * (1.0f / (1.0f + expf(-{a})))",
    OpType.RELU: "fmaxf({a}, 0.0f)",
    OpType.ABS: "fabsf({a})",
}


class CodeGenerator:
    """
    Generates CUDA kernel source from fused operation groups.

    For each FusionGroup, emits:
    1. Kernel signature with typed arguments
    2. Thread index computation
    3. Vectorized loads from global memory
    4. Fused computation in registers
    5. Vectorized stores to global memory
    """

    def __init__(self, compute_capability: int = 80, fp16: bool = False):
        self.cc = compute_capability
        self.fp16 = fp16
        self.dtype = "half" if fp16 else "float"
        self.ptr_type = "__half" if fp16 else "float"
        self.compute_type = "float"  # always compute in fp32, store in fp16

    def generate(self, graph: DataflowGraph,
                 group: FusionGroup) -> GeneratedKernel:
        """Generate a CUDA kernel for a fusion group."""
        ops = [graph.ops[op_id] for op_id in group.op_ids]
        name = self._make_name(ops)

        if group.has_matmul:
            return self._generate_matmul_kernel(graph, group, name)
        elif group.has_reduction:
            return self._generate_reduction_kernel(graph, group, name)
        else:
            return self._generate_elementwise_kernel(graph, group, name)

    def _generate_elementwise_kernel(self, graph: DataflowGraph,
                                     group: FusionGroup,
                                     name: str) -> GeneratedKernel:
        """Generate a fused elementwise kernel."""
        ops = [graph.ops[op_id] for op_id in group.op_ids]
        config = KernelConfig(block_size_x=256, vectorize=4)

        # Collect input/output tensors
        inputs = set()
        for op in ops:
            for inp_id in op.inputs:
                if graph.ops[inp_id].op_type == OpType.INPUT:
                    inputs.add(inp_id)

        input_list = sorted(inputs)
        num_inputs = len(input_list)

        # Generate kernel source
        lines = []
        lines.append(f"// Molten auto-generated: {name}")
        lines.append(f"// Fused ops: {' -> '.join(op.op_type.name for op in ops)}")
        if self.fp16:
            lines.append(f"// Precision: fp16 I/O, fp32 compute")
            lines.append("#include <cuda_fp16.h>")
        lines.append("")

        # Signature — fp16 I/O with fp32 compute, or pure fp32
        ptr_t = self.ptr_type
        args = []
        for i, inp_id in enumerate(input_list):
            args.append(f"const {ptr_t}* __restrict__ in{i}")
        args.append(f"{ptr_t}* __restrict__ out")
        args.append("const int N")

        lines.append(f"__global__ void {name}({', '.join(args)}) {{")
        lines.append(f"    const int idx = blockIdx.x * blockDim.x + threadIdx.x;")
        lines.append(f"    if (idx >= N) return;")
        lines.append("")

        # Load inputs (cast to fp32 for compute if fp16 I/O)
        for i in range(num_inputs):
            if self.fp16:
                lines.append(f"    float v{i} = __half2float(in{i}[idx]);")
            else:
                lines.append(f"    float v{i} = in{i}[idx];")

        # Fused computation — track op_id -> variable name mapping
        result_var = f"v0"
        temp_counter = num_inputs
        op_id_to_var: dict[int, str] = {}
        for i, inp_id in enumerate(input_list):
            op_id_to_var[inp_id] = f"v{i}"

        for op in ops:
            template = _ELEMENTWISE_TEMPLATES.get(op.op_type)
            if template is None:
                continue

            if len(op.inputs) == 2:
                a_var = op_id_to_var.get(op.inputs[0], f"v0")
                b_var = op_id_to_var.get(op.inputs[1], f"v0")
                expr = template.format(a=a_var, b=b_var)
            elif len(op.inputs) == 1:
                a_var = op_id_to_var.get(op.inputs[0], f"v0")
                expr = template.format(a=a_var)
            else:
                continue

            result_var = f"t{temp_counter}"
            temp_counter += 1
            op_id_to_var[op.id] = result_var
            lines.append(f"    float {result_var} = {expr};")

        if self.fp16:
            lines.append(f"    out[idx] = __float2half({result_var});")
        else:
            lines.append(f"    out[idx] = {result_var};")
        lines.append("}")

        source = "\n".join(lines)
        return GeneratedKernel(
            name=name, source=source, config=config,
            num_args=num_inputs + 2,
        )

    def _generate_reduction_kernel(self, graph: DataflowGraph,
                                   group: FusionGroup,
                                   name: str) -> GeneratedKernel:
        """Generate a kernel with reduction (softmax, norm, etc.)."""
        config = KernelConfig(
            block_size_x=256,
            shared_mem_bytes=256 * 4,  # float per thread
        )

        lines = []
        lines.append(f"// Molten auto-generated: {name} (with reduction)")
        lines.append(f"__global__ void {name}(")
        lines.append(f"    const float* __restrict__ input,")
        lines.append(f"    float* __restrict__ output,")
        lines.append(f"    const int rows, const int cols) {{")
        lines.append(f"")
        lines.append(f"    extern __shared__ float sdata[];")
        lines.append(f"    const int tid = threadIdx.x;")
        lines.append(f"    const int row = blockIdx.x;")
        lines.append(f"    if (row >= rows) return;")
        lines.append(f"")
        lines.append(f"    const float* row_in = input + row * cols;")
        lines.append(f"    float* row_out = output + row * cols;")
        lines.append(f"")

        # Check what type of reduction
        has_softmax = any(
            graph.ops[op_id].op_type == OpType.SOFTMAX
            for op_id in group.op_ids
        )
        has_rms = any(
            graph.ops[op_id].op_type == OpType.RMS
            for op_id in group.op_ids
        )

        if has_softmax:
            lines.extend(self._softmax_body())
        elif has_rms:
            lines.extend(self._rms_norm_body())
        else:
            lines.extend(self._generic_reduction_body())

        lines.append("}")

        source = "\n".join(lines)
        return GeneratedKernel(name=name, source=source, config=config, num_args=4)

    def _softmax_body(self) -> list[str]:
        return [
            "    // Pass 1: find max (for numerical stability)",
            "    float max_val = -INFINITY;",
            "    for (int i = tid; i < cols; i += blockDim.x) {",
            "        max_val = fmaxf(max_val, row_in[i]);",
            "    }",
            "    sdata[tid] = max_val;",
            "    __syncthreads();",
            "    for (int s = blockDim.x / 2; s > 0; s >>= 1) {",
            "        if (tid < s) sdata[tid] = fmaxf(sdata[tid], sdata[tid + s]);",
            "        __syncthreads();",
            "    }",
            "    max_val = sdata[0];",
            "    __syncthreads();",
            "",
            "    // Pass 2: exp and sum",
            "    float sum_val = 0.0f;",
            "    for (int i = tid; i < cols; i += blockDim.x) {",
            "        float e = expf(row_in[i] - max_val);",
            "        row_out[i] = e;",
            "        sum_val += e;",
            "    }",
            "    sdata[tid] = sum_val;",
            "    __syncthreads();",
            "    for (int s = blockDim.x / 2; s > 0; s >>= 1) {",
            "        if (tid < s) sdata[tid] += sdata[tid + s];",
            "        __syncthreads();",
            "    }",
            "    float inv_sum = 1.0f / sdata[0];",
            "    __syncthreads();",
            "",
            "    // Pass 3: normalize",
            "    for (int i = tid; i < cols; i += blockDim.x) {",
            "        row_out[i] *= inv_sum;",
            "    }",
        ]

    def _rms_norm_body(self) -> list[str]:
        return [
            "    // Compute sum of squares",
            "    float ss = 0.0f;",
            "    for (int i = tid; i < cols; i += blockDim.x) {",
            "        float v = row_in[i];",
            "        ss += v * v;",
            "    }",
            "    sdata[tid] = ss;",
            "    __syncthreads();",
            "    for (int s = blockDim.x / 2; s > 0; s >>= 1) {",
            "        if (tid < s) sdata[tid] += sdata[tid + s];",
            "        __syncthreads();",
            "    }",
            "    float rms = rsqrtf(sdata[0] / (float)cols + 1e-6f);",
            "    __syncthreads();",
            "",
            "    // Normalize",
            "    for (int i = tid; i < cols; i += blockDim.x) {",
            "        row_out[i] = row_in[i] * rms;",
            "    }",
        ]

    def _generic_reduction_body(self) -> list[str]:
        return [
            "    float acc = 0.0f;",
            "    for (int i = tid; i < cols; i += blockDim.x) {",
            "        acc += row_in[i];",
            "    }",
            "    sdata[tid] = acc;",
            "    __syncthreads();",
            "    for (int s = blockDim.x / 2; s > 0; s >>= 1) {",
            "        if (tid < s) sdata[tid] += sdata[tid + s];",
            "        __syncthreads();",
            "    }",
            "    if (tid == 0) row_out[0] = sdata[0];",
        ]

    def _generate_matmul_kernel(self, graph: DataflowGraph,
                                group: FusionGroup,
                                name: str) -> GeneratedKernel:
        """Generate a tiled matmul kernel with fused epilogue."""
        config = KernelConfig(
            block_size_x=16,
            block_size_y=16,
            shared_mem_bytes=2 * 16 * 16 * 4,  # two tiles
        )

        # Collect epilogue ops (ops after matmul)
        matmul_idx = None
        epilogue_ops = []
        for op_id in group.op_ids:
            if graph.ops[op_id].op_type == OpType.MATMUL:
                matmul_idx = op_id
            elif matmul_idx is not None:
                epilogue_ops.append(graph.ops[op_id])

        epilogue_expr = "acc"
        for op in epilogue_ops:
            template = _ELEMENTWISE_TEMPLATES.get(op.op_type)
            if template:
                epilogue_expr = template.format(a=epilogue_expr, b="bias_val")

        lines = [
            f"// Molten auto-generated: {name} (matmul + epilogue)",
            f"#define TILE 16",
            f"__global__ void {name}(",
            f"    const float* __restrict__ A,",
            f"    const float* __restrict__ B,",
            f"    float* __restrict__ C,",
            f"    const int M, const int N, const int K) {{",
            f"",
            f"    __shared__ float As[TILE][TILE];",
            f"    __shared__ float Bs[TILE][TILE];",
            f"",
            f"    const int row = blockIdx.y * TILE + threadIdx.y;",
            f"    const int col = blockIdx.x * TILE + threadIdx.x;",
            f"    float acc = 0.0f;",
            f"",
            f"    for (int t = 0; t < (K + TILE - 1) / TILE; t++) {{",
            f"        int ak = t * TILE + threadIdx.x;",
            f"        int bk = t * TILE + threadIdx.y;",
            f"        As[threadIdx.y][threadIdx.x] = (row < M && ak < K) ? A[row * K + ak] : 0.0f;",
            f"        Bs[threadIdx.y][threadIdx.x] = (bk < K && col < N) ? B[bk * N + col] : 0.0f;",
            f"        __syncthreads();",
            f"",
            f"        #pragma unroll",
            f"        for (int i = 0; i < TILE; i++) {{",
            f"            acc += As[threadIdx.y][i] * Bs[i][threadIdx.x];",
            f"        }}",
            f"        __syncthreads();",
            f"    }}",
            f"",
            f"    if (row < M && col < N) {{",
            f"        C[row * N + col] = {epilogue_expr};",
            f"    }}",
            f"}}",
        ]

        source = "\n".join(lines)
        return GeneratedKernel(name=name, source=source, config=config, num_args=6)

    def _resolve_var(self, op_id: int, input_list: list[int],
                     graph: DataflowGraph) -> str:
        if op_id in input_list:
            return f"v{input_list.index(op_id)}"
        return f"t{op_id}"

    @staticmethod
    def _make_name(ops: list[Op]) -> str:
        parts = []
        for op in ops:
            name = op.op_type.name.lower()
            if name not in parts:
                parts.append(name)
        return "molten_fused_" + "_".join(parts[:4])


def compile_graph(graph: DataflowGraph,
                  compute_capability: int = 80,
                  fp16: bool = False) -> list[GeneratedKernel]:
    """Full pipeline: fuse + codegen."""
    from molten.fusion import FusionEngine

    engine = FusionEngine()
    groups = engine.fuse(graph)
    codegen = CodeGenerator(compute_capability, fp16=fp16)

    kernels = []
    for group in groups:
        kernel = codegen.generate(graph, group)
        kernels.append(kernel)

    return kernels
