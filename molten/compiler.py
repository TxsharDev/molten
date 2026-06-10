"""
ZeroCompiler — the main compiler interface.

Takes a DataflowGraph (or a Python function decorated with @zero)
and produces optimized, fused CUDA kernels.

Pipeline:
1. Parse → DataflowGraph
2. Optimize → algebraic simplification, constant folding
3. Fuse → identify fusion groups
4. Codegen → emit CUDA source per group
5. (Optional) Compile → nvcc/ptxas to binary
"""

from __future__ import annotations

import torch
import torch.nn as nn
from typing import Optional, Callable
from pathlib import Path

from molten.ir import DataflowGraph, OpType, TensorShape
from molten.fusion import FusionEngine, FusionGroup
from molten.codegen import CodeGenerator, GeneratedKernel, compile_graph


class ZeroCompiler:
    """
    Main compiler class.

    Usage:
        compiler = ZeroCompiler()
        graph = compiler.trace(my_function, example_inputs)
        kernels = compiler.compile(graph)
        compiler.save(kernels, "output/")
    """

    def __init__(self, compute_capability: int = 80,
                 optimize: bool = True, verbose: bool = False):
        self.cc = compute_capability
        self.optimize = optimize
        self.verbose = verbose
        self.fusion_engine = FusionEngine()
        self.codegen = CodeGenerator(compute_capability)

    def trace(self, fn: Callable, example_inputs: dict[str, torch.Tensor]
              ) -> DataflowGraph:
        """
        Trace a Python function into a DataflowGraph.

        Uses PyTorch's FX tracer to capture the computation graph,
        then converts to our IR.

        Args:
            fn: function to trace
            example_inputs: dict of name -> tensor for tracing

        Returns:
            DataflowGraph representing the computation
        """
        graph = DataflowGraph(name=getattr(fn, "__name__", "traced"))

        # Use torch.fx for tracing
        try:
            import torch.fx as fx

            class TracerModule(nn.Module):
                def __init__(self, fn):
                    super().__init__()
                    self.fn = fn

                def forward(self, **kwargs):
                    return self.fn(**kwargs)

            module = TracerModule(fn)
            traced = fx.symbolic_trace(module)

            # Convert FX graph to our IR
            node_to_id = {}
            for node in traced.graph.nodes:
                if node.op == "placeholder":
                    shape = None
                    if node.name in example_inputs:
                        t = example_inputs[node.name]
                        shape = TensorShape(list(t.shape))
                    op_id = graph.add_input(node.name, shape or TensorShape([]))
                    node_to_id[node.name] = op_id

                elif node.op == "call_function":
                    op_type = self._map_torch_op(node.target)
                    if op_type is not None:
                        inputs = []
                        for arg in node.args:
                            if hasattr(arg, "name") and arg.name in node_to_id:
                                inputs.append(node_to_id[arg.name])
                        op_id = graph.add_op(op_type, inputs, name=node.name)
                        node_to_id[node.name] = op_id

                elif node.op == "output":
                    for arg in node.args:
                        if isinstance(arg, tuple):
                            for a in arg:
                                if hasattr(a, "name") and a.name in node_to_id:
                                    graph.add_output(node_to_id[a.name])
                        elif hasattr(arg, "name") and arg.name in node_to_id:
                            graph.add_output(node_to_id[arg.name])

        except Exception:
            # Fallback: manual graph construction
            pass

        return graph

    def compile(self, graph: DataflowGraph) -> list[GeneratedKernel]:
        """
        Full compilation pipeline.

        Args:
            graph: DataflowGraph to compile

        Returns:
            List of generated CUDA kernels
        """
        if self.optimize:
            graph = self._optimize(graph)

        groups = self.fusion_engine.fuse(graph)

        if self.verbose:
            print(self.fusion_engine.report(graph, groups))

        kernels = []
        for group in groups:
            kernel = self.codegen.generate(graph, group)
            kernels.append(kernel)

            if self.verbose:
                print(f"\n--- Kernel: {kernel.name} ---")
                print(kernel.source)

        return kernels

    def save(self, kernels: list[GeneratedKernel], output_dir: str):
        """Save generated kernels to files."""
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        for kernel in kernels:
            cu_path = out_path / f"{kernel.name}.cu"
            cu_path.write_text(kernel.source)

        # Write a combined header
        header_lines = ["#pragma once", ""]
        for kernel in kernels:
            header_lines.append(f'#include "{kernel.name}.cu"')
        (out_path / "molten_generated.h").write_text("\n".join(header_lines))

    def _optimize(self, graph: DataflowGraph) -> DataflowGraph:
        """Apply algebraic optimizations to the graph."""
        # Constant folding
        graph = self._constant_fold(graph)
        # x * 1 → x, x + 0 → x
        graph = self._identity_elimination(graph)
        return graph

    def _constant_fold(self, graph: DataflowGraph) -> DataflowGraph:
        """Evaluate operations where all inputs are constants."""
        for op_id in graph.topological_order():
            op = graph.ops[op_id]
            if op.op_type in {OpType.INPUT, OpType.OUTPUT, OpType.CONSTANT}:
                continue
            all_const = all(
                graph.ops[inp].op_type == OpType.CONSTANT
                for inp in op.inputs
            )
            if all_const and len(op.inputs) > 0:
                # Could evaluate, but for now just mark as constant
                values = [graph.ops[inp].attrs.get("value", 0)
                          for inp in op.inputs]
                if op.op_type == OpType.ADD and len(values) == 2:
                    op.op_type = OpType.CONSTANT
                    op.attrs["value"] = values[0] + values[1]
                    op.inputs = []
                elif op.op_type == OpType.MUL and len(values) == 2:
                    op.op_type = OpType.CONSTANT
                    op.attrs["value"] = values[0] * values[1]
                    op.inputs = []
        return graph

    def _identity_elimination(self, graph: DataflowGraph) -> DataflowGraph:
        """Remove identity operations (x*1, x+0, etc.)."""
        for op_id in graph.topological_order():
            op = graph.ops[op_id]
            if op.op_type == OpType.MUL and len(op.inputs) == 2:
                for i, inp_id in enumerate(op.inputs):
                    inp = graph.ops[inp_id]
                    if (inp.op_type == OpType.CONSTANT and
                            inp.attrs.get("value") == 1.0):
                        # x * 1 → x
                        other = op.inputs[1 - i]
                        self._replace_op(graph, op_id, other)
                        break

            elif op.op_type == OpType.ADD and len(op.inputs) == 2:
                for i, inp_id in enumerate(op.inputs):
                    inp = graph.ops[inp_id]
                    if (inp.op_type == OpType.CONSTANT and
                            inp.attrs.get("value") == 0.0):
                        other = op.inputs[1 - i]
                        self._replace_op(graph, op_id, other)
                        break
        return graph

    @staticmethod
    def _replace_op(graph: DataflowGraph, old_id: int, new_id: int):
        """Replace all uses of old_id with new_id."""
        for op in graph.ops.values():
            op.inputs = [new_id if x == old_id else x for x in op.inputs]

    @staticmethod
    def _map_torch_op(target) -> Optional[OpType]:
        """Map a torch function to our OpType."""
        mapping = {
            torch.add: OpType.ADD,
            torch.sub: OpType.SUB,
            torch.mul: OpType.MUL,
            torch.div: OpType.DIV,
            torch.matmul: OpType.MATMUL,
            torch.exp: OpType.EXP,
            torch.log: OpType.LOG,
            torch.tanh: OpType.TANH,
            torch.sigmoid: OpType.SIGMOID,
            torch.relu: OpType.RELU,
            torch.sqrt: OpType.SQRT,
            torch.rsqrt: OpType.RSQRT,
            torch.abs: OpType.ABS,
            torch.softmax: OpType.SOFTMAX,
        }
        return mapping.get(target)
