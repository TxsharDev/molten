"""Tests for CUDA code generation."""

import pytest
from molten.ir import DataflowGraph, OpType, TensorShape
from molten.fusion import FusionEngine
from molten.codegen import CodeGenerator, compile_graph


class TestCodeGenerator:
    def test_elementwise_kernel(self):
        g = DataflowGraph("add_relu")
        x = g.add_input("x", TensorShape([1024]))
        y = g.add_input("y", TensorShape([1024]))
        a = g.add(x, y)
        r = g.add_op(OpType.RELU, [a])
        g.add_output(r)

        kernels = compile_graph(g)
        assert len(kernels) >= 1
        assert "molten_fused" in kernels[0].name
        assert "__global__" in kernels[0].source
        assert "fmaxf" in kernels[0].source  # RELU

    def test_softmax_kernel(self):
        g = DataflowGraph("softmax")
        x = g.add_input("x", TensorShape([32, 128]))
        s = g.softmax(x)
        g.add_output(s)

        kernels = compile_graph(g)
        assert len(kernels) >= 1
        source = kernels[0].source
        assert "__shared__" in source
        assert "expf" in source

    def test_matmul_kernel(self):
        g = DataflowGraph("gemm")
        a = g.add_input("a", TensorShape([64, 128]))
        b = g.add_input("b", TensorShape([128, 64]))
        m = g.matmul(a, b)
        g.add_output(m)

        kernels = compile_graph(g)
        assert len(kernels) >= 1
        source = kernels[0].source
        assert "TILE" in source
        assert "__syncthreads" in source

    def test_rms_norm_kernel(self):
        g = DataflowGraph("rmsnorm")
        x = g.add_input("x", TensorShape([4, 512]))
        w = g.add_input("w", TensorShape([512]))
        out = g.rms_norm(x, w, "norm")
        g.add_output(out)

        kernels = compile_graph(g)
        assert len(kernels) >= 1

    def test_compile_graph_end_to_end(self):
        """Test full pipeline: IR → fusion → codegen."""
        g = DataflowGraph("transformer_block")
        x = g.add_input("x", TensorShape([4, 512]))
        w = g.add_input("w", TensorShape([512]))
        normed = g.rms_norm(x, w, "norm")

        q = g.add_input("q", TensorShape([4, 512]))
        k = g.add_input("k", TensorShape([512, 4]))
        scores = g.matmul(q, k)
        attn = g.softmax(scores)
        g.add_output(attn)

        kernels = compile_graph(g)
        assert len(kernels) >= 1
        for k in kernels:
            assert k.source
            assert k.name
