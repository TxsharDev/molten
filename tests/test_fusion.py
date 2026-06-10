"""Tests for the fusion engine."""

import pytest
from molten.ir import DataflowGraph, OpType, TensorShape
from molten.fusion import FusionEngine


class TestFusionEngine:
    def test_elementwise_chain_fuses(self):
        """Chain of elementwise ops should become one group."""
        g = DataflowGraph("elem_chain")
        x = g.add_input("x", TensorShape([4, 8]))
        y = g.add_input("y", TensorShape([4, 8]))
        a = g.add(x, y)
        b = g.mul(a, x)
        c = g.add_op(OpType.RELU, [b])
        g.add_output(c)

        engine = FusionEngine()
        groups = engine.fuse(g)

        # All elementwise ops should be in one group
        assert len(groups) == 1
        assert len(groups[0].op_ids) == 3  # add, mul, relu

    def test_matmul_not_fused_with_matmul(self):
        """Two matmuls should be in separate groups."""
        g = DataflowGraph("two_matmul")
        x = g.add_input("x", TensorShape([4, 8]))
        w1 = g.add_input("w1", TensorShape([8, 16]))
        w2 = g.add_input("w2", TensorShape([16, 8]))
        m1 = g.matmul(x, w1)
        m2 = g.matmul(m1, w2)
        g.add_output(m2)

        engine = FusionEngine()
        groups = engine.fuse(g)

        # Two matmuls = at least 2 groups
        assert len(groups) >= 2
        for group in groups:
            matmul_count = sum(
                1 for op_id in group.op_ids
                if g.ops[op_id].op_type == OpType.MATMUL
            )
            assert matmul_count <= 1

    def test_matmul_plus_epilogue_fuses(self):
        """Matmul + elementwise epilogue should fuse."""
        g = DataflowGraph("matmul_epilogue")
        x = g.add_input("x", TensorShape([4, 8]))
        w = g.add_input("w", TensorShape([8, 16]))
        b = g.add_input("bias", TensorShape([16]))
        m = g.matmul(x, w)
        added = g.add(m, b)
        activated = g.add_op(OpType.RELU, [added])
        g.add_output(activated)

        engine = FusionEngine()
        groups = engine.fuse(g)

        # Should be 1 group: matmul + bias + relu
        assert len(groups) == 1

    def test_report(self):
        g = DataflowGraph("test")
        x = g.add_input("x", TensorShape([4]))
        y = g.add(x, x)
        g.add_output(y)

        engine = FusionEngine()
        groups = engine.fuse(g)
        report = engine.report(g, groups)
        assert "Fusion Report" in report
        assert "Kernel" in report
