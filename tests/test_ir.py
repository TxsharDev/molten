"""Tests for the dataflow IR."""

import pytest
from molten.ir import DataflowGraph, OpType, TensorShape


class TestDataflowGraph:
    def test_add_ops(self):
        g = DataflowGraph("test")
        x = g.add_input("x", TensorShape([4, 8]))
        w = g.add_input("w", TensorShape([4, 8]))
        add = g.add(x, w)
        out = g.add_output(add)

        assert len(g.ops) == 4
        assert g.ops[x].op_type == OpType.INPUT
        assert g.ops[add].op_type == OpType.ADD

    def test_topological_order(self):
        g = DataflowGraph("test")
        x = g.add_input("x", TensorShape([4]))
        y = g.add_input("y", TensorShape([4]))
        z = g.add(x, y)
        out = g.add_output(z)

        order = g.topological_order()
        # Inputs before add, add before output
        assert order.index(x) < order.index(z)
        assert order.index(y) < order.index(z)
        assert order.index(z) < order.index(out)

    def test_consumers_producers(self):
        g = DataflowGraph("test")
        x = g.add_input("x", TensorShape([4]))
        y = g.add_input("y", TensorShape([4]))
        z = g.add(x, y)

        assert g.consumers(x) == [z]
        assert set(g.producers(z)) == {x, y}

    def test_is_elementwise(self):
        g = DataflowGraph("test")
        x = g.add_input("x", TensorShape([4]))
        y = g.add_input("y", TensorShape([4]))
        add = g.add(x, y)
        mm = g.matmul(x, y)
        sm = g.softmax(x)

        assert g.is_elementwise(add)
        assert not g.is_elementwise(mm)
        assert not g.is_elementwise(sm)

    def test_is_reduction(self):
        g = DataflowGraph("test")
        x = g.add_input("x", TensorShape([4]))
        sm = g.softmax(x)
        rms = g.add_op(OpType.RMS, [x])

        assert g.is_reduction(sm)
        assert g.is_reduction(rms)

    def test_rms_norm_builder(self):
        g = DataflowGraph("rmsnorm")
        x = g.add_input("x", TensorShape([4, 8]))
        w = g.add_input("w", TensorShape([8]))
        out = g.rms_norm(x, w, "norm")

        assert len(g.ops) >= 4  # input, input, rms, div, mul

    def test_repr(self):
        g = DataflowGraph("test")
        x = g.add_input("x", TensorShape([4]))
        s = repr(g)
        assert "test" in s


class TestTensorShape:
    def test_known_shape(self):
        s = TensorShape([4, 8, 16])
        assert s.ndim == 3
        assert s.is_fully_known

    def test_symbolic_shape(self):
        s = TensorShape(["batch", 8, "seq"])
        assert s.ndim == 3
        assert not s.is_fully_known
