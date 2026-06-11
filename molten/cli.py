"""
Molten CLI — command-line interface for kernel generation.

Usage:
    molten compile graph.json --output kernels/
    molten benchmark --dim 3584 --seq 2048
    molten info
"""

import argparse
import sys
import json
from pathlib import Path

from molten.ir import DataflowGraph, OpType, TensorShape
from molten.compiler import ZeroCompiler
from molten.codegen import compile_graph


def cmd_compile(args):
    """Compile a DataflowGraph JSON to .cu files."""
    graph_path = Path(args.graph)
    if not graph_path.exists():
        print("Error: %s not found" % graph_path)
        return 1

    data = json.loads(graph_path.read_text())
    # Build graph from JSON spec
    g = DataflowGraph(data.get("name", "unnamed"))
    for op_spec in data.get("ops", []):
        g.add_op(
            OpType[op_spec["type"]],
            inputs=op_spec.get("inputs", []),
            name=op_spec.get("name", ""),
        )

    compiler = ZeroCompiler(verbose=True, fp16=args.fp16)
    kernels = compiler.compile(g)

    out_dir = Path(args.output)
    compiler.save(kernels, str(out_dir))
    print("Generated %d kernel(s) in %s" % (len(kernels), out_dir))


def cmd_benchmark(args):
    """Generate and benchmark fused kernels at given dimensions."""
    dim = args.dim
    seq = args.seq

    print("Molten Benchmark")
    print("  Dimensions: seq=%d, dim=%d" % (seq, dim))

    # Build RMSNorm graph
    g = DataflowGraph("rmsnorm_bench")
    x = g.add_input("x", TensorShape([seq, dim]))
    w = g.add_input("w", TensorShape([dim]))
    out = g.rms_norm(x, w, "norm")
    g.add_output(out)

    compiler = ZeroCompiler(verbose=True)
    kernels = compiler.compile(g)

    print("\n  Generated %d kernel(s):" % len(kernels))
    for k in kernels:
        print("    %s (%d lines)" % (k.name, k.source.count("\n")))

    if args.output:
        compiler.save(kernels, args.output)
        print("  Saved to %s" % args.output)


def cmd_info():
    """Show Molten version and capabilities."""
    from molten import __version__
    print("Molten v%s" % __version__)
    print("  Fused GPU kernel generation from mathematical specifications")
    print("")
    print("  Supported ops: elementwise (add, mul, gelu, silu, relu, ...)")
    print("                 reduction (softmax, rmsnorm, sum, mean)")
    print("                 matmul (with epilogue fusion)")
    print("  Output: portable .cu files (no framework dependency)")
    print("  Precision: fp32 compute, fp16 I/O optional")


def main():
    parser = argparse.ArgumentParser(
        prog="molten",
        description="Molten: math to fused CUDA kernels"
    )
    sub = parser.add_subparsers(dest="command")

    # compile
    p_compile = sub.add_parser("compile", help="Compile a graph to .cu")
    p_compile.add_argument("graph", help="Path to graph JSON")
    p_compile.add_argument("--output", "-o", default="output/", help="Output directory")
    p_compile.add_argument("--fp16", action="store_true", help="Generate fp16 I/O kernels")

    # benchmark
    p_bench = sub.add_parser("benchmark", help="Generate and benchmark kernels")
    p_bench.add_argument("--dim", type=int, default=3584, help="Hidden dimension")
    p_bench.add_argument("--seq", type=int, default=2048, help="Sequence length")
    p_bench.add_argument("--output", "-o", default=None, help="Save kernels to dir")

    # info
    sub.add_parser("info", help="Show version and capabilities")

    args = parser.parse_args()

    if args.command == "compile":
        return cmd_compile(args)
    elif args.command == "benchmark":
        return cmd_benchmark(args)
    elif args.command == "info":
        return cmd_info()
    else:
        parser.print_help()


if __name__ == "__main__":
    sys.exit(main() or 0)
