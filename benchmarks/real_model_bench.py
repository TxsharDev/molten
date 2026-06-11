"""
Real Model Benchmark — Molten vs PyTorch on Qwen2.5-7B RMSNorm

First time Molten has been tested against an actual model's layers.
Loads Qwen2.5-7B, hooks RMSNorm layers, measures eager/compiled timing,
then generates the Molten fused kernel at the model's real dimensions.

Run with: python benchmarks/real_model_bench.py
"""

import sys
import time
import json
from pathlib import Path
from dataclasses import dataclass, asdict

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent.parent))

from molten.ir import DataflowGraph, TensorShape, OpType
from molten.codegen import compile_graph, CodeGenerator
from molten.fusion import FusionEngine, FusionGroup


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DEVICE = "cuda:1"  # RTX 5090
MODEL_ID = "Qwen/Qwen2.5-7B"
WARMUP = 50
TRIALS = 200
RESULT_PATH = Path(__file__).parent / "real_model_molten.json"

# Qwen2.5-7B dimensions
HIDDEN_SIZE = 3584
NUM_LAYERS = 28
NUM_HEADS = 28
NUM_KV_HEADS = 4
HEAD_DIM = 128
INTERMEDIATE_SIZE = 18944


@dataclass
class LayerResult:
    layer_name: str
    method: str
    shape: str
    time_us: float
    bandwidth_gb_s: float

    def __str__(self):
        return (f"  {self.layer_name:40s} | {self.method:15s} | "
                f"{self.time_us:8.1f} us | {self.bandwidth_gb_s:6.1f} GB/s")


# ---------------------------------------------------------------------------
# Benchmarking helpers
# ---------------------------------------------------------------------------

def bench_fn(fn, args, warmup=WARMUP, trials=TRIALS):
    """Benchmark with CUDA events, return median time in microseconds."""
    for _ in range(warmup):
        fn(*args)
    torch.cuda.synchronize()

    times = []
    for _ in range(trials):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn(*args)
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end) * 1000)  # ms -> us

    times.sort()
    q1 = len(times) // 4
    q3 = 3 * len(times) // 4
    return sum(times[q1:q3]) / (q3 - q1)


def bandwidth_gb_s(nbytes, time_us):
    if time_us == 0:
        return 0.0
    return nbytes / (time_us * 1e-6) / 1e9


# ---------------------------------------------------------------------------
# RMSNorm reference (matches HF Qwen2RmsNorm)
# ---------------------------------------------------------------------------

def rmsnorm_eager(hidden_states, weight, eps=1e-6):
    variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    hidden_states = hidden_states * torch.rsqrt(variance + eps)
    return (weight * hidden_states).to(hidden_states.dtype)


# ---------------------------------------------------------------------------
# Load model, discover RMSNorm layers
# ---------------------------------------------------------------------------

def load_model():
    print(f"Loading {MODEL_ID} on {DEVICE} (fp16, eager attn)...")
    from transformers import AutoModelForCausalLM, AutoConfig

    config = AutoConfig.from_pretrained(MODEL_ID, trust_remote_code=True)
    print(f"  hidden_size={config.hidden_size}  "
          f"num_layers={config.num_hidden_layers}  "
          f"num_heads={config.num_attention_heads}  "
          f"num_kv_heads={config.num_key_value_heads}")

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
        trust_remote_code=True,
        attn_implementation="eager",
        device_map=DEVICE,
    )
    model.eval()
    return model


def find_rmsnorm_layers(model):
    """Find all RMSNorm layers in the model."""
    layers = []
    for name, module in model.named_modules():
        cls_name = type(module).__name__
        if "RMSNorm" in cls_name or "rmsnorm" in cls_name.lower():
            layers.append((name, module))
    return layers


# ---------------------------------------------------------------------------
# Benchmark RMSNorm layers from the real model
# ---------------------------------------------------------------------------

def bench_model_rmsnorm(model, seq_lens=(1, 128, 512, 2048)):
    """Benchmark every RMSNorm layer at multiple sequence lengths."""
    norm_layers = find_rmsnorm_layers(model)
    print(f"\nFound {len(norm_layers)} RMSNorm layers in {MODEL_ID}")

    if not norm_layers:
        print("ERROR: No RMSNorm layers found.")
        return []

    # Show first few
    for name, mod in norm_layers[:4]:
        print(f"  {name}  weight.shape={tuple(mod.weight.shape)}")
    if len(norm_layers) > 4:
        print(f"  ... and {len(norm_layers) - 4} more")

    # Benchmark a representative subset: first, middle, last
    indices = [0, len(norm_layers) // 2, len(norm_layers) - 1]
    sample_layers = [(norm_layers[i][0], norm_layers[i][1]) for i in indices]

    all_results = []

    for seq_len in seq_lens:
        print(f"\n--- seq_len={seq_len} ---")
        print(f"  {'Layer':40s} | {'Method':15s} | {'Time':>10s} | {'BW':>10s}")
        print(f"  {'-'*40}-+-{'-'*15}-+-{'-'*10}-+-{'-'*10}")

        for name, mod in sample_layers:
            weight = mod.weight.data
            hidden = weight.shape[0]
            eps = getattr(mod, 'variance_epsilon',
                          getattr(mod, 'eps', 1e-6))

            x = torch.randn(1, seq_len, hidden, dtype=torch.float16,
                             device=DEVICE)
            nbytes = x.numel() * x.element_size() * 3  # read x, read w, write out

            # --- Eager ---
            t_eager = bench_fn(rmsnorm_eager, (x, weight, eps))
            r = LayerResult(name, "eager", f"(1,{seq_len},{hidden})",
                            t_eager, bandwidth_gb_s(nbytes, t_eager))
            all_results.append(r)
            print(r)

            # --- torch.compile ---
            compiled_norm = torch.compile(rmsnorm_eager)
            t_compiled = bench_fn(compiled_norm, (x, weight, eps))
            r = LayerResult(name, "torch.compile", f"(1,{seq_len},{hidden})",
                            t_compiled, bandwidth_gb_s(nbytes, t_compiled))
            all_results.append(r)
            print(r)

            # --- Speedup ---
            speedup = t_eager / t_compiled if t_compiled > 0 else 0
            print(f"  {'':40s}   compile speedup: {speedup:.2f}x")

    return all_results


# ---------------------------------------------------------------------------
# Molten: build DataflowGraph for RMSNorm at real model dimensions
# ---------------------------------------------------------------------------

def build_rmsnorm_graph(hidden=HIDDEN_SIZE):
    """
    Build Molten DataflowGraph for RMSNorm at the model's actual hidden size.

    RMSNorm(x, w) = x / sqrt(mean(x^2) + eps) * w
    The graph decomposes into: x -> RMS reduction -> div(x, rms) -> mul(weight)
    """
    g = DataflowGraph(f"qwen25_7b_rmsnorm_h{hidden}")

    # Inputs at real model dimensions
    x = g.add_input("hidden_states", TensorShape([1, 2048, hidden]), dtype="float16")
    w = g.add_input("rmsnorm_weight", TensorShape([hidden]), dtype="float16")

    # Use the built-in rms_norm convenience method
    normed = g.rms_norm(x, w, "rmsnorm")
    g.add_output(normed, "normalized")

    return g


def analyze_molten_kernel(hidden=HIDDEN_SIZE):
    """Generate and analyze Molten's fused kernel for RMSNorm."""
    print("\n" + "=" * 80)
    print("MOLTEN KERNEL ANALYSIS -- Qwen2.5-7B RMSNorm")
    print("=" * 80)

    g = build_rmsnorm_graph(hidden)
    print(f"\nDataflowGraph:")
    print(g)

    # Fusion analysis
    engine = FusionEngine()
    groups = engine.fuse(g)
    report = engine.report(g, groups)
    print(f"\n{report}")

    # Code generation (fp16 I/O, targeting Blackwell SM100)
    kernels = compile_graph(g, compute_capability=100, fp16=True)

    print(f"\nGenerated {len(kernels)} kernel(s):\n")
    for i, k in enumerate(kernels):
        print(f"--- Kernel {i}: {k.name} ---")
        print(f"Config: block={k.config.block_size_x}x{k.config.block_size_y}, "
              f"shmem={k.config.shared_mem_bytes}B, "
              f"vectorize={k.config.vectorize}, "
              f"args={k.num_args}")
        print(f"\nGenerated CUDA source:")
        print("-" * 60)
        print(k.source)
        print("-" * 60)

    # Also generate fp32 variant for comparison
    kernels_fp32 = compile_graph(g, compute_capability=100, fp16=False)

    ops_fused = [
        g.ops[op_id].op_type.name
        for group in groups
        for op_id in group.op_ids
    ]

    return {
        "graph_name": g.name,
        "total_ops": len(g.ops),
        "fusion_groups": len(groups),
        "fusion_ratio": len(g.ops) / max(len(groups), 1),
        "kernels": [
            {
                "name": k.name,
                "config": asdict(k.config),
                "num_args": k.num_args,
                "source_lines": len(k.source.split("\n")),
                "fp16": True,
            }
            for k in kernels
        ],
        "kernels_fp32": [
            {
                "name": k.name,
                "source_lines": len(k.source.split("\n")),
                "fp16": False,
            }
            for k in kernels_fp32
        ],
        "ops_fused": ops_fused,
        "kernel_source_fp16": kernels[0].source if kernels else "",
        "kernel_source_fp32": kernels_fp32[0].source if kernels_fp32 else "",
    }


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def print_summary(bench_results, molten_info):
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    # Gather eager vs compile comparison at seq=2048
    eager_times = {}
    compile_times = {}
    for r in bench_results:
        if "2048" in r.shape:
            if r.method == "eager":
                eager_times[r.layer_name] = r.time_us
            elif r.method == "torch.compile":
                compile_times[r.layer_name] = r.time_us

    if eager_times:
        avg_eager = sum(eager_times.values()) / len(eager_times)
        avg_compile = sum(compile_times.values()) / len(compile_times)
        avg_speedup = avg_eager / avg_compile if avg_compile > 0 else 0

        print(f"\nAt seq_len=2048, hidden={HIDDEN_SIZE}:")
        print(f"  Avg eager RMSNorm:     {avg_eager:.1f} us")
        print(f"  Avg compiled RMSNorm:  {avg_compile:.1f} us")
        print(f"  torch.compile speedup: {avg_speedup:.2f}x")

    print(f"\nMolten analysis:")
    print(f"  Graph ops:      {molten_info['total_ops']}")
    print(f"  Fusion groups:  {molten_info['fusion_groups']} "
          f"(fusion ratio: {molten_info['fusion_ratio']:.1f}x)")
    print(f"  Ops fused:      {' -> '.join(molten_info['ops_fused'])}")
    print(f"  Kernels generated: {len(molten_info['kernels'])} "
          f"(fp16) + {len(molten_info['kernels_fp32'])} (fp32)")

    for k in molten_info["kernels"]:
        print(f"  Kernel '{k['name']}': {k['source_lines']} lines, "
              f"block={k['config']['block_size_x']}, "
              f"shmem={k['config']['shared_mem_bytes']}B")

    print(f"\nOn Qwen2.5-7B's actual RMSNorm with hidden={HIDDEN_SIZE}, "
          f"Molten generates a fused kernel that collapses "
          f"{len(molten_info['ops_fused'])} ops "
          f"({' -> '.join(molten_info['ops_fused'])}) into "
          f"{molten_info['fusion_groups']} GPU kernel(s), "
          f"eliminating intermediate global memory round-trips.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    if not torch.cuda.is_available():
        print("CUDA not available.")
        return

    device_name = torch.cuda.get_device_name(1)
    print(f"Device: {device_name}")
    props = torch.cuda.get_device_properties(1)
    print(f"VRAM:   {props.total_mem / 1e9:.1f} GB")
    print(f"Compute capability: {props.major}.{props.minor}")
    print(f"Model:  {MODEL_ID}")
    print(f"Hidden: {HIDDEN_SIZE}")

    # Phase 1: Load model and benchmark real RMSNorm layers
    model = load_model()
    bench_results = bench_model_rmsnorm(model, seq_lens=(1, 128, 512, 2048))

    # Free model memory before Molten analysis
    del model
    torch.cuda.empty_cache()

    # Phase 2: Molten kernel generation and analysis
    molten_info = analyze_molten_kernel(HIDDEN_SIZE)

    # Phase 3: Summary
    print_summary(bench_results, molten_info)

    # Save results
    output = {
        "device": device_name,
        "model": MODEL_ID,
        "hidden_size": HIDDEN_SIZE,
        "num_layers": NUM_LAYERS,
        "benchmark_results": [asdict(r) for r in bench_results],
        "molten_analysis": molten_info,
    }
    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULT_PATH, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {RESULT_PATH}")


if __name__ == "__main__":
    main()
