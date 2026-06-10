"""
Runtime — JIT compiles and executes generated CUDA kernels.

Closes the loop: DataflowGraph → fused CUDA source → compiled kernel → callable.
Uses torch.utils.cpp_extension.load_inline for JIT compilation.

This is where Molten goes from "generates code" to "runs faster."
"""

from __future__ import annotations

import torch
import hashlib
import os
import tempfile
from pathlib import Path
from typing import Optional, Callable
from dataclasses import dataclass

from molten.codegen import GeneratedKernel, KernelConfig


@dataclass
class CompiledKernel:
    """A compiled, callable CUDA kernel."""
    name: str
    source: str
    config: KernelConfig
    _module: object = None

    def __call__(self, *args, **kwargs):
        """Launch the kernel with given tensor arguments."""
        if self._module is None:
            raise RuntimeError(f"Kernel '{self.name}' not compiled. Call compile() first.")
        fn = getattr(self._module, self.name)
        return fn(*args, **kwargs)


class MoltenRuntime:
    """
    JIT compiler and executor for Molten-generated kernels.

    Compiles CUDA source to callable functions via torch's cpp_extension.
    Caches compiled modules by source hash to avoid recompilation.
    """

    def __init__(self, cache_dir: Optional[str] = None, verbose: bool = False):
        self.verbose = verbose
        self.cache_dir = cache_dir or os.path.join(
            tempfile.gettempdir(), "molten_cache"
        )
        os.makedirs(self.cache_dir, exist_ok=True)
        self._compiled: dict[str, CompiledKernel] = {}

    def compile(self, kernel: GeneratedKernel) -> CompiledKernel:
        """Compile a generated kernel into a callable."""
        source_hash = hashlib.md5(kernel.source.encode()).hexdigest()[:12]

        if source_hash in self._compiled:
            return self._compiled[source_hash]

        # Write source files to disk, then use torch.utils.cpp_extension.load
        cpp_source, cuda_source = self._make_extension(kernel)
        build_dir = os.path.join(self.cache_dir, source_hash)
        os.makedirs(build_dir, exist_ok=True)

        cu_path = os.path.join(build_dir, f"{kernel.name}.cu")
        with open(cu_path, "w") as f:
            f.write(cuda_source)

        try:
            from torch.utils.cpp_extension import load

            module = load(
                name=f"molten_{kernel.name}_{source_hash}",
                sources=[cu_path],
                build_directory=build_dir,
                verbose=self.verbose,
            )

            compiled = CompiledKernel(
                name=f"{kernel.name}_launch",
                source=kernel.source,
                config=kernel.config,
                _module=module,
            )
            self._compiled[source_hash] = compiled
            return compiled

        except Exception as e:
            if self.verbose:
                print(f"Compilation failed for {kernel.name}: {e}")
                print(f"Source file: {cu_path}")
            raise

    def _make_extension(self, kernel: GeneratedKernel) -> tuple[str, str]:
        """Generate torch cpp_extension compatible source."""
        name = kernel.name
        cfg = kernel.config

        # CUDA source: the generated kernel + a launch wrapper
        cuda_source = f"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

{kernel.source}

torch::Tensor {name}_launch({"torch::Tensor input" if not kernel.config.block_size_y > 1 else "torch::Tensor A, torch::Tensor B"}) {{
"""
        if cfg.block_size_y > 1:
            # Matmul kernel
            cuda_source += f"""
    const int M = A.size(0);
    const int K = A.size(1);
    const int N = B.size(1);

    auto output = torch::zeros({{M, N}}, A.options());

    dim3 block({cfg.block_size_x}, {cfg.block_size_y});
    dim3 grid((N + block.x - 1) / block.x, (M + block.y - 1) / block.y);

    {name}<<<grid, block>>>(
        A.data_ptr<float>(), B.data_ptr<float>(),
        output.data_ptr<float>(), M, N, K
    );
    return output;
}}
"""
        elif kernel.config.shared_mem_bytes > 0:
            # Reduction kernel
            cuda_source += f"""
    const int cols = input.size(-1);
    const int rows = input.numel() / cols;

    auto output = torch::zeros_like(input);

    dim3 block({cfg.block_size_x});
    dim3 grid(rows);
    int smem = {cfg.shared_mem_bytes};

    {name}<<<grid, block, smem>>>(
        input.data_ptr<float>(), output.data_ptr<float>(),
        rows, cols
    );
    return output;
}}
"""
        else:
            # Elementwise kernel — detect number of inputs from kernel source
            num_inputs = kernel.source.count("__restrict__ in")
            if num_inputs >= 2:
                # Two-input elementwise (e.g., gelu_add, silu_gate)
                cuda_source = f"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

{kernel.source}

torch::Tensor {name}_launch(torch::Tensor in0, torch::Tensor in1) {{
    const int N = in0.numel();
    auto output = torch::zeros_like(in0);

    dim3 block({cfg.block_size_x});
    dim3 grid((N + block.x - 1) / block.x);

    {name}<<<grid, block>>>(
        in0.data_ptr<float>(), in1.data_ptr<float>(),
        output.data_ptr<float>(), N
    );
    return output;
}}
"""
            else:
                # Single-input elementwise
                cuda_source += f"""
    const int N = input.numel();
    auto output = torch::zeros_like(input);

    dim3 block({cfg.block_size_x});
    dim3 grid((N + block.x - 1) / block.x);

    {name}<<<grid, block>>>(
        input.data_ptr<float>(), output.data_ptr<float>(), N
    );
    return output;
}}
"""

        # Add pybind module to CUDA source so it's a standalone loadable .cu
        cuda_source += f"""
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {{
    m.def("{name}_launch", &{name}_launch, "Molten generated kernel");
}}
"""

        cpp_source = ""  # not needed when everything is in the .cu file
        return cpp_source, cuda_source

    def clear_cache(self):
        """Clear the compilation cache."""
        import shutil
        if os.path.exists(self.cache_dir):
            shutil.rmtree(self.cache_dir)
            os.makedirs(self.cache_dir, exist_ok=True)
        self._compiled.clear()
