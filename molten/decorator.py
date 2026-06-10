"""
@zero decorator — the user-facing API.

    @zero
    def my_layer(x, w):
        return silu(x @ w.T)

    # First call: traces + compiles + caches
    # Subsequent calls: runs cached kernel
    output = my_layer(x, w)
"""

from __future__ import annotations

import torch
import functools
from dataclasses import dataclass
from typing import Optional, Callable

from molten.compiler import ZeroCompiler
from molten.codegen import GeneratedKernel


@dataclass
class ZeroConfig:
    compute_capability: int = 80
    optimize: bool = True
    cache_kernels: bool = True
    verbose: bool = False
    fallback_to_torch: bool = True  # if compilation fails, run in PyTorch


class CompiledFunction:
    """Wrapper around a compiled function with kernel cache."""

    def __init__(self, fn: Callable, config: ZeroConfig):
        self.fn = fn
        self.config = config
        self.compiler = ZeroCompiler(
            compute_capability=config.compute_capability,
            optimize=config.optimize,
            verbose=config.verbose,
        )
        self._compiled = False
        self._kernels: list[GeneratedKernel] = []
        self._graph = None

    def __call__(self, *args, **kwargs):
        if not self._compiled:
            self._compile_from_args(args, kwargs)

        # v0.1: @zero traces and generates .cu files but executes via PyTorch.
        # The generated kernels can be compiled and benchmarked separately
        # via MoltenRuntime. Direct dispatch from @zero is planned for v0.2.
        return self.fn(*args, **kwargs)

    def _compile_from_args(self, args, kwargs):
        """Trace the function with real arguments and compile."""
        # Build example inputs dict from args
        import inspect
        sig = inspect.signature(self.fn)
        param_names = list(sig.parameters.keys())

        example_inputs = {}
        for i, arg in enumerate(args):
            if i < len(param_names) and isinstance(arg, torch.Tensor):
                example_inputs[param_names[i]] = arg
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                example_inputs[k] = v

        try:
            self._graph = self.compiler.trace(self.fn, example_inputs)
            self._kernels = self.compiler.compile(self._graph)
            self._compiled = True

            if self.config.verbose:
                print(f"Molten: compiled {len(self._kernels)} kernel(s) "
                      f"for '{self.fn.__name__}'")

        except Exception as e:
            if self.config.verbose:
                print(f"Molten: compilation failed for '{self.fn.__name__}': {e}")
            self._compiled = False

    @property
    def kernels(self) -> list[GeneratedKernel]:
        return self._kernels

    @property
    def graph(self):
        return self._graph

    def save_kernels(self, output_dir: str):
        """Save compiled kernels to disk."""
        if self._kernels:
            self.compiler.save(self._kernels, output_dir)


def zero(fn=None, *, config: Optional[ZeroConfig] = None):
    """
    Decorator that compiles a function into fused CUDA kernels.

    Usage:
        @zero
        def my_fn(x, w):
            return x @ w.T

        # Or with config:
        @zero(config=ZeroConfig(verbose=True))
        def my_fn(x, w):
            return x @ w.T
    """
    if config is None:
        config = ZeroConfig()

    if fn is not None:
        # @zero without arguments
        compiled = CompiledFunction(fn, config)
        functools.update_wrapper(compiled, fn)
        return compiled

    # @zero(config=...) with arguments
    def decorator(fn):
        compiled = CompiledFunction(fn, config)
        functools.update_wrapper(compiled, fn)
        return compiled

    return decorator
