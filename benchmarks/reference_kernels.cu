// Hand-written fused CUDA kernels — the targets Molten must match.
// These are what a senior kernel engineer would write for Qwen3/Llama ops.

#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

// ============================================================
// Fused RMSNorm kernel
// One pass: compute variance, normalize, scale — all in registers
// ============================================================
__global__ void fused_rmsnorm_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    float* __restrict__ output,
    const int rows,
    const int cols,
    const float eps
) {
    extern __shared__ float sdata[];
    const int tid = threadIdx.x;
    const int row = blockIdx.x;
    if (row >= rows) return;

    const float* row_in = input + row * cols;
    float* row_out = output + row * cols;

    // Pass 1: sum of squares
    float ss = 0.0f;
    for (int i = tid; i < cols; i += blockDim.x) {
        float v = row_in[i];
        ss += v * v;
    }
    sdata[tid] = ss;
    __syncthreads();

    // Warp reduction
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }

    float rms_inv = rsqrtf(sdata[0] / (float)cols + eps);
    __syncthreads();

    // Pass 2: normalize and scale
    for (int i = tid; i < cols; i += blockDim.x) {
        row_out[i] = row_in[i] * rms_inv * weight[i];
    }
}

torch::Tensor fused_rmsnorm_cuda(torch::Tensor input, torch::Tensor weight, float eps) {
    const int cols = input.size(-1);
    const int rows = input.numel() / cols;
    auto output = torch::empty_like(input);

    // Use contiguous view — no reshape overhead
    auto flat_in = input.contiguous().view({rows, cols});
    auto flat_out = output.view({rows, cols});

    const int threads = 256;
    fused_rmsnorm_kernel<<<rows, threads, threads * sizeof(float)>>>(
        flat_in.data_ptr<float>(), weight.data_ptr<float>(),
        flat_out.data_ptr<float>(), rows, cols, eps
    );
    return output;
}

// ============================================================
// Fused SiLU * gate kernel
// silu(gate) * x in a single pass
// ============================================================
__global__ void fused_silu_gate_kernel(
    const float* __restrict__ x,
    const float* __restrict__ gate,
    float* __restrict__ output,
    const int N
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    float g = gate[idx];
    float sigmoid_g = 1.0f / (1.0f + expf(-g));
    output[idx] = x[idx] * g * sigmoid_g;
}

torch::Tensor fused_silu_gate_cuda(torch::Tensor x, torch::Tensor gate) {
    auto output = torch::empty_like(x);
    const int N = x.numel();
    const int threads = 256;
    const int blocks = (N + threads - 1) / threads;

    fused_silu_gate_kernel<<<blocks, threads>>>(
        x.data_ptr<float>(), gate.data_ptr<float>(),
        output.data_ptr<float>(), N
    );
    return output;
}

// ============================================================
// Fused RMSNorm + SiLU*gate — the triple fusion
// Three memory-bound ops become one kernel, one read of x
// ============================================================
__global__ void fused_rmsnorm_silu_gate_kernel(
    const float* __restrict__ x,
    const float* __restrict__ weight,
    const float* __restrict__ gate,
    float* __restrict__ output,
    const int rows,
    const int cols,
    const float eps
) {
    extern __shared__ float sdata[];
    const int tid = threadIdx.x;
    const int row = blockIdx.x;
    if (row >= rows) return;

    const float* row_x = x + row * cols;
    const float* row_gate = gate + row * cols;
    float* row_out = output + row * cols;

    // Pass 1: sum of squares for RMS
    float ss = 0.0f;
    for (int i = tid; i < cols; i += blockDim.x) {
        float v = row_x[i];
        ss += v * v;
    }
    sdata[tid] = ss;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    float rms_inv = rsqrtf(sdata[0] / (float)cols + eps);
    __syncthreads();

    // Pass 2: normalize, scale, silu_gate — all fused
    for (int i = tid; i < cols; i += blockDim.x) {
        float normed = row_x[i] * rms_inv * weight[i];
        float g = row_gate[i];
        float sigmoid_g = 1.0f / (1.0f + expf(-g));
        row_out[i] = normed * g * sigmoid_g;
    }
}

torch::Tensor fused_rmsnorm_silu_gate_cuda(
    torch::Tensor x, torch::Tensor weight, torch::Tensor gate, float eps
) {
    const int cols = x.size(-1);
    const int rows = x.numel() / cols;
    auto output = torch::empty_like(x);

    auto flat_x = x.contiguous().view({rows, cols});
    auto flat_gate = gate.contiguous().view({rows, cols});
    auto flat_out = output.view({rows, cols});

    const int threads = 256;
    fused_rmsnorm_silu_gate_kernel<<<rows, threads, threads * sizeof(float)>>>(
        flat_x.data_ptr<float>(), weight.data_ptr<float>(),
        flat_gate.data_ptr<float>(), flat_out.data_ptr<float>(),
        rows, cols, eps
    );
    return output;
}

// ============================================================
// Fused GELU + Add kernel
// ============================================================
__global__ void fused_gelu_add_kernel(
    const float* __restrict__ x,
    const float* __restrict__ bias,
    float* __restrict__ output,
    const int rows,
    const int cols
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = rows * cols;
    if (idx >= total) return;

    const int col = idx % cols;
    float v = x[idx];
    // GELU approximation (tanh version)
    float gelu = 0.5f * v * (1.0f + tanhf(0.7978845608f * (v + 0.044715f * v * v * v)));
    output[idx] = gelu + bias[col];
}

torch::Tensor fused_gelu_add_cuda(torch::Tensor x, torch::Tensor bias) {
    auto output = torch::empty_like(x);
    const int total = x.numel();
    const int cols = x.size(-1);
    const int rows = total / cols;
    const int threads = 256;
    const int blocks = (total + threads - 1) / threads;

    fused_gelu_add_kernel<<<blocks, threads>>>(
        x.data_ptr<float>(), bias.data_ptr<float>(),
        output.data_ptr<float>(), rows, cols
    );
    return output;
}

// ============================================================
// Bindings
// ============================================================
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_rmsnorm", &fused_rmsnorm_cuda, "Fused RMSNorm");
    m.def("fused_silu_gate", &fused_silu_gate_cuda, "Fused SiLU*gate");
    m.def("fused_rmsnorm_silu_gate", &fused_rmsnorm_silu_gate_cuda, "Fused RMSNorm+SiLU*gate");
    m.def("fused_gelu_add", &fused_gelu_add_cuda, "Fused GELU+Add");
}
