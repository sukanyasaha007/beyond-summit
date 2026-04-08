# NVIDIA GPUs & CUDA -- Quick Review

---

## Why This Matters at an AMD Event

The summit is about moving "beyond CUDA." To follow the conversations, you need to know what CUDA is, why it created lock-in, and what each speaker is trying to replace.

---

## What CUDA Actually Is

CUDA is **three things bundled together** (this is why lock-in is sticky):

1. **Programming model** -- kernels, threads, blocks, grids. How you express parallelism on the GPU.
2. **Compiler toolchain** -- `nvcc` compiles `.cu` → PTX (intermediate representation) → SASS (machine code). Every layer is proprietary.
3. **Library ecosystem** -- cuBLAS, cuDNN, NCCL, TensorRT. 15+ years of optimized libraries that everything depends on.

When speakers say "CUDA compatibility," ask **which layer**:
- ZLUDA intercepts the **runtime API** (layer 1)
- SCALE recompiles at the **compiler** level (layer 2)
- ROCm/HIP replaces **all three** with AMD-native equivalents

---

## NVIDIA GPU Lineup

| | A100 | H100 | B200 |
|---|------|------|------|
| **Architecture** | Ampere | Hopper | Blackwell |
| **HBM** | 80 GB HBM2e | 80 GB HBM3 | 192 GB HBM3e |
| **Bandwidth** | 2.0 TB/s | 3.35 TB/s | 8.0 TB/s |
| **FP16 TFLOPS** | 312 | 989 | ~2,250 |
| **FP8 TFLOPS** | N/A | 1,979 | ~4,500 |
| **TDP** | 400W | 700W | 1,000W |
| **Interconnect** | NVLink 3 | NVLink 4 | NVLink 5 |

B200 is the current top-of-line. 192 GB HBM3e finally matches AMD MI300X on memory, but at 1,000W power draw.

---

## The GPU Execution Model

Think of it like a verification testbench:

- **Host (CPU)** = testbench controller. Orchestrates what runs and when.
- **Device (GPU)** = the DUT. Massively parallel, thousands of execution units.
- **Kernel** = function that runs on the GPU. Every thread executes the same code on different data (SIMT: Single Instruction, Multiple Threads).

### Thread Hierarchy

```
Grid (all blocks for one kernel launch)
  └── Block (threads sharing fast local memory)
        └── Warp (32 threads executing in lockstep)
              └── Thread (single execution unit)
```

**AMD difference:** AMD wavefronts are **64-wide** vs. NVIDIA warps at **32-wide**. Kernels using warp-level primitives (`__shfl_sync`, warp reductions) need adjustment for AMD.

---

## Memory Hierarchy

From fastest to slowest:

| Level | CUDA Name | AMD Name | Speed |
|-------|-----------|----------|-------|
| Registers | Registers | Registers | Fastest |
| Shared memory | `__shared__` | LDS (Local Data Share) | ~100x faster than HBM |
| L1/L2 cache | Automatic | Automatic | Hardware managed |
| Global memory (HBM) | `cudaMalloc` | `hipMalloc` | Slowest, but huge |

**Key insight:** LLM inference is **memory-bandwidth-bound**, not compute-bound. The GPU reads the entire model's weights from HBM for every token. The speed limit is memory read speed, not math. This is why AMD's larger HBM and high bandwidth are competitive.

---

## The Software Stack

```
Your Python code (PyTorch, JAX, vLLM)
       │
Framework layer (ATen, XLA, Triton kernels)
       │
Library layer (cuBLAS, cuDNN, NCCL)
       │
Runtime (CUDA Runtime API)
       │
Driver (nvidia.ko)
       │
Hardware (A100 / H100 / B200)
```

Most AI workloads never write raw CUDA. They go through PyTorch/JAX, which call cuBLAS/cuDNN underneath. The lock-in bites when you need custom fused kernels, multi-GPU comms (NCCL), or inference optimization (TensorRT).

---

## CUDA Libraries -- Where the Lock-In Lives

| Library | Purpose | Why It Locks You In |
|---------|---------|---------------------|
| **cuBLAS** | Matrix multiplication (GEMM) | Foundation of all training. Tuned per GPU generation. |
| **cuDNN** | Convolution, attention, batch norm | Flash-attention and fused ops rely on this. AMD's MIOpen has gaps. |
| **NCCL** | Multi-GPU collective comms | All-reduce at scale. 10+ years of optimization. AMD's RCCL is behind. |
| **TensorRT** | Inference graph optimization | Layer fusion, quantization, auto-tuning. AMD's MIGraphX less adopted. |
| **Triton** | Hardware-agnostic kernel compiler | The escape hatch -- write once, compile to NVIDIA or AMD. |

---

## Key CUDA Terms

| Term | What It Means |
|------|--------------|
| **PTX** | NVIDIA intermediate representation. Like GPU assembly. ZLUDA/SCALE must handle this. |
| **SASS** | Final GPU machine code. Not portable by design. |
| **nvcc** | NVIDIA's CUDA compiler. |
| **SM** (Streaming Multiprocessor) | NVIDIA's compute unit. H100 has 132 SMs. AMD equivalent: CU (Compute Unit). |
| **Warp** | 32 threads in lockstep. NVIDIA's hardware scheduling unit. |
| **Occupancy** | How well the GPU's execution units stay busy. Low = wasted silicon. |
| **Kernel fusion** | Combining multiple ops into one kernel to reduce memory traffic. |
| **SIMT** | All threads in a warp run the same instruction on different data. |
| **NVLink** | Proprietary GPU-to-GPU interconnect. 1.8 TB/s per GPU on NVLink 5. |
| **NVSwitch** | Extends NVLink across nodes. Enables all-to-all GPU communication at cluster scale. |

---

## Why NVIDIA Is Hard to Displace

1. **15+ years of ecosystem** -- every framework, tutorial, and StackOverflow answer assumes CUDA.
2. **Custom kernels everywhere** -- flash-attention, quantization ops, fused layers. Hand-tuned CUDA that breaks on AMD.
3. **NVLink at scale** -- 256+ GPU training depends on NVLink/NVSwitch. Infinity Fabric has lower bandwidth.
4. **Compiler maturity** -- `nvcc` and Nsight profiling are battle-tested. ROCm tools are rougher.
5. **Inertia** -- engineers know CUDA. Switching cost is real even when hardware is cheaper.

---

## "Beyond CUDA" Paths

| Path | Approach | Effort | Maturity |
|------|----------|--------|----------|
| **HIP / hipify** | Translate CUDA source to AMD | Medium (90% auto) | Production |
| **ZLUDA** | Run CUDA binaries on AMD unmodified | Zero | Experimental |
| **SCALE** | Recompile CUDA source for AMD at build time | Low | Emerging |
| **Triton** | Write hardware-agnostic kernels | Medium (rewrite) | Growing |
| **Modular / Mojo** | New language, targets both | High (rewrite) | Early |
| **Framework-level** | Use PyTorch/JAX, let backend handle it | Zero | Works for standard ops |

**Your angle:** You never write raw CUDA. Your stack (PyTorch, HuggingFace, vLLM) abstracts it. The question is: does that abstraction hold on AMD, or do you hit edge cases?

---

## Questions to Ask (NVIDIA/CUDA Context)

- To ZLUDA: Which CUDA layer does ZLUDA intercept? What breaks?
- To Spectral Compute: Does SCALE handle cuDNN and cuBLAS calls, or only raw kernels?
- To AMD: Is Triton the primary portability path, or is AMD pushing HIP?
- To Modular: Can Mojo replace hand-written CUDA kernels in practice today?
- To anyone: "What percentage of your CUDA codebase ported to AMD without issues, and what broke in the remaining 10%?"
