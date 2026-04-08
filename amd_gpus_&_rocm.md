# AMD GPUs & ROCm -- Quick Review

---

## AMD Instinct GPU Lineup

| | MI300X | MI325X | MI355X |
|---|--------|--------|--------|
| **Architecture** | CDNA3 | CDNA3 | CDNA4 (new) |
| **HBM** | 192 GB HBM3 | 256 GB HBM3e | 288 GB HBM3e |
| **Bandwidth** | 5.3 TB/s | 6.0 TB/s | 6.0+ TB/s |
| **FP16 TFLOPS** | 1,307 | 1,307 | ~2,300 (est.) |
| **FP8 TFLOPS** | 2,615 | 2,615 | ~4,600 (est.) |
| **TDP** | 750W | 750W | ~750W |
| **Interconnect** | Infinity Fabric | Infinity Fabric | Infinity Fabric |

**MI355X** is the one everyone will be talking about. CDNA4 is the generational jump. Native FP8 support is a key selling point.

---

## Where AMD Wins vs. NVIDIA

1. **Memory capacity** -- 288 GB on MI355X is unmatched. Fit larger models without sharding across GPUs. An H100 has only 80 GB.
2. **Price-performance** -- 30-50% cheaper per GPU-hour on cloud providers like TensorWave.
3. **Power efficiency** -- MI355X at ~750W delivers similar FP8 TFLOPS to NVIDIA B200 at 1,000W. That is 25% less power for comparable compute.
4. **Open-source stack** -- ROCm is fully open. No licensing, no vendor lock-in at the software level.

## Where AMD Loses vs. NVIDIA

1. **Software ecosystem** -- CUDA has 15+ years of libraries, tools, tutorials, and muscle memory. ROCm is catching up but not there.
2. **Interconnect at scale** -- NVLink 5 does 1.8 TB/s per GPU. Infinity Fabric has lower bandwidth. Multi-node training at 256+ GPUs is where this gap hurts.
   - **The problem:** Distributed training requires all GPUs to exchange gradients (all-reduce operation). NVIDIA's NVSwitch intelligently routes this traffic across nodes at near-NVLink speeds. AMD lacks an equivalent switch. Cross-node communication falls back to RoCE (~50-200 Gbps), which is 10x slower than NVLink.
   - **Real impact:** All-reduce time jumps from ~100ms (NVIDIA) to ~500ms (AMD) at 256+ GPUs. Training iteration time increases ~10-20%.
   - **AMD's counter:** Larger HBM (288 GB) lets you fit bigger batches per GPU, reducing communication frequency. Single-node training (8 GPUs) is competitive. Inference (where AMD wins) has no all-reduce overhead.
   - **Bottom line:** Training at 256+ GPU scale favors NVIDIA. Training at single-node or inference favors AMD.

### Interconnect Bandwidth Comparison

| Link Type | Per-Link BW | Use Case |
|-----------|-----------|----------|
| **NVLink 5** (NVIDIA GPU-to-GPU) | 1.8 TB/s | Intra-node and inter-node (via NVSwitch fabric) |
| **Infinity Fabric** (AMD GPU-to-GPU) | 0.4-1.0 TB/s | Intra-node only |
| **RoCE** (Ethernet-based inter-node) | 50-400 Gbps | Cross-node fallback (when no high-speed switch) |
| **Standard Ethernet** (with CPU involvement) | 10-100 Gbps | General network, slow |

**Key insight:** NVSwitch lets NVIDIA route all-reduce traffic efficiently across both intra-node (NVLink) and inter-node paths. AMD has no equivalent switch, so cross-node communication falls back to RoCE. This is the main bottleneck for AMD at 256+ GPU scales.
3. **Compiler/profiler maturity** -- Nsight Compute is polished. Omniperf/rocprof are functional but rougher.
4. **Custom kernel support** -- Flash-attention, fused operators, and custom CUDA kernels often need manual porting to HIP/AMD.

---

## ROCm -- AMD's Software Stack

ROCm is AMD's answer to the CUDA toolkit. It includes:

- **HIP** -- GPU programming language. Syntactically almost identical to CUDA. The `hipify` tool auto-converts ~90% of CUDA code to HIP.
- **Compilers** -- `hipcc` (HIP compiler), plus Triton has an AMD backend.
- **Libraries** -- rocBLAS (matrix math), MIOpen (deep learning primitives), RCCL (multi-GPU comms), rocFFT, rocSPARSE.
- **Profiling** -- Omniperf, rocprof. Functional but less polished than NVIDIA Nsight.
- **Containers/drivers** -- ROCm ships Docker containers with pre-configured environments.

### Library Mapping: NVIDIA → AMD

| NVIDIA | AMD | Maturity |
|--------|-----|----------|
| cuBLAS | rocBLAS | Mature, competitive |
| cuDNN | MIOpen | Functional, some gaps (flash-attention) |
| NCCL | RCCL | Works, less optimized at scale |
| cuFFT | rocFFT | Mature |
| TensorRT | MIGraphX | Less adopted |
| Thrust | rocThrust | Drop-in replacement |
| cuSPARSE | rocSPARSE | Mature |
| Nsight Compute | Omniperf / rocprof | Functional, rougher UX |

### The Real Pain Points

- **Flash-attention on AMD** -- not all flash-attention variants are natively supported. Some rely on community ports. Performance gap exists. Key question for AMD speakers.
- **RCCL at scale** -- works for small clusters, but NCCL has years of optimization at 256+ GPU scale. Distributed training is the weak spot.
- **Custom CUDA kernels** -- any model with hand-written CUDA (quantization kernels, fused ops) breaks on AMD until ported. This is the #1 friction point.
- **Driver stability** -- ROCm across driver versions in production has been a source of complaints. Ask about this.

---

## Portability Paths: Getting From CUDA to AMD

| Approach | How | Effort | Performance | Who |
|----------|-----|--------|-------------|-----|
| **HIP / hipify** | Translate CUDA source to HIP | Medium (90% auto, 10% manual) | Best (native) | AMD |
| **ZLUDA** | Run CUDA binaries on AMD unmodified | Zero | Unknown ceiling | Andrzej Janik |
| **SCALE** | Recompile CUDA source for AMD at build time | Low (no code changes) | Good | Spectral Compute |
| **Triton** | Write kernels in Triton, compile to both | Medium (rewrite kernels) | Good | OpenAI / community |
| **Modular / Mojo** | New language, targets both | High (rewrite) | TBD | Modular |
| **Framework-level** | Just use PyTorch/JAX, let the backend handle it | Zero | Good for standard ops | Works today |

**The realistic path for most teams:** Use PyTorch on ROCm (works today), deal with custom kernel gaps on a case-by-case basis, use Triton for new kernel work.

---

## Hardware Details That Matter

**Wavefronts vs. Warps** -- AMD wavefronts are 64-wide; NVIDIA warps are 32-wide. Kernels optimized for NVIDIA warp-level primitives (`__shfl_sync`, warp reductions) need non-trivial changes for AMD. This is a real porting issue.

**Memory hierarchy:**
- Registers → LDS (AMD's shared memory, equivalent to CUDA `__shared__`) → L1/L2 cache → HBM (global)
- AMD calls shared memory "LDS" (Local Data Share). Same concept, different name.
- LDS is ~100x faster than global HBM. Performance-critical kernels must use it.

**Occupancy** -- How many threads actively use the GPU's execution units. Depends on registers/thread, shared memory/block, threads/block. AMD GPUs have different register file sizes, so CUDA kernels ported to HIP may need occupancy re-tuning.

---

## Key Numbers to Have in Your Head

- MI355X: **288 GB HBM**, **6 TB/s bandwidth**, **~4,600 FP8 TFLOPS**, **750W**
- H100: 80 GB, 3.35 TB/s, 1,979 FP8, 700W
- B200: 192 GB, 8.0 TB/s, ~4,500 FP8, 1,000W
- AMD HBM advantage: MI355X has **3.6x** the memory of H100, **1.5x** of B200
- AMD power advantage: ~25% less power than B200 for comparable FP8 compute
- AMD bandwidth: MI355X 6 TB/s sits between H100 (3.35) and B200 (8.0). Not the fastest, but paired with more memory.

---

## Questions to Ask AMD Speakers (Anush Elangovan, Neha Prakriya)

- What is next for ROCm? Which CUDA gaps are closing in 2026?
- How production-ready is FP8 on MI355X for real training and inference?
- Is AMD investing in Triton or building a proprietary compiler?
- Flash-attention on MI355X -- native or community port? Benchmarks vs. NVIDIA?
- How does RCCL perform at 256+ GPU scale vs. NCCL?
- Does AMD's own chip design team use Instinct GPUs with ROCm? (Dogfooding question)
- When AMD trains models internally, MI300X clusters or still on NVIDIA? (For Neha)
