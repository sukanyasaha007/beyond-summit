# LLM Training Stack & Distributed Training -- Quick Review

---

## Why Training Matters at This Summit

While **inference** is AMD's strongest story, **training** is where the gaps are clearest. Distributed training at scale (256+ GPUs) depends on interconnect bandwidth, and NVIDIA's NVLink still dominates AMD's Infinity Fabric here. Understanding these gaps is important for asking good questions.

Most speakers focus on inference and serving. But Zyphra trained Zamba models, Liquid AI does research training, and TensorWave supports training workloads. Worth asking about their experience.

---

## Training vs. Inference

| | Training | Inference |
|---|----------|-----------|
| **Compute bound or memory bound?** | Compute-heavy (matrix multiplies during backprop) | Memory-bound (reading weights repeatedly) |
| **GPU bottleneck** | TFLOPS (compute throughput) | Bandwidth (memory read speed) |
| **Batch size** | Large (256-1024 typical) | Small (1-64 for latency-sensitive use) |
| **AMD advantage** | Smaller (TFLOPS comparable) | Larger (more HBM, good bandwidth) |
| **NVIDIA advantage** | Larger (higher TFLOPS, NVLink at scale) | Smaller (NVLink helps multi-GPU inference) |
| **Duration** | Hours to weeks (requires stability) | Milliseconds to seconds (throughput matters) |

**Key insight:** AMD's case for training is weaker than for inference. But NVIDIA's compute advantage is narrowing -- B200 is close to MI355X in raw TFLOPS.

---

## Parallelism Strategies for Large Models

When a model does not fit on one GPU, split it across multiple GPUs:

### Data Parallelism (DP)

Each GPU gets a different batch of data. All GPUs have identical model parameters (replicated).

```
GPU 0: forward/backward on batch[0:100]    \
GPU 1: forward/backward on batch[100:200]   } All-reduce gradients, update shared params
GPU 2: forward/backward on batch[200:300]   /
GPU 3: forward/backward on batch[300:400]  /
```

- **Pros:** Simple, works well on regular hardware, minimal inter-GPU communication
- **Cons:** Model must fit in one GPU's memory. Redundant model copies waste memory.
- **When to use:** Small-to-medium models or clusters where interconnect is limited

### Tensor Parallelism (TP)

Split the model itself across GPUs. Every layer computation is distributed.

```
GPU 0: forward/backward on layer_1 [part A], layer_2 [part A], ...
GPU 1: forward/backward on layer_1 [part B], layer_2 [part B], ...
(synchronized between parts via AllGather)
```

- **Pros:** Fit larger models in total memory (model sharded across GPUs)
- **Cons:** Requires **fast inter-GPU communication** (NVLink / Infinity Fabric). Communication overhead grows with model width.
- **When to use:** Very large models (70B+) where model parallelism is necessary

### Pipeline Parallelism (PP)

Split the model into stages (layers). Each GPU handles a pipeline stage. Requests flow through stages.

```
GPU 0: layers 0-8      (stage A)
GPU 1: layers 9-16     (stage B)
GPU 2: layers 17-24    (stage C)
Request flows through stages sequentially.
```

- **Pros:** Works across slower networks (no bandwidth bottleneck per stage)
- **Cons:** Pipeline bubbles (idle GPUs waiting for previous stage to finish). Lower GPU utilization.
- **When to use:** Cross-node training, slow interconnects, or extreme model sizes

### Sequence Parallelism (SP) / Context Parallelism

Split the input sequence across GPUs. Advanced technique for long-context training.

- Used by newer models (Llama 3.1, etc.) when sequence length is massive
- Reduces per-GPU memory usage for KV cache during training
- Less common today; growing in importance

### Common Patterns in Practice

Most production training uses **combinations:**

- **DP + TP:** Data parallelism across nodes, tensor parallelism within nodes (intra-node communication is faster)
- **DP + PP:** Data parallelism for gradients, pipeline parallelism for huge models
- **DP + TP + PP:** Full stack for extreme scales (e.g., training 500B+ models on 1000+ GPUs)

---

## Key Collective Operations

All parallelism strategies need to synchronize gradients and activations across GPUs:

| Operation | What It Does | Which Layer It's In |
|-----------|-------------|---------------------|
| **AllReduce** | Sum gradients across all GPUs, send result back to all GPUs | NCCL (NVIDIA) / RCCL (AMD) |
| **AllGather** | Collect activations from all GPUs to all GPUs (for TP) | NCCL / RCCL |
| **ReduceScatter** | Inverse of AllGather | NCCL / RCCL |
| **Broadcast** | One GPU sends to all others | NCCL / RCCL |
| **P2P** | Point-to-point transfers | NCCL / RCCL |

**Why this matters:** NCCL (NVIDIA) has 10+ years of optimization at scale. RCCL (AMD) is younger. At 256+ GPU scales, NCCL's optimization advantage shows up in per-iteration time.

---

## Interconnect Bandwidth (Critical for TP and AllReduce at Scale)

| Interconnect | Per-GPU BW | Max Per Link | Use Case |
|--------------|-----------|--------------|----------|
| **NVLink 5** (NVIDIA) | 1.8 TB/s | -- | H100, B200. High-end TP, multi-GPU inference |
| **NVSwitch** (NVIDIA) | Extends NVLink across nodes | Switch-based | 256+ GPU clusters. All-to-all connectivity |
| **Infinity Fabric** (AMD) | Lower BW (0.4-1.0 TB/s est.) | -- | MI300X, MI355X. Slower for TP. |
| **RoCE** (generic) | Depends on network | 10-400 Gbps | Cross-cloud, CPU-GPU. When not using direct interconnect |
| **PCIe 5** (generic) | ~15 GB/s per GPU | -- | Slow fallback. Not for production training |

**Critical insight:** TP requires **fast, low-latency inter-GPU channels**. NVLink is 100x+ faster than RoCE. NVIDIA's NVSwitch at scale is AMD's major gap. This is why:
- Multi-node training is NVIDIA's strength
- Single-node (8 GPU) training can be competitive on AMD
- Data parallelism on AMD (slower inter-GPU links OK) is more viable than TP at scale

---

## Frameworks & Tools for Distributed Training

| Framework | NVIDIA Support | AMD Support | Notes |
|-----------|---|---|---------|
| **PyTorch FSDP** | Excellent | Good (via ROCm) | Uses ZeRO-like gradient sharding. Good for DP + TP combos. |
| **Hugging Face Transformers** | Excellent | Good (via ROCm) | High-level training APIs. Most people use this. |
| **DeepSpeed** | Excellent | Partial (experimental) | Microsoft's distributed training engine. ZeRO optimizations. Less mature on AMD. |
| **Megatron-LM** | Excellent | Partial | NVIDIA's TP/PP/DP framework. Reference implementation. Community AMD ports exist. |
| **JAX** | Good | Experimental | Functional approach to distributed training. Cleaner but smaller ecosystem. |
| **vLLM** (inference) | N/A | Excellent on AMD | Not for training, but relevant to inference workloads at the event |

**Practical recommendation:** If training on AMD, use **PyTorch FSDP** or **HuggingFace Transformers** (both have solid ROCm support). DeepSpeed on AMD is getting better but less battle-tested.

---

## Memory Management During Training: ZeRO

**ZeRO** (Zero Redundancy Optimizer) is Microsoft DeepSpeed's technique to reduce GPU memory during training. It has 3 stages:

| Stage | What It Does | Memory Saved |
|-------|-------------|--------------|
| **ZeRO-1** | Shard optimizer states across GPUs | ~4x (for Adam optimizer with moment buffers) |
| **ZeRO-2** | Shard gradients as well | ~8x |
| **ZeRO-3** | Shard model parameters too | ~N x (scales with # GPUs) |

ZeRO-3 lets you fit models that do not fit on a single node, even without TP. Trade-off: slower computation due to frequent communication.

**On AMD:** FSDP (PyTorch's equivalent) is available. DeepSpeed support is lagging.

---

## Quantization During Training

Unlike inference, training with quantization is harder (numerical stability). Common approaches:

| Format | Use In Training? | Notes |
|--------|--|---------|
| **FP16** | Yes, standard | Slight numerical issues, but works. Widely used. |
| **FP8** | Yes, newer | MI355X native FP8. Requires loss scaling (extra computation). Getting production-ready. |
| **QAT** (quantization-aware training) | Yes, for distillation | Train with fake quantization, then quantize post-training. |
| **Bfloat16** | Yes, alternative to FP16 | Better numerical stability. NVIDIA and AMD both support it. |
| **4-bit / 8-bit weight-gradient** | Experimental | Memory gains, slower compute. Research stage. |

**MI355X + FP8 training** is a key AMD selling point. Ask speakers (Anush Elangovan, Neha Prakriya) how stable FP8 training is in practice on their hardware.

---

## Data Loading & I/O During Training

Often overlooked, but I/O is a real bottleneck:

- **Problem:** Reading training data from disk/network is slow. While GPU is loading data, compute stops.
- **Solution:** Pre-fetch, multi-worker DataLoaders, cache in memory
- **On AMD:** PyTorch DataLoader works the same. No AMD-specific consideration. The GPU interconnect bottleneck (for multi-GPU) is more pressing than I/O.

---

## Training Stability & Driver Issues

An overlooked factor at AMD events:

| Issue | Impact | Status on AMD |
|-------|--------|---------|
| **Driver crashes** | Training stops mid-epoch, loses progress | Has been a problem. AMD ROCm driver stability improving but still a concern |
| **Memory leaks** | Memory usage grows over time, OOMs mid-training | Rare, but has happened with some ROCm versions |
| **Numerical reproducibility** | Same seed should give same results (for debugging) | Harder to guarantee on AMD across driver versions |
| **Interconnect stability** | AllReduce hangs, silent data corruption | RCCL less tested than NCCL at 256+ GPU scale |

**Worth asking TensorWave:** What driver versions are stable for multi-day training runs? Are there known gotchas with RCCL at scale?

---

## Practical Example: Training a 70B Model

**On 8x NVIDIA H100 (single node):**
- TP=2, DP=4 (2-way tensor parallelism, 4-way data parallelism)
- NVLink handles TP communication (fast)
- NCCL AllReduce for DP gradients
- Training stable, ~160 tokens/sec

**On 8x AMD MI300X (single node):**
- TP=2, DP=4 (same strategy)
- Infinity Fabric handles TP communication (slower than NVLink)
- RCCL AllReduce for DP gradients
- Slower due to TP communication bottleneck, but should work
- Estimated ~120-140 tokens/sec (real data varies)

**On 16x GPUs (2 nodes, multi-node):**
- NVIDIA: TP=2 (within node), DP=4, PP=2 (across nodes). NVLink + NVSwitch. Scales well.
- AMD: TP=2 (within node), DP=4, PP=2. Infinity Fabric within node, RoCE/Slingshot across nodes. More complex tuning needed.

---

## Questions to Ask at the Summit (Training Focus)

- To Zyphra (Quentin Anthony): How was the experience training Zamba on AMD? What broke? Which framework worked best?
- To AMD (Neha Prakriya): How does RCCL perform at 256+ GPU scale vs. NCCL? Are there tuning knobs?
- To AMD: Is AMD investing in DeepSpeed support on ROCm, or focusing on PyTorch FSDP?
- To TensorWave: Do you offer multi-node training, or just single-node / inference clusters?
- To anyone: "What is the largest training run (by # GPUs) you have done on AMD, and how does it compare to NVIDIA at that scale?"
- To Liquid AI: How does your distributed training infrastructure handle AMD GPUs?

---

## Key Takeaway

For **large-scale distributed training**, NVIDIA still has an edge (NVLink, NVSwitch, NCCL maturity).

For **single-node or small-cluster training** (8-64 GPUs), AMD is competitive. PyTorch FSDP or HuggingFace Transformers both work well on ROCm.

The training story at this event is about **closing the gap**, not beating NVIDIA. Focus on what AMD excels at: **inference**, **embedding generation**, **fine-tuning on smaller datasets**, and **cost**.
