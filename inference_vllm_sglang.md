# LLM Inference: vLLM, SGLang & Serving -- Quick Review

---

## Why Inference Matters at This Summit

Inference (running a trained model to generate outputs) is AMD's strongest use case. LLM inference is **memory-bandwidth-bound**: the bottleneck is reading model weights from HBM, not computing. AMD's larger HBM (288 GB on MI355X) and high bandwidth (5.3-6.0 TB/s) make it genuinely competitive with NVIDIA here -- more so than for training.

TensorWave (the host) is primarily an inference cloud. Their custom engine **ScalarLM** competes with vLLM. Expect inference to be the dominant topic.

---

## The Two Open-Source Inference Engines

### vLLM

- The most widely deployed open-source LLM inference engine.
- **Key innovation: PagedAttention** -- manages GPU memory for the KV cache (key-value cache) the way an OS manages virtual memory pages. Eliminates memory fragmentation, allows serving more concurrent requests.
- **ROCm support:** Mature. vLLM on AMD is a real production path today.
- **Features:** Continuous batching, tensor parallelism, prefix caching, multiple quantization formats (GPTQ, AWQ, FP8, GGUF).
- **Who uses it:** Most AMD cloud providers, including TensorWave (alongside ScalarLM).

### SGLang

- Newer, gaining traction fast. Developed at UC Berkeley.
- **Key innovation: RadixAttention** -- caches and reuses KV cache across requests using a radix tree data structure. Especially good for workloads where many requests share a long system prompt or share prefixes.
- **ROCm support:** Works, but less battle-tested than vLLM on AMD.
- **Features:** Structured generation (constrained decoding -- force the model to output valid JSON, regex patterns, etc.), multi-turn optimization, faster for shared-prefix workloads.
- **Where it wins:** Multi-turn chat (reuses context), structured output (JSON APIs), workloads with common prompt prefixes.

### vLLM vs. SGLang -- When to Use Which

| Dimension | vLLM | SGLang |
|-----------|------|--------|
| **Maturity** | More mature, wider adoption | Newer, fast-moving |
| **AMD/ROCm** | Better tested on AMD | Works, less production mileage |
| **KV cache strategy** | PagedAttention (memory efficiency) | RadixAttention (prefix reuse) |
| **Best for** | General-purpose serving, diverse workloads | Shared-prefix, structured output, multi-turn |
| **Quantization** | GPTQ, AWQ, FP8, GGUF | GPTQ, AWQ, FP8 |
| **Structured output** | Basic support | First-class (constrained decoding) |
| **Community** | Larger | Growing fast |

### ScalarLM (TensorWave)

- TensorWave's proprietary inference engine. Built specifically for AMD GPUs.
- Not open-source. Competes with vLLM on TensorWave's own cloud.
- Ask Greg Diamos (ScalarLM architect): How does ScalarLM compare to vLLM in throughput and latency on MI300X?

---

## Key Inference Concepts

### KV Cache

During autoregressive generation, the model stores intermediate attention states (keys and values) for all previous tokens. This is the **KV cache**. It grows linearly with sequence length and batch size, and is the main memory consumer during inference.

- A 70B model at 4K context with batch size 32 can use 50+ GB of KV cache alone.
- This is why HBM capacity matters: AMD's 192-288 GB lets you serve longer contexts or more concurrent users before running out of memory.
- PagedAttention (vLLM) and RadixAttention (SGLang) are both strategies to manage KV cache memory efficiently.

### Continuous Batching

Traditional batching: wait for N requests, process them together, return all at once. Slow for real-time use.

**Continuous batching:** Dynamically insert and remove requests from a running batch as they arrive and complete. No waiting. Keeps GPU utilization high. Both vLLM and SGLang support this. Essential for production serving.

### Speculative Decoding

Use a small, fast "draft" model to predict the next K tokens. Then verify all K tokens with the large model in **one forward pass** (instead of K sequential passes). If the draft model is right (which it often is for common tokens), you get K tokens for the cost of ~1 large-model pass.

- Speeds up inference 2-3x in favorable cases.
- Requires a compatible draft model.
- Status on AMD: ask speakers. This is newer and may have ROCm-specific gaps.

### Quantization

Compressing model weights from FP16 (16-bit) to smaller formats to reduce memory usage and speed up inference.

| Format | Bits | How It Works | AMD Status |
|--------|------|-------------|------------|
| **FP16** | 16 | Standard, no compression | Fully supported |
| **FP8** | 8 | Native on MI355X. 2x throughput vs. FP16 with small accuracy loss | Key MI355X feature |
| **GPTQ** | 4 | Post-training quantization. Popular. | Works on ROCm via vLLM |
| **AWQ** | 4 | Activation-aware quantization. Better quality than GPTQ for some models. | Works on ROCm via vLLM |
| **GGUF** | 2-8 | llama.cpp format. Flexible mixed precision. | Supported in vLLM |

**FP8 on MI355X** is a major summit talking point. It doubles compute throughput natively in hardware. Ask AMD speakers how production-ready FP8 training and inference are.

### Tensor Parallelism vs. Pipeline Parallelism

When a model is too large for one GPU, split it across multiple GPUs:

- **Tensor parallelism (TP):** Split individual layers across GPUs. Each GPU computes part of every layer. Requires fast inter-GPU communication (NVLink / Infinity Fabric). Lower latency, bandwidth-hungry.
- **Pipeline parallelism (PP):** Split the model into stages, each on a different GPU. Requests flow through stages. Less communication needed but more complex scheduling.

For inference, **TP is standard**. A 70B model on 4x MI300X uses TP=4. The quality of the GPU interconnect (Infinity Fabric) directly affects TP performance.

### Prefix Caching

Many requests share the same system prompt (e.g., "You are a helpful assistant..."). Prefix caching stores the KV cache for shared prefixes and reuses it across requests.

- vLLM supports this.
- SGLang's RadixAttention is built around this concept -- stores prefixes in a radix tree for fast lookup.
- Big efficiency win for RAG workloads where retrieved context is prepended to every query.

---

## Key Metrics for Inference

| Metric | What It Means | What's Good |
|--------|--------------|-------------|
| **Time to first token (TTFT)** | Latency before the first token appears | <500ms for interactive use |
| **Tokens/sec (throughput)** | How many tokens generated per second across all requests | Higher = better GPU utilization |
| **Inter-token latency** | Time between consecutive tokens for one request | <50ms feels "real-time" |
| **Tail latency (P99)** | Worst-case latency for the 99th percentile request | Matters for SLAs |
| **Tokens per dollar** | Cost efficiency | The TCO metric that matters |

**Rule of thumb:** For interactive chat, TTFT and inter-token latency matter. For batch processing (e.g., processing specs, running evals), throughput and tokens-per-dollar matter.

---

## Inference on AMD vs. NVIDIA -- Where Things Stand

| | AMD (MI300X/MI355X) | NVIDIA (H100/B200) |
|---|---------------------|---------------------|
| **vLLM support** | Mature, production-ready | Mature, production-ready |
| **SGLang support** | Works, less tested | Better tested |
| **Quantization (FP8)** | Native on MI355X, strong story | Native on H100+, slightly faster |
| **Quantization (GPTQ/AWQ)** | Works via vLLM | Works via vLLM |
| **Flash-attention** | Community ports, some gaps | Native, fully optimized |
| **Speculative decoding** | Unclear maturity on ROCm | Supported |
| **Large model serving** | 288 GB HBM = fit bigger models | 80-192 GB HBM, may need more sharding |
| **Multi-GPU inference (TP)** | Infinity Fabric, lower bandwidth | NVLink, faster TP communication |

**AMD's pitch:** Bigger memory means you can serve larger models on fewer GPUs, which means lower cost even if per-GPU performance is slightly lower.

---

## Questions to Ask at the Summit (Inference Focus)

- To TensorWave (Greg Diamos): How does ScalarLM compare to vLLM on MI300X in throughput and latency?
- To TensorWave: What is the best serving stack for mixed workloads (embeddings + LLM generation)?
- To AMD (Anush Elangovan): How production-ready is FP8 inference on MI355X?
- To AMD: What is the state of speculative decoding on ROCm?
- To AMD: Flash-attention on MI355X -- native or community port? Benchmarks?
- To Artificial Analysis (Micah Hill-Smith): How do AMD inference providers rank vs. NVIDIA in your benchmarks?
- To Artificial Analysis: What metric matters most in production -- TTFT, tokens/sec, or P99 latency?
- To Featherless AI (Eugene Cheah): What quantization works best on AMD for serverless?
- To Zyphra (Quentin Anthony): Do MoE models benefit from AMD's larger HBM during inference?
- To anyone running vLLM on AMD: What are the sharp edges? What does not work yet?