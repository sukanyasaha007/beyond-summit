# Beyond Summit 2026 -- Preparation Notes

Quick-reference materials for **Beyond Summit 2026** (April 8, 2026, San Francisco) -- an AMD/ROCm AI infrastructure event hosted by TensorWave.

---

## Quick Start

1. **Start here:** [prep_notes_v2.md](prep_notes_v2.md)
   - Key talking points, all 20+ speakers with 1-2 line backgrounds, targeted questions for each
   - GPU comparison table
   - Elevator pitch and logistics checklist

2. **Deep dives** (open when you need them):
   - [amd_gpus_&_rocm.md](amd_gpus_&_rocm.md) -- AMD GPU lineup (MI300X/MI325X/MI355X), ROCm stack, library mapping, where AMD wins/loses
   - [nvidia_gpus_&_cuda.md](nvidia_gpus_&_cuda.md) -- CUDA architecture (3 layers of lock-in), execution model, memory hierarchy, "beyond CUDA" paths
   - [inference_vllm_sglang.md](inference_vllm_sglang.md) -- vLLM vs. SGLang vs. ScalarLM, KV cache, quantization, inference metrics
   - [eda_review.md](eda_review.md) -- EDA jargon, VerilogEval/ChipNeMo benchmarks, probe angles for chip design AI

3. **Hands-on practice:**
   - [llm_finetune_colab.ipynb](llm_finetune_colab.ipynb) -- Google Colab notebook for QLoRA fine-tuning with real Verilog data on T4 GPU

---

## File Breakdown

| File | Purpose | When to Use |
|------|---------|------------|
| **prep_notes_v2.md** | Main quick reference. Key themes, all speakers, questions, logistics | Scan 10 mins before the event; bring to the event for quick lookups |
| **amd_gpus_&_rocm.md** | AMD GPU specs, ROCm software stack, library gaps, where AMD wins/loses, portability paths | When speakers discuss MI355X, ROCm maturity, RCCL scale, or you want to ask differentiated questions |
| **nvidia_gpus_&_cuda.md** | CUDA architecture (programming model, compiler, libraries), NVIDIA lineup, why CUDA lock-in is sticky, "beyond CUDA" solutions | When speakers discuss ZLUDA, SCALE, Triton, Modular, or why switching from CUDA is hard |
| **inference_vllm_sglang.md** | vLLM vs. SGLang comparison, KV cache strategies, speculative decoding, quantization (FP8/GPTQ/AWQ), inference metrics, AMD vs. NVIDIA | When talks focus on LLM serving, TensorWave's ScalarLM, or inference workloads |
| **eda_review.md** | EDA jargon, VerilogEval/ChipNeMo/VGen benchmarks, AMD vs. NVIDIA in chip design AI, your use case mapping | If you want to ask chip design AI questions; probing AMD's internal AI strategy |
| **llm_finetune_colab.ipynb** | Runnable Google Colab notebook: QLoRA fine-tuning of TinyLlama-1.1B on T4 GPU with real Verilog data | Before the event: run locally to understand fine-tuning stack on AMD GPUs |

---

## Key Themes to Track at the Event

1. **ROCm Ecosystem Maturity** -- How close is developer experience to CUDA parity?
2. **MI355X Performance** -- AMD's latest GPU with 288 GB HBM3e. The star of the show.
3. **Multi-GPU / Distributed Training** -- RCCL vs. NCCL at scale. Infinity Fabric vs. NVLink.
4. **Inference Stack** -- vLLM, SGLang, ScalarLM. AMD's strongest story.
5. **Portability** -- ZLUDA (runtime), SCALE (compile-time), Triton (hardware-agnostic kernels).
6. **Cost / TCO** -- AMD's 30-50% price-performance advantage and 25% lower power draw.
7. **EDA AI** -- NVIDIA has ChipNeMo, VerilogEval. AMD has none. Strategic gap worth probing.

---

## Who's Speaking (At a Glance)

| Name | Company | Role | Key Angle |
|------|---------|------|-----------|
| Darrick Horton | TensorWave | CEO / Host | AMD GPU cloud, MI355X availability, ScalarLM vs. vLLM |
| Anush Elangovan | AMD | VP AI Software | ROCm gaps, FP8, Triton, EDA AI |
| Neha Prakriya | AMD | GenAI Training at Scale | Distributed training on AMD, internal dogfooding |
| Ashish Vaswani | Essential AI | CEO | Transformer paper author. What comes after transformers? |
| Dylan Patel | SemiAnalysis | Founder | GPU economics, cost/supply chain data |
| Andrzej Janik | ZLUDA | Founder | Running CUDA on AMD without code changes |
| Mostafa Hagog | Modular | VP Engineering | Mojo: new language targeting all GPUs |
| Mathias Lechner | Liquid AI | CTO | Non-transformer architectures, state-space models |
| Quentin Anthony | Zyphra | VP Engineering | Efficient models (Zamba), MoE on AMD |
| Michael Sondergaard | Spectral Compute | CEO | SCALE: compile-time CUDA→AMD translation |
| Greg Diamos | TensorWave | ScalarLM Architect | AMD-native inference engine |
| Eric Hartford | Quixi AI / Cognitive Computations | Founder | Open-source fine-tuning, VerilogEval on AMD? |

(20+ speakers / companies total)

---

## Jargon Quick Links

Key terms appear in speaker questions with `[J]` markers. Definitions are in the relevant review note:

- **GPU/AI Terms:** See [amd_gpus_&_rocm.md](amd_gpus_&_rocm.md), [nvidia_gpus_&_cuda.md](nvidia_gpus_&_cuda.md), [inference_vllm_sglang.md](inference_vllm_sglang.md)
- **EDA/Chip Design Terms:** See [eda_review.md](eda_review.md)

---

## Your Elevator Pitch

"I work on AI-powered verification tooling for chip design -- using LLMs and RAG to help engineers navigate specifications and coverage data. I am here to understand how AMD's compute stack is maturing for production AI workloads."

---

## Before the Event

- [ ] Skim [prep_notes_v2.md](prep_notes_v2.md) (10 mins)
- [ ] Run [llm_finetune_colab.ipynb](llm_finetune_colab.ipynb) on Google Colab (30 mins) to get hands-on feel for fine-tuning and quantization
- [ ] Charge devices, bring portable charger
- [ ] Business cards / LinkedIn QR code for contact sharing
- [ ] Notebook for session notes
- [ ] Print or bookmark this repo for quick phone reference

---

## During the Event

- Track **where the sharp edges are** on AMD vs. NVIDIA (custom kernels, library gaps, driver stability)
- Ask about **real performance numbers** -- not marketing claims
- Probe **EDA AI** -- NVIDIA's ChipNeMo vs. AMD's nothing. Why?
- Focus on **inference** -- AMD's strongest story. Get real throughput/latency data.
- **Network:** Talk to engineers in the audience, not just VPs on stage

---

## After the Event

- Write up key takeaways in this repo
- Follow up with contacts within 48 hours
- Evaluate if AMD GPUs fit your workload (fine-tuning, inference, embeddings)

---

## Resources Referenced

- **VerilogEval:** `github.com/NVlabs/verilog-eval`
- **VGen:** `github.com/shailja-thakur/VGen`
- **Verilog dataset:** `shailja/Verilog_GitHub` on HuggingFace (109k modules)
- **vLLM:** `github.com/lm-sys/vllm`
- **SGLang:** `github.com/sgl-project/sglang`
- **ROCm:** `rocmdocs.amd.com`
- **ZLUDA:** `github.com/andrzejjanik/zluda`
- **SCALE (Spectral Compute):** `spectral.compute`
- **Modular/Mojo:** `modular.com`

---

**Event Date:** April 8, 2026 | 8:30 AM - 7:30 PM | San Francisco

Breakfast, lunch, snacks, and evening bites provided. Format includes keynotes, breakout sessions, and networking.