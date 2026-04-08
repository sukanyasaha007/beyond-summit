# Beyond Summit 2026 -- Quick Prep

**Date:** April 8, 2026 | 8:30 AM - 7:30 PM | San Francisco, CA

---

## 1. Key Points to Review

- This is an **AMD / ROCm** event. The entire narrative is "the world beyond CUDA." Almost every speaker is either building on AMD GPUs, enabling portability away from NVIDIA, or analyzing the shift.
- **MI355X** (CDNA4) is AMD's newest GPU -- 288 GB HBM3e, ~4,600 FP8 TFLOPS, ~750W. Expect it to dominate the conversation.
- **TensorWave** is the host. They sell AMD GPU cloud (bare metal, inference, training). Their inference engine is **ScalarLM**.
- The key software battle: **ROCm** vs. **CUDA**. ROCm is open-source but less mature. Track where the gaps are closing and where they are not.
- **Inference** is AMD's strongest story: more HBM = bigger models in memory, high bandwidth = faster token generation. Training at scale (256+ GPUs) is still NVIDIA's territory because of **NVLink/NVSwitch** interconnect advantages.
- **Cost/TCO** is AMD's second strongest argument: 30-50% cheaper per GPU-hour, lower TDP (power draw).
- **NVIDIA has EDA-specific AI** (ChipNeMo, VerilogEval). **AMD has none.** AMD sees itself as a compute platform, not an EDA AI player. Good angle to probe.
- **Portability approaches** to watch: ZLUDA (runtime CUDA-on-AMD), SCALE/Spectral Compute (compile-time CUDA-on-AMD), Triton (hardware-agnostic kernels), Modular/Mojo (new compiler).
- **Ashish Vaswani** (Essential AI CEO) is here -- he is the lead author of the transformer paper. The biggest name at the event.
- **Dylan Patel** (SemiAnalysis) is the go-to analyst for GPU economics. His cost/supply chain data is widely cited.

### GPU Comparison at a Glance

| Spec | NVIDIA H100 | NVIDIA B200 | AMD MI300X | AMD MI325X | AMD MI355X |
|------|------------|------------|-----------|-----------|-----------|
| **HBM** | 80 GB | 192 GB | 192 GB | 256 GB | 288 GB |
| **Bandwidth** | 3.35 TB/s | 8.0 TB/s | 5.3 TB/s | 6.0 TB/s | 6.0+ TB/s |
| **FP8 TFLOPS** | 1,979 | ~4,500 | 2,615 | 2,615 | ~4,600 |
| **TDP** | 700W | 1,000W | 750W | 750W | ~750W |
| **Interconnect** | NVLink | NVLink 5 | Infinity Fabric | Infinity Fabric | Infinity Fabric |

**AMD wins:** Memory capacity, price-per-TFLOPS, power efficiency, open-source stack.
**NVIDIA wins:** Software ecosystem (15 yrs of CUDA), interconnect at scale, compiler/profiler maturity.

---

## 2. Jargon & Deep Dives -- Separate Review Notes

Detailed jargon, GPU specs, and domain reviews are in dedicated files for focused study:

- **[amd_gpus_&_rocm.md](amd_gpus_&_rocm.md)** -- AMD GPU lineup (MI300X/MI325X/MI355X), ROCm stack, library mapping, portability paths, where AMD wins/loses
- **[nvidia_gpus_&_cuda.md](nvidia_gpus_&_cuda.md)** -- CUDA architecture (programming model, compiler, libraries), NVIDIA GPU lineup, execution model, memory hierarchy, why CUDA lock-in is sticky
- **[inference_vllm_sglang.md](inference_vllm_sglang.md)** -- vLLM vs. SGLang vs. ScalarLM, KV cache, continuous batching, speculative decoding, quantization (FP8/GPTQ/AWQ), inference metrics
- **[eda_review.md](eda_review.md)** -- EDA jargon, VerilogEval/ChipNeMo/VGen benchmarks, AMD vs. NVIDIA in EDA AI, angles to probe

Jargon terms are marked with `[J]` in the speaker questions below. If you see one you do not recognize, check the relevant review note above.

---

## 3. Speakers & Questions

### TensorWave -- AMD GPU Cloud Provider (Host)

**Darrick Horton** (CEO) -- Founded TensorWave to build AMD-native GPU cloud. Bet the company on MI300X when everyone else was chasing NVIDIA allocations.
**Jeff Tatarchuk** (CGO) -- Handles go-to-market and partnerships.
**Piotr Tomasik** (President) -- Operations and infrastructure.
**Greg Diamos** (ScalarLM Architect) -- Previously at NVIDIA and Baidu. Built ScalarLM, TensorWave's inference engine.
**Ilya Tabakh** (VP Innovation) -- R&D and new product development.
**Company Background:** TensorWave is an independent startup (not acquired by AMD) focused on providing bare-metal AMD GPU cloud infrastructure. They specialize in MI-Series accelerators (MI300X, MI325X, MI355X) for AI training, inference, and HPC. SOC2 Type II certified, HIPAA compliant, with UEC-ready networking and direct liquid cooling for energy efficiency.

**Questions:**
- How long does a team take to migrate from NVIDIA cloud to TensorWave?
- How does **[ScalarLM](inference_vllm_sglang.md#scalarlm)** compare to **[vLLM](inference_vllm_sglang.md#vllm)** on AMD in throughput and latency?
- When is **[MI355X](amd_gpus_&_rocm.md#amd-instinct-gpu-lineup)** available on TensorWave, and what does it cost vs. MI300X?
- Is TensorWave worth it for smaller workloads (embeddings, RAG) or only large training jobs?
- What comes pre-installed on a fresh node? **[ROCm](amd_gpus_&_rocm.md#rocm--amds-software-stack)** version, containers, drivers?
- What is the best serving stack for mixed workloads (embeddings + LLM generation)?
- (EDA) Has anyone used your clusters for EDA AI -- fine-tuning on Verilog code or running **[VerilogEval](eda_review.md#verilogevalvgen-benchmarks)**?

---

### AMD -- GPU Manufacturer

**Anush Elangovan** (VP AI Software) -- Leads the ROCm software ecosystem team. Responsible for closing the gap with CUDA.
**Neha Prakriya** (GenAI Training at Scale) -- Focuses on large-scale model training on Instinct GPUs.

**Questions:**
- What is next for **[ROCm](amd_gpus_&_rocm.md#rocm--amds-software-stack)**? Which CUDA gaps are being closed in 2026?
- How production-ready is **[FP8](amd_gpus_&_rocm.md#amd-instinct-gpu-lineup)** on **[MI355X](amd_gpus_&_rocm.md#amd-instinct-gpu-lineup)** for training and inference?
- Is AMD investing in **[Triton](amd_gpus_&_rocm.md#compilers)** or building a proprietary compiler?
- **[Flash-attention](amd_gpus_&_rocm.md#the-real-pain-points)** on MI355X -- native or community port? Benchmarks vs. NVIDIA?
- How do embedding model workloads (high throughput, small models) perform on MI355X vs. LLM inference?
- What is the state of **[ROCm](amd_gpus_&_rocm.md#rocm--amds-software-stack)** support for smaller, specialized models that could run inference in EDA pipelines?
- NVIDIA published **[ChipNeMo](eda_review.md#chipnemo)** and **[VerilogEval](eda_review.md#verilogevalvgen-benchmarks)** for chip design AI. Is AMD planning EDA-specific AI, or leaving it to EDA vendors on ROCm?
- (Indirect EDA) Does AMD's own chip design team run ML training on Instinct GPUs with ROCm? (Dogfooding - using own products for internal work)
- (Indirect EDA) **Coverage closure** (percentage of test conditions covered) at AMD's scale is massive. How does the verification team handle it?
- (For Neha) When AMD trains models internally, do you use MI300X clusters or still depend on NVIDIA?
- Why did AMD build **[ROCm](amd_gpus_&_rocm.md#rocm--amds-software-stack)** when **[Triton](amd_gpus_&_rocm.md#compilers)** already provides cross-platform kernel compilation?
- How does **[Triton](amd_gpus_&_rocm.md#compilers)** integrate with **[ROCm](amd_gpus_&_rocm.md#rocm--amds-software-stack)** for AI/ML workloads, and what are the performance trade-offs?
- What are AMD's plans for enhancing **[Triton](amd_gpus_&_rocm.md#compilers)** support in future **[ROCm](amd_gpus_&_rocm.md#rocm--amds-software-stack)** versions?
- Why focus on **[hipcc](amd_gpus_&_rocm.md#compilers)** instead of **[Triton](amd_gpus_&_rocm.md#compilers)**, since many CUDA compiler users already know Triton and might find it easier to migrate?
- How does AMD maintain strong relationships with big competitive cloud providers (AWS, Google Cloud) and also fast-paced emerging providers like TensorWave?
- *Tip:* Talk to AMD engineers in the audience at networking, not just the VPs on stage.

---

### Essential AI -- Enterprise AI

**Ashish Vaswani** (CEO) -- Lead author of "Attention Is All You Need," the original transformer paper. Co-invented the architecture that powers every modern LLM.
**Company Overview:** Essential AI is an open-source AI company focused on accelerating deep learning science and engineering. They build "instruments of intelligence" -- high-quality LLMs and datasets for STEM and code capabilities. Their flagship release is Rnj-1 (base and instruction-tuned models), available on Hugging Face. Mission emphasizes open collaboration to prevent AI progress from being privatized. They publish research on efficient training (e.g., Muon optimizer) and datasets (e.g., Essential-Web v1.0 with 24T tokens).

**Questions:**
- What comes after transformers? What architectures are gaining real traction?
- Does Essential AI target AMD GPUs, or only NVIDIA?
- What do hybrid architectures (transformers + state-space models) look like in production?
- How does Essential AI approach agentic AI systems -- autonomous agents for enterprise tasks?
- How does Rnj-1 perform on AMD GPUs with ROCm? Any optimizations for open-source stacks?
- What's next after Rnj-1 -- larger models, multimodal, or new architectures?
- How does Essential AI's open-source approach differ from closed companies like OpenAI or Anthropic?

---

### SemiAnalysis -- Semiconductor & AI Analysis

**Dylan Patel** (Founder) -- The most-cited independent analyst for GPU economics, supply chain, and chip strategy. His reports move markets.

**Questions:**
- What are the real **TCO** (Total Cost of Ownership) numbers for AMD vs. NVIDIA at scale?
- Where does AMD win (memory, price) and where does it still lose (software, compilers)?
- Is AMD GPU capacity actually easier to procure than NVIDIA right now?
- How do custom chips (Google TPU, AWS Trainium) compare to AMD and NVIDIA on TCO?

---

### Liquid AI -- Non-Transformer Architectures

**Mathias Lechner** (CTO) -- MIT spin-off. Researches liquid neural networks and state-space models as alternatives to transformers.

**Questions:**
- Do liquid networks run well on AMD GPUs with **[ROCm](amd_gpus_&_rocm.md#rocm--amds-software-stack)**?
- Are they more memory-efficient for long documents (relevant to reading chip specs)?
- Can these models run on standard serving tools like **[vLLM](inference_vllm_sglang.md#vllm)**?
- Could liquid networks be better than transformers for ingesting long **[RTL](eda_review.md)** (Register Transfer Language) specs?

---

### ZLUDA -- CUDA-on-AMD Runtime

**Andrzej Janik** (Founder) -- Built **[ZLUDA](amd_gpus_&_rocm.md#portability-paths-getting-from-cuda-to-amd)**, a drop-in binary compatibility layer that runs CUDA programs on AMD GPUs without recompilation.

**Questions:**
- What percentage of CUDA applications work through **[ZLUDA](portability-paths)** today?
- How much slower is it vs. native **[ROCm](amd_gpus_&_rocm.md#rocm--amds-software-stack)** / **[HIP](amd_gpus_&_rocm.md#hip-gpu-programming-language)**?
- Is it production-ready or still experimental?
- Is ZLUDA a real migration path, or only for prototyping before a full **[HIP](amd_gpus_&_rocm.md#hip-gpu-programming-language)** port?

---

### Modular -- Mojo Language & Inference Engine

**Mostafa Hagog** (VP Engineering) -- Building Mojo, a new systems language that targets multiple GPU backends from a single codebase.

**Questions:**
- How does Modular compare to **[Triton](amd_gpus_&_rocm.md#compilers)** for writing GPU kernels that target AMD?
- Is anyone using Mojo for production custom kernels yet?
- Can Modular run identical code on NVIDIA and AMD with zero changes?
- Where does Modular beat stock PyTorch the most?

---

### Zyphra -- Efficient Language Models

**Quentin Anthony** (VP Engineering) -- Builds the Zamba series: smaller, efficient models (often **MoE** - Mixture of Experts) that compete with larger ones.

**Questions:**
- How was the experience training on AMD? What broke?
- Do **MoE** (Mixture of Experts) and hybrid architectures benefit from AMD's larger **[HBM](amd_gpus_&_rocm.md#amd-instinct-gpu-lineup)**?
- Which distributed training framework works best on AMD -- DeepSpeed, FSDP, or Megatron? How is **[RCCL](amd_gpus_&_rocm.md#libraries)** performing?

---

### Spectral Compute -- SCALE (Compile-Time CUDA→AMD)

**Michael Sondergaard** (CEO) -- Built **[SCALE](amd_gpus_&_rocm.md#portability-paths-getting-from-cuda-to-amd)**, which recompiles CUDA source code to AMD GPU targets at build time. Compile-time alternative to ZLUDA's runtime approach.

**Questions:**
- How does **[SCALE](amd_gpus_&_rocm.md#portability-paths-getting-from-cuda-to-amd)** differ from **[ZLUDA](amd_gpus_&_rocm.md#portability-paths-getting-from-cuda-to-amd)** in practice? When would you use one vs. the other?
- How fast is SCALE-compiled code vs. native **[ROCm](amd_gpus_&_rocm.md#rocm--amds-software-stack)** / **[HIP](amd_gpus_&_rocm.md#hip-gpu-programming-language)**?
- Which CUDA libraries does it cover (cuDNN, cuBLAS, etc.)?
- Is SCALE free or commercial?

---

### Artificial Analysis -- Independent AI Benchmarking

**Micah Hill-Smith** (CEO) -- Publishes independent leaderboards ranking AI models and inference providers on speed, cost, and quality.

**Questions:**
- How does AMD-based inference rank vs. NVIDIA-based providers in your benchmarks?
- What metric matters most for production inference -- time-to-first-token, tokens/sec, or tail latency?
- Are AMD providers catching up in the rankings?

---

### Featherless AI -- Serverless Model Inference

**Eugene Cheah** (CEO) -- Deploy any HuggingFace model as a serverless endpoint without managing infrastructure.

**Questions:**
- How stable is **[ROCm](amd_gpus_&_rocm.md#rocm--amds-software-stack)** for running many different model architectures?
- What **[quantization](inference_vllm_sglang.md#quantization)** formats work best on AMD for serverless workloads?
- How do you handle less popular models on AMD hardware?

---

### dstack -- Multi-Cloud GPU Orchestration

**Marc Fleury** (Co-Founder) -- Open-source tool for managing GPU compute jobs across multiple cloud providers.

**Questions:**
- Does dstack support AMD GPUs and TensorWave?
- How easy is it to move a job from NVIDIA to AMD in dstack?
- How does dstack compare to other GPU job orchestrators (SkyPilot, etc.)?

---

### Red Hat -- Enterprise Linux & Kubernetes AI

**Ron Haberman** (AI Incubation) -- Working on AI platform tooling within Red Hat's OpenShift (Kubernetes) ecosystem.

**Questions:**
- Does OpenShift AI support AMD GPUs well?
- How mature are AMD GPU device plugins for Kubernetes?
- Are enterprise customers actually running AMD GPU workloads on OpenShift in production?
- How does Red Hat approach agentic AI in Kubernetes -- e.g., AI-driven orchestration or autonomous scaling?

---

### MLPerf -- Industry AI Benchmarks

**David Kanter** (Founder) -- Created **[MLPerf](https://mlperf.org/)**, the industry-standard hardware-neutral AI benchmark suite. The "SPEC" of AI hardware.

**Questions:**
- How are AMD GPUs doing in recent **[MLPerf](amd_gpus_&_rocm.md)** (industry AI benchmark suite) results vs. NVIDIA and Google TPU?
- Where does AMD score best and worst?
- Are more companies submitting AMD results, or is it still mostly NVIDIA?

---

### Credo -- GPU Cluster Interconnects (Sponsor)

**Bill Brennan** (CEO) -- Makes high-speed cables and optical interconnects for GPU clusters. The physical layer that connects GPUs across racks.

**Questions:**
- What bandwidth do **[MI355X](amd_gpus_&_rocm.md#amd-instinct-gpu-lineup)** clusters need from interconnects?
- How does Credo compare to **[NVLink](nvidia_gpus_&_cuda.md)** for GPU-to-GPU communication at rack scale?
- At what cluster scale does **[Infinity Fabric](amd_gpus_&_rocm.md#interconnect-bandwidth-comparison)**'s bandwidth gap vs. NVLink materially hurt training?

---

### Sanmina -- Server & Rack Manufacturing

**Casey Cerretani** (CTO) -- Builds the physical servers and racks that GPUs go into. System-level thermal and power engineering.

**Questions:**
- What are the cooling and power challenges for AMD GPU racks vs. NVIDIA?
- How does liquid cooling differ between AMD and NVIDIA server designs?

---

### PG&E -- Power Utility

**Jon Stallman** (Utility Partnerships & Innovation) -- From the power company that supplies Northern California. Here because AI datacenters are driving massive power demand.

**Questions:**
- How much power are AI datacenters demanding in the Bay Area?
- How are utilities planning for megawatt-scale GPU clusters?
- Does the **[TDP](amd_gpus_&_rocm.md#amd-instinct-gpu-lineup)** (Thermal Design Power) difference (750W AMD vs. 1,000W NVIDIA) matter at datacenter scale?

---

### AT&T -- Enterprise AI User

**Farbod Tavakkoli** (Data Scientist) -- Uses AI/ML for telecom network operations at AT&T.

**Questions:**
- What AI workloads is AT&T running on AMD GPUs?
- How does a large enterprise evaluate AMD vs. NVIDIA for production?
- Are there agentic AI applications in telecom network operations -- e.g., autonomous fault detection or optimization?

---

### Wafer AI -- Wafer-Scale AI Chips

**Emilio Andere** (CEO) -- Building wafer-scale AI chips (entire chip on one silicon wafer, like Cerebras). Alternative to discrete GPUs.

**Questions:**
- Where does wafer-scale compute fit alongside GPUs?
- What workloads suit wafer-scale better than discrete GPUs?

---

### Quixi AI / Cognitive Computations -- Open-Source Fine-Tuning

**Eric Hartford** (Founder) -- Open-source fine-tuning leader. Known for "dolphin" and "samantha" model families. Prolific in the HuggingFace community.

**Questions:**
- How is fine-tuning large models on AMD GPUs with **[ROCm](amd_gpus_&_rocm.md#rocm--amds-software-stack)** in practice?
- Which fine-tuning tool works best on AMD -- Axolotl, LLaMA-Factory, or PEFT?
- What are the biggest pain points for the open-source community on AMD?
- (EDA) Has anyone tried running **[VerilogEval](eda_review.md#verilogevalvgen-benchmarks)** (Verilog code generation benchmark) on AMD hardware?

---

### Silares

**Akhil Sharma** (CEO) -- Limited public info. Ask what Silares does in AI infrastructure.

---

### Lumerian Labs

**Jay Dawani** (CEO) -- Limited public info. Ask what Lumerian Labs is building and how it connects to AMD/open infrastructure.

---

### Event Sponsors

| Sponsor | Role |
|---------|------|
| AMD | GPU manufacturer, ROCm stack |
| ZT Systems | AI/HPC server design (acquired by AMD in 2024) |
| Credo | High-speed interconnects |
| EdgeCore | AI datacenter infrastructure |

---

## 4. Elevator Pitch

"I work on AI-powered verification tooling for chip design -- using LLMs and RAG to help engineers navigate specifications and coverage data. I am here to understand how AMD's compute stack is maturing for production AI workloads."

---

## 5. Logistics

- [ ] Confirm registration
- [ ] Charge devices, bring portable charger
- [ ] LinkedIn QR code for contact sharing
- [ ] Notebook for session notes
- [ ] Review session schedule when published

---

## 6. Post-Event

- Write up key takeaways
- Follow up with contacts within 48 hours
- Evaluate if AMD GPUs are worth exploring for your workloads
