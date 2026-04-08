# Beyond Summit 2026 -- Preparation Notes

**Date:** April 8, 2026 | 8:30 AM - 7:30 PM  
**Location:** San Francisco, CA (check registration for exact address)

---

## Key Themes to Track

### 1. ROCm Ecosystem Maturity
- ROCm is AMD's answer to CUDA. Track how close the developer experience is to CUDA parity.
- Questions to ask speakers:
  - What are the remaining sharp edges when porting CUDA kernels to HIP?
  - How stable is ROCm across driver versions in production?
  - What is the state of flash-attention on MI300X/MI355X?

### 2. MI355X Performance Tuning
- AMD Instinct MI355X is the latest generation. Understand memory bandwidth, HBM capacity, and how it compares to H100/B200.
- Questions:
  - What batch sizes and sequence lengths show the best throughput on MI355X?
  - How does the memory hierarchy differ in practice vs. NVIDIA Hopper/Blackwell?
  - What profiling tools are mature enough for production use (rocprof, omniperf)?

### 3. Multi-GPU / Distributed Training
- RCCL (AMD's NCCL equivalent) maturity.
- Topology-aware placement on AMD GPU clusters.
- Questions:
  - What interconnect (Infinity Fabric, PCIe, RoCE) configurations are teams using?
  - How does RCCL collective performance compare to NCCL at scale (256+ GPUs)?

### 4. Inference Stack: vLLM, TensorRT-LLM alternatives
- vLLM on ROCm is a major talking point. Understand current throughput/latency numbers.
- Questions:
  - What quantization formats (GPTQ, AWQ, FP8) work best on MI355X?
  - What is the state of speculative decoding on AMD hardware?
  - How does continuous batching perform under real production load?

### 5. Portability (PyTorch, JAX, vLLM)
- PyTorch has first-class ROCm support. JAX support is less mature.
- Questions:
  - What is the JAX-on-ROCm story for training? Is XLA fully functional?
  - Are there framework-level abstractions that make GPU-agnostic code realistic?

### 6. Cost / TCO
- AMD GPUs often have better price-performance on paper. Get real numbers.
- Questions:
  - What is the all-in TCO (power, cooling, software eng overhead) vs. NVIDIA?
  - How do cloud pricing models compare (TensorWave, CoreWeave, Lambda)?

---

## EDA Jargon Cheat Sheet

- **Coverage closure** = getting your test suite to hit all the design scenarios you defined as goals before the chip goes to manufacturing. Think of it as filling a checklist -- every box must be ticked.
- **RTL** = Register Transfer Level. The Verilog/VHDL code that describes what a chip does. It is the "source code" of hardware.
- **DUT** = Design Under Test. The chip block you are verifying.
- **Regression** = re-running all tests after a code change to make sure nothing broke.
- **Formal verification** = proving a design property is always true mathematically, without running simulations.
- **State space** = all possible combinations of inputs and internal states a chip can be in. For real chips, this is astronomically large.
- **Dogfooding** = a company using its own product internally.

---

## Verilog/EDA AI Benchmarks: NVIDIA vs. AMD

**NVIDIA has EDA-specific AI benchmarks. AMD has none.**

| Benchmark | Source | What it is |
|-----------|--------|-----------|
| **VerilogEval v1/v2** | NVlabs | 156 Verilog problems with testbenches. Code-completion + spec-to-RTL. Industry standard. (`github.com/NVlabs/verilog-eval`) |
| **VGen** | NYU | Fine-tuned CodeGen on Verilog. Uses `shailja/Verilog_GitHub` dataset (109k modules). (`github.com/shailja-thakur/VGen`) |
| **ChipNeMo** | NVIDIA | Internal model fine-tuned on proprietary chip design data. Paper only, not open-sourced. |

AMD has zero public EDA/Verilog benchmarks or models. Their strategy is to be a compute platform, leaving EDA AI to the EDA vendors.

---

## Companies & Speakers -- What to Ask Each

### TensorWave (Host / AMD GPU Cloud Provider)
**Speakers:** Darrick Horton (CEO), Jeff Tatarchuk (CGO), Piotr Tomasik (President), Greg Diamos (ScalarLM Architect), Ilya Tabakh (VP Innovation)
**What they do:** AMD GPU cloud -- bare metal, inference, training. MI300X/MI325X/MI355X.
**Questions:**
- How long does it take a team to move from NVIDIA cloud to TensorWave?
- How does ScalarLM compare to vLLM on AMD?
- When is MI355X available and what does it cost vs. MI300X?
- Is TensorWave worth it for smaller workloads like embeddings and RAG, or only big training jobs?
- What comes pre-installed on a fresh node? ROCm version, containers, drivers?
- What is the best serving stack for mixed workloads (embeddings + LLM generation) on AMD?
- **(EDA angle)** Has anyone used your clusters for EDA AI -- like fine-tuning on Verilog code?

### AMD (GPU Manufacturer)
**Speakers:** Anush Elangovan (VP AI Software), Neha Prakriya (GenAI Training at Scale)
**What they do:** ROCm software stack, Instinct GPU hardware, MI355X.
**Questions:**
- What is coming next for ROCm? Which CUDA gaps are being closed?
- How ready is FP8 on MI355X for real training and inference?
- Is AMD backing Triton or building their own compiler?
- Is AMD looking at GPU acceleration for EDA (simulation, verification)?
- Flash-attention on MI355X -- native or community port? How fast vs. NVIDIA?
- How do embedding model workloads (smaller models, high throughput) perform on MI355X vs. LLM inference?
- What is the state of ROCm support for smaller, specialized models (not just LLMs) that could run inference in EDA pipelines?
- NVIDIA invested in ChipNeMo and VerilogEval. Is AMD planning EDA-specific AI, or leaving it to EDA vendors building on ROCm?
- **(Indirect EDA probe)** Does AMD's own chip design team run ML training on Instinct GPUs with ROCm?
- **(Indirect EDA probe)** Coverage closure at AMD's scale must be a massive problem. How does the verification team handle that?
- **(Competitive framing)** NVIDIA published ChipNeMo for chip design AI. Has AMD done something similar?
- **(Best for Neha)** When AMD trains models internally, do you use your own MI300X clusters or still depend on NVIDIA?
- **Tip:** Talk to AMD engineers in the audience at networking, not just the VPs on stage.

### Essential AI
**Speaker:** Ashish Vaswani (CEO) -- lead author of the original transformer paper ("Attention Is All You Need").
**What they do:** Enterprise AI products.
**Questions:**
- What comes after transformers? What architectures are gaining real traction?
- Does Essential AI build for AMD GPUs too, or just NVIDIA?
- What do hybrid architectures (transformers + state-space models) look like in practice?

### SemiAnalysis
**Speaker:** Dylan Patel (Founder)
**What they do:** Semiconductor and AI analysis. The go-to source for GPU economics and supply chain data.
**Questions:**
- What are the real cost numbers for AMD vs. NVIDIA at scale?
- Where does AMD win (memory, price) and where does it still lose (software, compilers)?
- Is AMD GPU capacity actually easier to get than NVIDIA right now?
- How do custom chips (Google TPU, AWS Trainium) compare to AMD and NVIDIA?

### Liquid AI
**Speaker:** Mathias Lechner (CTO) -- MIT spin-off.
**What they do:** Non-transformer architectures (liquid neural networks, state-space models).
**Questions:**
- Do liquid networks run well on AMD GPUs?
- Are they more memory-efficient for long documents?
- Can these models run on standard serving tools like vLLM?
- Could liquid networks be better than transformers for reading long chip specs?

### ZLUDA
**Speaker:** Andrzej Janik (Founder)
**What they do:** Run CUDA programs on AMD GPUs without changing code. Drop-in compatibility layer.
**Questions:**
- What percentage of CUDA apps work through ZLUDA today?
- How much slower is it vs. native ROCm?
- Is it production-ready or still experimental?
- Is ZLUDA a real migration path or just for prototyping?

### Modular
**Speaker:** Mostafa Hagog (VP Engineering)
**What they do:** Mojo language and Modular inference engine. Write code once, run on any GPU.
**Questions:**
- How does Modular compare to Triton for targeting AMD GPUs?
- Is anyone actually using Mojo for custom GPU kernels?
- Can Modular run the same code on NVIDIA and AMD with zero changes?
- Where does Modular beat stock PyTorch the most?

### Zyphra
**Speaker:** Quentin Anthony (VP Engineering)
**What they do:** Efficient language models (Zamba series). Smaller models that compete with bigger ones.
**Questions:**
- How was the experience training on AMD? What broke?
- Do efficient architectures (MoE, hybrid) benefit from AMD's larger memory?
- Which training framework works best on AMD -- DeepSpeed, FSDP, or Megatron?

### Spectral Compute
**Speaker:** Michael Sondergaard (CEO)
**What they do:** SCALE -- compiles CUDA code to run on AMD at build time. Different from ZLUDA (which does it at runtime).
**Questions:**
- How does SCALE differ from ZLUDA in practice?
- How fast is SCALE-compiled code vs. native ROCm?
- Which CUDA libraries does it cover?
- Is SCALE free or paid?

### Artificial Analysis
**Speaker:** Micah Hill-Smith (CEO)
**What they do:** Independent benchmarking of AI models and inference providers. Publishes leaderboards.
**Questions:**
- How does AMD-based inference rank against NVIDIA-based providers?
- What matters most for inference in production -- first token speed, tokens/sec, or tail latency?
- Are AMD providers catching up in the rankings?

### Featherless AI
**Speaker:** Eugene Cheah (CEO)
**What they do:** Serverless model inference. Deploy any HuggingFace model without managing servers.
**Questions:**
- How stable is ROCm for running many different model architectures?
- How do you handle less popular models on AMD hardware?
- What quantization works best on AMD for serverless?

### dstack
**Speaker:** Marc Fleury (Co-Founder)
**What they do:** Open-source tool to manage GPU jobs across multiple clouds.
**Questions:**
- Does dstack support AMD GPUs and TensorWave?
- How easy is it to switch a job from NVIDIA to AMD in dstack?
- How does dstack compare to other GPU job orchestration tools?

### RedHat
**Speaker:** Ron Haberman (AI Incubation)
**What they do:** Enterprise Linux, OpenShift (Kubernetes), AI platform tooling.
**Questions:**
- Does OpenShift AI support AMD GPUs well?
- How mature are AMD GPU plugins for Kubernetes?
- Are real enterprise customers running AMD GPU workloads on OpenShift?

### MLPerf
**Speaker:** David Kanter (Founder)
**What they do:** Industry-standard AI benchmarks. Hardware-neutral.
**Questions:**
- How are AMD GPUs doing in recent MLPerf results vs. NVIDIA and Google TPU?
- Where does AMD score best and worst in MLPerf?
- Are more companies submitting AMD results, or is it still mostly NVIDIA?

### Credo (Sponsor + Speaker)
**Speaker:** Bill Brennan (CEO)
**What they do:** High-speed cables and interconnects for GPU clusters.
**Questions:**
- What bandwidth do MI355X clusters need?
- How does Credo compare to NVIDIA's NVLink for GPU-to-GPU communication?

### Sanmina
**Speaker:** Casey Cerretani (CTO)
**What they do:** Builds the physical servers and racks that GPUs go into.
**Questions:**
- What are the cooling and power challenges for AMD GPU racks vs. NVIDIA?
- How does liquid cooling differ between AMD and NVIDIA systems?

### PG&E
**Speaker:** Jon Stallman (Utility Partnerships & Innovation)
**What they do:** The power company for Northern California.
**Questions:**
- How much power are AI datacenters demanding in the Bay Area?
- How are utilities planning for megawatt-scale GPU clusters?

### AT&T
**Speaker:** Farbod Tavakkoli (Data Scientist)
**What they do:** Telecom. Uses AI/ML for network operations.
**Questions:**
- What AI workloads is AT&T running on AMD GPUs?
- How does a big enterprise decide between AMD and NVIDIA?

### Wafer AI
**Speaker:** Emilio Andere (CEO)
**What they do:** Wafer-scale AI chips (entire chip on one wafer, like Cerebras).
**Questions:**
- Where does wafer-scale compute fit alongside GPUs?
- What workloads suit wafer-scale better than discrete GPUs?

### Quixi AI / Cognitive Computations
**Speaker:** Eric Hartford (Founder)
**What they do:** Open-source fine-tuning leader. Known for "dolphin" and "samantha" models.
**Questions:**
- How is fine-tuning large models on AMD GPUs in practice?
- Which fine-tuning tool works best on ROCm -- Axolotl, LLaMA-Factory, or PEFT?
- What are the biggest pain points for the open-source community on AMD?
- **(EDA angle)** Has anyone tried running VerilogEval (Verilog code generation benchmark) on AMD GPUs?

### Silares
**Speaker:** Akhil Sharma (CEO)
**Questions:**
- What does Silares do in the AI infrastructure space?

### Lumerian Labs
**Speaker:** Jay Dawani (CEO)
**Questions:**
- What is Lumerian Labs building and how does it connect to AMD/open infrastructure?

---

### Event Sponsors
| Sponsor | What They Do |
|---------|-------------|
| **AMD** | GPU manufacturer, ROCm stack |
| **ZT Systems** | AI/HPC server design and manufacturing (acquired by AMD in 2024) |
| **Credo** | High-speed interconnects for AI clusters |
| **EdgeCore** | AI datacenter infrastructure |

---

## Networking Strategy

### Your Elevator Pitch
"I work on AI-powered verification tooling for chip design -- using LLMs and RAG to help engineers navigate specifications and coverage data. I am here to understand how AMD's compute stack is maturing for production AI workloads."

---

## Logistics Checklist

- [ ] Confirm registration approval
- [ ] Charge devices, bring portable charger
- [ ] Business cards or contact-sharing method (LinkedIn QR code)
- [ ] Notebook for session notes
- [ ] Review the speaker/session schedule when published
- [ ] Download any workshop prerequisites (ROCm SDK, Docker images) if hands-on sessions require it

---

## Quick Reference: AMD vs. NVIDIA GPU Comparison

| Spec | NVIDIA H100 | NVIDIA B200 | AMD MI300X | AMD MI325X | AMD MI355X |
|------|------------|------------|-----------|-----------|-----------|
| **Architecture** | Hopper | Blackwell | CDNA3 | CDNA3 | CDNA4 |
| **HBM** | 80 GB HBM3 | 192 GB HBM3e | 192 GB HBM3 | 256 GB HBM3e | 288 GB HBM3e |
| **Bandwidth** | 3.35 TB/s | 8.0 TB/s | 5.3 TB/s | 6.0 TB/s | 6.0+ TB/s |
| **FP16 TFLOPS** | 989 | ~2,250 | 1,307 | 1,307 | ~2,300 (est.) |
| **FP8 TFLOPS** | 1,979 | ~4,500 | 2,615 | 2,615 | ~4,600 (est.) |
| **TDP** | 700W | 1,000W | 750W | 750W | ~750W |
| **Interconnect** | NVLink | NVLink 5 | Infinity Fabric | Infinity Fabric | Infinity Fabric |

**Where AMD wins:** Memory capacity (288 GB unmatched), price-performance (30-50% cheaper per GPU-hour), open-source stack (ROCm is fully open).  
**Where NVIDIA wins:** Software ecosystem (15+ years of CUDA), interconnect at 1000+ GPU scale (NVLink/NVSwitch), compiler/profiling maturity, ecosystem lock-in.

---

## Quick Reference: What the GPU Specs Actually Mean

**Bandwidth (TB/s)** -- How fast the GPU reads/writes data from its own HBM memory. Like the bus width between a chip's memory controller and SRAM. LLM inference is memory-bandwidth-bound: the bottleneck is feeding weights to compute cores, not the math. Higher bandwidth = faster inference. During inference, the GPU reads the entire model's weights from memory for every token it generates. The speed limit is how fast it can read, not how fast it can compute. MI300X moves data ~1.6x faster than H100.

**FP16 / FP8 TFLOPS** -- Raw compute throughput: trillions of multiply-add operations per second. FP16 (16-bit) is standard for inference. FP8 (8-bit) packs 2x more ops -- used for quantized training/inference with slight accuracy tradeoff. MI355X's native FP8 is a key summit talking point.

**TDP (Watts)** -- Max power draw. 700W+ requires liquid cooling. A rack of 8 GPUs at 750W = 6 kW just for GPUs. B200 at 1,000W vs. MI355X at ~750W means AMD delivers similar TFLOPS at 25% less power -- real operational cost advantage. (This is why PG&E has a speaker.)

**Interconnect** -- How GPUs communicate within/across servers (like NoC connecting cores in a multi-core SoC).
- *NVLink (NVIDIA):* Proprietary, 1.8 TB/s per GPU on NVLink 5. NVSwitch extends across nodes.
- *Infinity Fabric (AMD):* Open, lower bandwidth at scale. Originally AMD CPU on-die interconnect, extended to GPUs.
- Multi-node training at 256+ GPUs is where interconnect becomes the dominant bottleneck -- NVIDIA's current advantage.

**Rule of thumb for conversations:**
- **Inference talk?** Focus on bandwidth + memory capacity (AMD strength)
- **Training at scale?** Focus on interconnect + TFLOPS (NVIDIA advantage, AMD catching up)
- **Cost/TCO talk?** Focus on TDP + price-per-TFLOPS (AMD advantage)

**Questions to ask based on these specs:**
- To AMD: "MI355X bandwidth is listed at 6 TB/s vs. B200 at 8 TB/s. Does the extra HBM capacity compensate for the bandwidth gap in practice for large model inference?"
- To TensorWave: "For inference-heavy workloads, is the 192 GB MI300X still the sweet spot, or should teams wait for MI355X?"
- To SemiAnalysis: "When you calculate TCO, how much does the 25% power difference between MI355X and B200 matter over a 3-year server lifecycle?"
- To Credo: "At what cluster scale does Infinity Fabric's bandwidth gap vs. NVLink start to materially hurt training throughput?"

---

## Quick Reference: AMD vs. NVIDIA Terminology

| NVIDIA         | AMD Equivalent       |
|----------------|----------------------|
| CUDA           | HIP / ROCm          |
| cuDNN          | MIOpen               |
| NCCL           | RCCL                 |
| TensorRT       | MIGraphX             |
| Nsight Compute | Omniperf / rocprof   |
| A100/H100/B200 | MI250X/MI300X/MI355X |
| NVLink         | Infinity Fabric Link |
| cuBLAS         | rocBLAS              |
| Triton (NVIDIA)| Triton (AMD backend) |

---

## Post-Event

- Write up key takeaways in this folder
- Note any contacts made and follow up within 48 hours
- Evaluate if AMD GPUs are worth exploring
