# EDA & Chip Design AI -- Quick Review

---

## Why EDA Matters at a GPU Summit

You work on AI-powered verification tooling. This summit is about GPU infrastructure, not EDA directly -- but there are real angles:

1. **NVIDIA is investing in EDA AI** (ChipNeMo, VerilogEval). AMD is not. This is a competitive gap worth probing.
2. **Fine-tuning on Verilog/RTL data** requires GPU compute. Understanding AMD's fitness for this workload is directly relevant.
3. **Smaller specialized models** (embeddings, classifiers for coverage analysis) are a different GPU workload profile than large LLM inference. Worth asking if AMD handles these well.
4. **AMD designs chips.** They must have massive verification challenges. Whether they dogfood their own GPUs for internal AI is an interesting question.

---

## EDA Jargon -- What You Need to Explain If Asked

| Term | Plain English |
|------|--------------|
| **RTL** | Register Transfer Level. The Verilog/VHDL source code that describes what a chip does. The "source code" of hardware. |
| **DUT** | Design Under Test. The specific chip block being verified. |
| **Coverage closure** | Getting your test suite to hit every scenario in the verification plan before tape-out. Filling a checklist where every box must be ticked. |
| **Regression** | Re-running all tests after a code change to check nothing broke. |
| **Formal verification** | Mathematically proving a design property holds for all inputs, without running simulations. |
| **State space** | All possible combinations of inputs and internal states a chip can be in. For real chips, astronomically large. |
| **Tape-out** | The point where the chip design is finalized and sent to manufacturing. No more changes. |
| **Testbench** | Code that drives inputs into the DUT and checks outputs. The "test harness" for chip verification. |
| **UVM** | Universal Verification Methodology. Industry-standard framework (SystemVerilog) for building testbenches. |
| **RAG** | Retrieval-Augmented Generation. Pulling relevant documents (specs, coverage reports) and feeding them to an LLM as context. Your primary use case. |

---

## Verilog/EDA AI Benchmarks -- The Landscape

### NVIDIA Has EDA-Specific AI. AMD Has None.

| Benchmark / Model | Source | What It Is |
|-------------------|--------|-----------|
| **VerilogEval v1/v2** | NVlabs | 156 Verilog coding problems with testbenches. Code-completion + spec-to-RTL. Industry standard for evaluating LLM Verilog generation. `github.com/NVlabs/verilog-eval` |
| **VGen** | NYU | Fine-tuned CodeGen model on Verilog. Uses the `shailja/Verilog_GitHub` dataset (109k Verilog modules). `github.com/shailja-thakur/VGen` |
| **ChipNeMo** | NVIDIA | NVIDIA's internal LLM fine-tuned on proprietary chip design data (RTL, specs, bug reports). Paper published, model NOT open-sourced. Shows 2-3x improvement on domain tasks over general LLMs. |
| **RTLCoder** | Various | Open-source attempts at Verilog generation models. Smaller, less established. |

**AMD's position:** Zero public EDA benchmarks, zero EDA-specific models, zero published research on AI for chip design. Their strategy appears to be: "We are a compute platform. EDA AI is the EDA vendors' problem." This is worth probing.

### Available Datasets

| Dataset | Size | What It Contains |
|---------|------|-----------------|
| `shailja/Verilog_GitHub` | 109k modules | Scraped Verilog from GitHub. Used by VGen. Largest public Verilog corpus. |
| VerilogEval | 156 problems | Curated problems with testbenches. Evaluation benchmark, not training data. |

---

## Your Use Case Mapped to GPU Workloads

| What You Do | GPU Workload Profile | AMD Relevance |
|-------------|---------------------|---------------|
| **RAG over specs/coverage** | Embedding model inference (small model, high throughput) | Good fit for AMD -- small models fit in HBM, bandwidth helps throughput |
| **LLM generation** (answering engineer questions) | LLM inference (memory-bandwidth-bound) | AMD's strongest story -- more HBM, competitive bandwidth |
| **Fine-tuning on Verilog/domain data** | Training (compute + memory) | Possible on AMD, but distributed training at scale is NVIDIA's advantage |
| **Coverage analysis / classification** | Small model inference or training | Fits easily on any modern GPU |

---

## Angles to Probe at the Summit

### To AMD (Anush Elangovan, Neha Prakriya)

- NVIDIA published ChipNeMo and VerilogEval for chip design AI. Is AMD planning anything similar, or leaving it to EDA vendors building on ROCm?
- Does AMD's own chip design team run ML training on Instinct GPUs with ROCm? (Dogfooding test)
- Coverage closure at AMD's scale must be a massive problem. How does the verification team handle it? (Indirect probe -- see if they mention AI)
- How do embedding model workloads perform on MI355X vs. LLM inference? (Directly relevant to RAG)
- What is the state of ROCm support for smaller, specialized models that could run inference in EDA pipelines?
- (For Neha) When AMD trains models internally, do you use MI300X or NVIDIA?

### To TensorWave

- Has anyone used your clusters for EDA AI -- fine-tuning on Verilog/RTL code?
- Is TensorWave cost-effective for smaller workloads like embedding generation and RAG, or only for large training?

### To Eric Hartford (Quixi AI / Cognitive Computations)

- Has anyone in the open-source community tried running VerilogEval on AMD hardware?
- What is the experience fine-tuning code-generation models (not just chat) on AMD GPUs?

### To Liquid AI (Mathias Lechner)

- Could liquid neural networks / state-space models be better than transformers for ingesting long RTL specifications? (Long-context efficiency angle)

### To Anyone

- "NVIDIA is the only GPU vendor investing in AI for chip design. Is that a strategic advantage, or does it not matter because the EDA vendors will build on whatever hardware is available?"

---

## ChipNeMo -- What You Need to Know

NVIDIA's internal project. Key facts:

- Fine-tuned LLaMA 2 (13B and 70B) on NVIDIA's proprietary chip design data: RTL code, design specs, bug reports, internal documentation.
- Domain-adapted with three techniques: domain-adaptive pre-training, supervised fine-tuning, and RAG.
- **Results:** 2-3x improvement on chip design tasks (code generation, bug summarization, spec Q&A) vs. general-purpose LLMs.
- **Not open-sourced.** Paper only. You cannot use the model.
- **Why it matters at the summit:** It demonstrates that domain-specific fine-tuning on EDA data yields significant gains. The question is whether the open-source community (or AMD) will produce something equivalent for non-NVIDIA hardware.

---

## Your Elevator Pitch (If EDA Comes Up)

"I work on AI-powered verification tooling for chip design -- using LLMs and RAG to help engineers navigate specifications and coverage data. I am here to understand how AMD's compute stack is maturing for production AI workloads, including smaller specialized models for EDA."

---

## Key Takeaway

Nobody at this summit is an EDA expert except possibly the AMD chip design engineers in the audience (not on stage). Your EDA knowledge is an edge in conversations. Use it to ask differentiated questions that nobody else at the event will think to ask.