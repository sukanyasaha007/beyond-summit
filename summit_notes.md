# Beyond Summit 2026 -- Live Note-Taking Templates

Use these templates during the event to capture notes quickly and consistently. Keep entries short -- you can expand later.

---

## Speaker Note Template

Use this for each talk or breakout session:

```
SPEAKER: [Name] | [Company]
TOPIC: [What they're talking about]

Key Claims:
  - Claim 1 (1-2 line summary)
  - Claim 2

Sharp Edges / Gaps:
  - What didn't they answer?
  - What claim seemed overblown?

Questions I Should Ask:
  - [Your follow-up if you get a chance]

Follow-up: [Name, email/LinkedIn if exchanged, what to send]
```

---

## Organize By Theme (Not Chronologically)

Instead of writing everything sequentially, create sections for each topic and fill them in as you hear relevant info:

### ROCm Maturity Status
(Capture from: AMD, Modular, Triton mentions)

- Flash-attention status: ___
- GPTQ/AWQ support: ___
- Driver stability claims: ___
- FP8 production readiness: ___

### Interconnect / Training at Scale
(Capture from: AMD, Credo, TensorWave, Zyphra)

- RCCL performance at scale: ___
- Anyone mention 256+ GPU training?: ___
- Claimed all-reduce time: ___

### Inference & Serving
(Capture from: TensorWave, Liquid AI, Featherless AI, Artificial Analysis)

- vLLM vs. SGLang maturity on AMD: ___
- ScalarLM advantages: ___
- Real throughput numbers (tokens/sec): ___
- Quantization that works best: ___

### Portability / CUDA → AMD
(Capture from: ZLUDA, SCALE, Modular, AMD)

- ZLUDA production readiness: ___
- SCALE compile success rate: ___
- Triton adoption rate: ___

### EDA / Chip Design AI
(Capture from: Anyone mentioning this)

- Does anyone use AMD for ML on chip design?: ___
- ChipNeMo vs. nothing from AMD: ___
- VerilogEval on AMD?: ___

### Cost / TCO
(Capture from: SemiAnalysis, TensorWave)

- Real per-GPU-hour cost AMD vs. NVIDIA: ___
- Power cost impact over 3 years: ___
- Cloud provider pricing data: ___

---

## Metrics Tracker

Whenever you hear **specific numbers**, write them down here:

| Metric | Value | Speaker | Context |
|--------|-------|---------|---------|
| Tokens/sec on [model] | ___ | ___ | [benchmark] |
| All-reduce time @ 256 GPUs | ___ | ___ | |
| RCCL vs. NCCL speedup | ___ | ___ | |
| FP8 accuracy loss (%) | ___ | ___ | |
| Driver stability uptime (%) | ___ | ___ | |
| Time-to-first-token (ms) | ___ | ___ | [model size] |
| Cost per GPU-hour | ___ | ___ | [provider] |

---

## Red Flags / Skepticism Checklist

Mark these if you hear them (they often signal overstatement or marketing speak):

- [ ] "We're competitive with NVIDIA" (but no benchmarks given)
- [ ] "ROCm is production-ready" (ask: at what scale? what's missing?)
- [ ] "Flash-attention just works" (does it work on any model or just transformers?)
- [ ] "No performance gap" (at what batch size? what model?)
- [ ] "Cost per token is lower" (for what inference task? what quantization?)

---

## Networking Contact Template

When you exchange contact info:

```
NAME: [Full name]
COMPANY: [Company]
ROLE: [Their role]
CONTACT: [Email or LinkedIn]

CONVO TOPICS:
  - [What you discussed]
  - [Their area of expertise]

TO SEND: [What you promised to share]
FOLLOW-UP: [When to reach out]
```

---

## Session-End Checklist

At the end of each talk or breakout:

- [ ] Did I capture the speaker's main claim?
- [ ] Did I note any unanswered questions?
- [ ] Are there specific numbers I should follow up on?
- [ ] Is there someone in the audience I should talk to?

---

## Pro Tips for Tomorrow

- **Circulate during breaks.** The best info is informal. Ask AMD engineers (not just VPs): "What broke when you tried to run your internal models on MI300X?"
- **Ask the same question to multiple people.** If both TensorWave and Zyphra say "RCCL works fine at 64 GPUs, untested at 256+," that's a consistent signal.
- **Capture contradictions.** If AMD says "FP8 is production-ready" but TensorWave says "still has numerical stability issues," that's worth noting.
- **Take photos of slides** if allowed. Easier to expand notes later than try to remember exact specs.

---

## What to Focus On

1. **What bridges the gap?** (ZLUDA works? Triton mature? Quantization helps?)
2. **Where are the sharp edges?** (Driver stability? RCCL? Custom kernels?)
3. **What's the real cost math?** (Is 20% slower worth 40% cheaper for your workload?)
4. **Who's actually using AMD at scale?** (Names, companies, sizes)

**Don't get lost in:** Hype, marketing claims without numbers, vendor posturing.

---

## Post-Event Consolidation (That Night)

Spend 30 mins after the event organizing your notes:

1. **Consolidate into detailed notes** per speaker
2. **Update metrics tracker** with all numbers collected
3. **Flag follow-ups** -- list promises you made
4. **Commit to repo** with summary of key findings

---

## File Structure for Notes

```
Beyond Summit/
  └── event-notes/
      ├── 2026-04-08_session_morning_keynotes.md
      ├── 2026-04-08_speaker_AMD_anush.md
      ├── 2026-04-08_speaker_TensorWave_greg.md
      ├── 2026-04-08_networking_contacts.md
      └── metrics_collected.md
```

---

## Quick Reference: Questions to Ask

**To AMD (Anush, Neha):**
- Flash-attention status on MI355X?
- RCCL performance at 256+ GPU scale?
- FP8 production-readiness?
- EDA AI strategy vs. NVIDIA's ChipNeMo?

**To TensorWave (Greg Diamos, Darrick):**
- ScalarLM vs. vLLM benchmarks?
- MI355X availability date?
- Real training/inference numbers?

**To SemiAnalysis (Dylan Patel):**
- Actual AMD vs. NVIDIA TCO at 256+ GPUs?
- Where does AMD win/lose most?

**To anyone:**
- "What broke when you ported to AMD?"
- "Largest cluster you've run: size? performance?"
- "Which CUDA libraries were hard to replace?"
