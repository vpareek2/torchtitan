# SWA + Efficient Memory: Project Driver

*A high-level thesis for local exact attention plus long-horizon compressed memory*  
Working Draft - May 13, 2026

## Executive summary

This project explores whether long-context language models should be built around two complementary mechanisms: **exact local attention** over a sliding window, and **efficient long-horizon memory** for information outside that window.

The central intuition is that dense token-to-token attention is most valuable locally, where precise interactions drive syntax, binding, code structure, recent reasoning, and short-range recall. Older context often behaves less like a set of tokens that all need pairwise comparison and more like persistent state: facts, entities, goals, plans, summaries, and latent continuity. A model may therefore benefit from an architecture that treats local context as high-resolution working memory and long context as compressed memory.

We are not trying to prove that global dense attention is never useful. We are trying to determine whether it should be the default backbone for long-context modeling, or whether it can become an optional escape hatch while sliding-window attention and efficient memory carry most of the work.

## Project thesis

**Hypothesis:** A hybrid model that combines sliding-window attention (SWA) with an efficient recurrent, linear, or state-space memory path can deliver a better long-context quality/efficiency tradeoff than either SWA-only models or efficient-memory-only models. The hybrid should preserve high-quality local reasoning while extending useful context through a cheaper, compressed memory mechanism.

**Plain-language version:** Use exact attention where exact attention matters most. Use efficient memory where long context mostly needs continuity, state, and fuzzy recall.

## Core idea

- **SWA is the local workspace.** It gives the model a precise, high-bandwidth view of recent tokens. This is where exact token interactions, local reasoning, syntax, code structure, recent tool outputs, and formatting should be handled.
- **Efficient memory is the long-horizon carrier.** Linear attention, gated recurrence, delta-style memory, or state-space mechanisms can compress information beyond the local window into a lower-cost state.
- **The hybrid is a memory hierarchy.** Rather than treating all context as equally deserving of dense attention, the model should separate high-resolution recent context from lower-resolution historical state.
- **Exact global retrieval can remain an escape hatch.** The goal is not dogmatic removal of full attention. The goal is to see whether full/global mechanisms can be reserved for cases where exact long-range lookup is truly needed.

## Why this matters

- **Long context is increasingly central.** Models are being asked to read codebases, transcripts, books, documents, logs, notebooks, plans, and long conversations.
- **The cost of naive context scaling is high.** Dense attention over very long sequences increases compute and memory pressure, especially during prefill and with large KV caches during inference.
- **Most useful behavior may not require full all-pairs attention.** Many tasks need exact local processing plus persistent global state, not constant dense comparison between every old token and every new token.
- **A clean hybrid could improve both research and product tradeoffs.** If the architecture works, it could support longer contexts, lower serving cost, better streaming behavior, and more flexible memory-aware modeling.

## What we are trying to prove

1. **Role separation works.** SWA can handle precise local computation while efficient memory handles longer-range continuity and state.
2. **The hybrid beats its parts.** The combined model should outperform SWA-only outside the window and outperform efficient-memory-only on local fidelity and short-context quality.
3. **The memory path is actually used.** The model should not simply lean on SWA and ignore the efficient branch. We need evidence that information is written into, retained by, and read from the long-memory path.
4. **The efficiency gain is real.** The model should show a materially better tradeoff between quality and runtime cost, especially as context length grows.
5. **The design is scalable enough to justify investment.** Early results should give us confidence that the approach is worth scaling, even if the first versions are not final or competitive models.

## What we are not trying to prove yet

- That one specific efficient mechanism is the final answer.
- That full/global attention is useless in every setting.
- That the first architecture variant will be the production architecture.
- That one benchmark can decide the project.
- That we need to lock model size, training framework, layer ratios, or kernels before the hypothesis is validated.

## Claims and evidence needed

| Claim | Evidence we need |
|---|---|
| SWA is enough for high-fidelity local computation. | Comparable local language quality, syntax, code behavior, formatting, and short-range reasoning versus strong attention baselines. |
| Efficient memory can carry useful long-range state. | Better beyond-window behavior than SWA-only, plus diagnostics showing memory state stores and exposes relevant information. |
| The hybrid is better than either branch alone. | Consistent improvements over SWA-only and efficient-only baselines under the same training and compute constraints. |
| The architecture gives real efficiency benefits. | Lower KV/cache pressure, better throughput, lower memory use, or better context scaling at comparable quality. |
| The approach remains practical after training and adaptation. | Stable training, compatible post-training behavior, and no hidden complexity that erases the benefit. |

## Evaluation approach

Evaluation should be comparative, capability-separated, and efficiency-aware. We should avoid asking a single aggregate benchmark to answer the whole question. The architecture is making a claim about roles, so the evaluation needs to test those roles independently.

### 1. Compare against the right baselines

- A strong dense or global-attention reference where feasible.
- An SWA-only model to test whether the memory branch adds value.
- An efficient-memory-only model to test whether local exact attention adds value.
- One or more hybrid variants to test whether the design choices matter.

### 2. Separate capability classes

- **Local quality.** Does the model preserve short-context language quality, code quality, syntax, formatting, and local reasoning?
- **Long-horizon continuity.** Does the model maintain facts, entities, goals, and state beyond the SWA window?
- **Exact retrieval.** Where does the compressed-memory approach fail, and when is an exact retrieval or global-attention escape hatch needed?
- **Document and conversation understanding.** Can the model use earlier sections, long-running threads, and repeated references across a long input?
- **Instruction-following behavior.** Does the architecture remain usable after adaptation, or do long-memory benefits disappear?
- **Runtime behavior.** How does quality change as we increase context length, batch size, decode length, and serving constraints?

### 3. Measure utilization, not just loss

A key risk is that the model learns to use SWA for everything it can and ignores the efficient-memory path. Evaluation therefore needs probes that make memory usage visible. Examples include branch ablations, state resets, window stress tests, memory-state diagnostics, and controlled tasks where the answer depends on information outside the local window.

### 4. Treat efficiency as a first-class metric

The project should evaluate not only whether the model is good, but whether it is good at the right cost. Relevant measurements include memory footprint, KV/cache behavior, prefill throughput, decode throughput, scaling with context length, training stability, and implementation complexity. An architecture that wins on quality but loses the efficiency story does not validate the core thesis.

## What success looks like

- The hybrid retains strong local quality instead of degrading into a weak efficient-attention model.
- The hybrid handles beyond-window dependencies better than SWA-only models.
- The efficient-memory path demonstrably carries useful information and is not a decorative component.
- The architecture delivers a meaningful quality/efficiency tradeoff as context grows.
- The results are robust enough to justify scaling the architecture and investing in better kernels, training recipes, and post-training workflows.

## What would weaken or falsify the thesis

- The memory branch is consistently ignored or only helps on artificial tasks.
- SWA-only models match the hybrid on all meaningful long-context evaluations.
- Efficient-memory-only models match the hybrid on local quality, making SWA unnecessary.
- Exact long-context retrieval failures are too common to avoid frequent global attention.
- Training or implementation complexity erases the expected efficiency gains.
- The architecture works at small scale but breaks down or loses its advantage as it scales.

## Decision gates

1. **Trainability gate.** Can the hybrid train stably and reach reasonable quality compared with simple baselines?
2. **Utility gate.** Does the efficient-memory path improve beyond-window behavior in a measurable way?
3. **Utilization gate.** Can we show the long-memory branch is being used, not bypassed?
4. **Efficiency gate.** Do the runtime and memory advantages show up under realistic sequence lengths and serving conditions?
5. **Scale gate.** Are the early results strong enough to justify a larger run and more serious systems work?

## Open design questions

The project should keep the following choices open until evidence points toward a clear direction:

- Whether to interleave SWA and memory as separate layers or combine them within a layer.
- Which efficient-memory mechanism is most appropriate.
- How often the model should use SWA versus long-memory blocks.
- How large the local window should be.
- Whether the model needs occasional global attention, retrieval, summary tokens, or another exact-recall escape hatch.
- Whether training needs explicit pressure to use memory, such as window dropout, state stress tests, or auxiliary objectives.
- How to expose diagnostics that make memory usage visible during training and evaluation.

## Guiding principles

- **Keep the hypothesis clean.** The core question is about the division of labor between local exact attention and efficient long-memory state.
- **Do not overfit to one mechanism too early.** The first implementation should be a vehicle for testing the hypothesis, not a premature commitment to one memory family.
- **Make failure informative.** Every experiment should tell us whether the problem is local fidelity, long-memory capacity, memory utilization, training stability, or systems overhead.
- **Instrument the model.** We should design for observability from the start: memory norms, gates, state resets, branch ablations, and context stress tests.
- **Scale only after signal.** The project should earn each larger run by proving something specific at the previous stage.

## Working project narrative

The project is based on a simple architectural belief: long-context models should not treat every old token as if it belongs in the same high-resolution workspace as the current local context. The model should attend precisely to what is nearby, remember efficiently what is far away, and use exact global mechanisms only when the task truly demands them. We will test whether this memory hierarchy can preserve model quality while improving long-context efficiency and making the long-memory path a useful part of the model rather than an afterthought.

