# Awesome-Efficient-VLA [![Awesome](https://awesome.re/badge.svg)](https://awesome.re)

[![arXiv](https://img.shields.io/badge/arXiv-2510.17111-b31b1b.svg)](https://arxiv.org/pdf/2510.17111)
[![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-green.svg)](https://github.com/guanweifan/awesome-efficient-vla/commits/main/)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=flat)](https://github.com/guanweifan/awesome-efficient-vla/pulls)
[![Wiki](https://img.shields.io/badge/Wiki-efficient--vla--wiki-1677FF.svg)](https://github.com/guanweifan/efficient-vla-wiki)

<p align="center">
  <img src="./imgs/overview.png" width="100%" height="80%">
</p>

📖 Survey: A live-updated hub for **[Efficient VLA for Embodied Manipulation](https://arxiv.org/pdf/2510.17111)** across architecture, perception, action & pipeline, providing real-time updates on the fast-moving VLA field.

🚀 Updates: Weekly / Bi-weekly. Contributions & ⭐ are welcome to help researchers navigate the field!

> [!IMPORTANT]
> **✨ Want a more structured way to study this field?**  
> Try [efficient-vla-wiki](https://github.com/guanweifan/efficient-vla-wiki). It turns this paper list into an interactive research workspace where you can ask focused questions, compare methods across papers, and keep building your own understanding of the field.

---

## Quick Navigation

- [Latest Updates](#latest-updates)
- [Efficiency Bottleneck Map](#efficiency-bottleneck-map)
- [Complete Paper List](#complete-paper-list)
- [Related Surveys](#related-surveys)
- [Citation](#citation)
- [Star History](#star-history)
- [Classification Logic](#classification-logic)

## 🔥 Latest Updates (2026-04-14 to 2026-04-27)

- **1.1 Static Backbone Selection:** [PokeVLA](#static-backbone-selection)
- **3.1 Raw Action Generation:** [SpanVLA](#raw-action-generation), [FASTER: Value-Guided Sampling](#raw-action-generation)
- **3.2 Reasoning-Aware Action Generation:** [OneVL](#reasoning-aware-action-generation)
- **4.1 Training Efficiency Techniques:** [DA-PTQ](#training-efficiency-techniques)

See the full list in the corresponding sections below.

## Efficiency Bottleneck Map

Start from the efficiency problem you care about, then jump to the corresponding taxonomy section. The complete paper list below keeps code links, venue notes, secondary relevance, and domain tags.

| If you care about... | Go to | Typical signals | Papers |
|---|---|---|---:|
| Smaller model backbone | [1.1 Static Backbone Selection](#static-backbone-selection) | compact VLM, small backbone, lightweight policy | 7 |
| Skipping unnecessary computation | [1.2 Dynamic Computation Pathways](#dynamic-computation-pathways) | routing, layer skipping, early exit, adaptive depth | 6 |
| Slow reasoning with fast control | [1.3 Dual-system Design](#dual-system-design) | dual-system policy, memory, fast controller | 10 |
| Fewer visual tokens | [2.1 Selective Feature Processing](#selective-feature-processing) | pruning, merging, salience selection, compression | 15 |
| Reusing temporal context | [2.2 Temporal Sharing and Reuse](#temporal-sharing-and-reuse) | history fusion, KV cache, feature reuse | 8 |
| Faster action decoding | [3.1 Raw Action Generation](#raw-action-generation) | action tokenizer, chunking, diffusion / flow, parallel decoding | 23 |
| Cheaper reasoning before action | [3.2 Reasoning-Aware Action Generation](#reasoning-aware-action-generation) | text CoT, latent CoT, visual subgoal, world dynamics | 9 |
| Cheaper adaptation or compression | [4.1 Training Efficiency Techniques](#training-efficiency-techniques) | distillation, RL, data selection, PTQ / QAT | 10 |
| Real-time deployment or evaluation | [4.2 Inference Efficiency Techniques](#inference-efficiency-techniques) | streaming, scheduling, edge deployment, metrics | 25 |

---

<a id="complete-paper-list"></a>
## Complete Paper List

Tag notes: `Code` links to an open-source repository; `Venue` marks accepted conference or journal versions; `Sec.` marks secondary relevance; `AD` means autonomous driving; `VLN` means vision-language navigation. Long sections show recent or representative papers first and fold older entries to keep the page scannable.

<a id="efficient-model-architecture"></a>
## 1. Efficient Model Architecture

Reduce structural redundancy inside the model itself through smaller backbones, adaptive computation paths, or slow-fast system decomposition.

<a id="static-backbone-selection"></a>
### 1.1 Static Backbone Selection

- [**PokeVLA: Empowering Pocket-Sized Vision-Language-Action Model with Comprehensive World Knowledge Guidance**](https://arxiv.org/pdf/2604.20834) · 🔥 New `2026-04` · Pocket-sized VLA with a lightweight embodied-aware VLM and spatial-semantic guidance.
- [**Evo-1: Lightweight Vision-Language-Action Model with Preserved Semantic Alignment**](https://arxiv.org/pdf/2511.04555) · `2025-11` · Compact multimodal backbone with a cross-modulated diffusion transformer. <sub>[Code](https://github.com/MINT-SJTU/Evo-1) · Sec. 4.1</sub>
- [**FLOWER: Democratizing Generalist Robot Policies with Efficient Vision-Language-Action Flow Policies**](https://arxiv.org/pdf/2509.04996) · `2025-09` · Intermediate-modality fusion with LLM layer pruning. <sub>[Code](https://github.com/intuitive-robots/flower_vla_calvin) · Venue: CoRL 2025</sub>
- [**SmolVLA: A vision-language-action model for affordable and efficient robotics**](https://arxiv.org/pdf/2506.01844) · `2025-06` · Single-GPU training with asynchronous inference. <sub>[Code](https://github.com/huggingface/lerobot) · Sec. 4.2</sub>
- [**NORA: A SMALL OPEN-SOURCED GENERALIST VISION-LANGUAGE ACTION MODEL FOR EMBODIED TASKS**](https://arxiv.org/pdf/2504.19854) · `2025-04` · Qwen-2.5-VL-3B backbone with a FAST+ tokenizer. <sub>[Code](https://github.com/declare-lab/nora)</sub>
- [**TinyVLA: Towards Fast, Data-Efficient Vision-Language-Action Models for Robotic Manipulation**](https://arxiv.org/pdf/2409.12514) · `2024-09` · High-speed multimodal backbone with a diffusion policy decoder. <sub>[Code](https://github.com/liyaxuanliyaxuan/TinyVLA) · Venue: RAL 2025 · Sec. 4.1</sub>
- [**RoboMamba: Efficient Vision-Language-Action Model for Robotic Reasoning and Manipulation**](https://arxiv.org/pdf/2406.04339) · `2024-06` · Mamba state-space backbone for linear-complexity reasoning and efficient inference. <sub>[Code](https://github.com/lmzpai/roboMamba) · Venue: NeurIPS 2024 · Sec. 4.1</sub>

<a id="dynamic-computation-pathways"></a>
### 1.2 Dynamic Computation Pathways

- [**Act, Think or Abstain: Complexity-Aware Adaptive Inference for Vision-Language-Action Models (ActThinkAbstain)**](https://arxiv.org/pdf/2603.05147) · `2026-03` · Adaptive routing between direct acting, deeper reasoning, and abstention.
- [**DySL-VLA: Efficient Vision-Language-Action Model Inference via Dynamic-Static Layer-Skipping for Robot Manipulation**](https://arxiv.org/pdf/2602.22896) · `2026-02` · Dynamic-static layer skipping that prioritizes action-critical layers. <sub>[Code](https://github.com/PKU-SEC-Lab/DYSL_VLA) · Sec. 4.1</sub>
- [**Environment-Aware Adaptive Pruning with Interleaved Inference Orchestration for Vision-Language-Action Models (EcoVLA)**](https://arxiv.org/pdf/2602.00780) · `2026-02` · Environment-aware adaptive channel pruning with interleaved orchestration. <sub>Sec. 4.2</sub>
- [**NANOVLA: ROUTING DECOUPLED VISION-LANGUAGE UNDERSTANDING FOR NANO-SIZED GENERALIST ROBOTIC POLICIES**](https://arxiv.org/pdf/2510.25122) · `2025-10` · Lightweight VLA with dynamic routing. <sub>Sec. 1.2</sub>
- [**MoLe-VLA: Dynamic Layer-skipping Vision Language Action Model via Mixture-of-Layers for Efficient Robot Manipulation**](https://arxiv.org/pdf/2503.20384) · `2025-03` · Mixture-of-Layers VLA with a spatial-temporal router. <sub>[Code](https://github.com/RoyZry98/MoLe-VLA-Pytorch) · Venue: AAAI 2026</sub>
- [**DeeR-VLA: Dynamic Inference of Multimodal Large Language Models for Efficient Robot Execution**](https://arxiv.org/pdf/2411.02359) · `2024-11` · Multi-exit MLLM-based VLA with resource-aware early termination. <sub>[Code](https://github.com/yueyang130/DeeR-VLA) · Venue: NeurIPS 2024 · Sec. 4.2</sub>

<a id="dual-system-design"></a>
### 1.3 Dual-system Design

- [**TacMamba: A Tactile History Compression Adapter Bridging Fast Reflexes and Slow VLA Reasoning**](https://arxiv.org/pdf/2603.01700) · `2026-03` · Tactile history compression for high-frequency tactile control.
- [**StreamVLA: Breaking the Reason-Act Cycle via Completion-State Gating**](https://arxiv.org/pdf/2602.01100) · `2026-02` · Lock-and-gated selective slow reasoning. <sub>Sec. 4.2</sub>
- [**Video2Act: A Dual-System Video Diffusion Policy with Robotic Spatio-Motional Modeling**](https://arxiv.org/pdf/2512.03044) · `2025-12` · Dual-system VLA with slow VDM reasoning. <sub>[Code](https://github.com/jiayueru/Video2Act)</sub>
- [**MEMER: SCALING UP MEMORY FOR ROBOT CONTROL VIA EXPERIENCE RETRIEVAL**](https://arxiv.org/pdf/2510.20328) · `2025-10` · Memory-aware high-level policy selecting keyframes for low-level execution. <sub>Venue: ICLR 2026 · Sec. 2.2</sub>
- [**MOTVLA: A VISION-LANGUAGE-ACTION MODEL WITH UNIFIED FAST-SLOW REASONING**](https://arxiv.org/pdf/2510.18337) · `2025-10` · Mixture-of-Transformers VLA with unified fast-slow reasoning.
- [**Fast-in-Slow: A Dual-System Foundation Model Unifying Fast Manipulation within Slow Reasoning (FiS-VLA)**](https://arxiv.org/pdf/2506.01953) · `2025-06` · Fast execution module inside a VLM-based reasoning model. <sub>[Code](https://github.com/CHEN-H01/Fast-in-Slow) · Venue: NeurIPS 2025</sub>
- [**Hume: Introducing System-2 Thinking in Visual-Language-Action Model**](https://arxiv.org/pdf/2505.21432) · `2025-05` · Value-guided slow thinking with lightweight reactive action denoising. <sub>[Code](https://github.com/hume-vla/hume)</sub>
- [**OPENHELIX: A Short Survey, Empirical Analysis, and Open-Source Dual-System VLA Model for Robotic Manipulation**](https://arxiv.org/pdf/2505.03912) · `2025-05` · Systematic structural evaluation with a low-cost design. <sub>[Code](https://github.com/OpenHelix-Team/OpenHelix)</sub>
- [**TOWARDS SYNERGISTIC, GENERALIZED AND EFFICIENT DUAL-SYSTEM FOR ROBOTIC MANIPULATION (RoboDual)**](https://arxiv.org/pdf/2410.08001) · `2024-10` · Generalist reasoning with a lightweight specialist diffusion policy.
- [**HiRT: Enhancing Robotic Control with Hierarchical Robot Transformers**](https://arxiv.org/pdf/2410.05273) · `2024-10` · Low-frequency VLM reasoning with high-frequency vision-based control. <sub>Venue: CoRL 2024</sub>

<a id="efficient-perception-feature"></a>
## 2. Efficient Perception Feature

Reduce spatial and temporal redundancy in visual representations through token filtering, compression, reuse, or caching.

<a id="selective-feature-processing"></a>
### 2.1 Selective Feature Processing

- [**2D or 3D: Who Governs Salience in VLA Models? Tri-Stage Token Pruning Framework with Modality Salience Awareness**](https://arxiv.org/pdf/2604.09244) · `2026-04` · Tri-stage 2D / 3D token pruning with modality salience awareness.
- [**VLA-InfoEntropy: A Training-Free Vision-Attention Information Entropy Approach for Vision-Language-Action Models Inference Acceleration and Success**](https://arxiv.org/pdf/2604.05323) · `2026-04` · Training-free dynamic token selection using visual entropy, attention entropy, and timestep cues.
- [**VLA-IAP: Training-Free Visual Token Pruning via Interaction Alignment for Vision-Language-Action Models**](https://arxiv.org/pdf/2603.22991) · `2026-03` · Geometric interaction anchors with semantic-motion pruning schedules.
- [**BFA++: Hierarchical Best-Feature-Aware Token Prune for Multi-View Vision Language Action Model**](https://arxiv.org/pdf/2602.20566) · `2026-02` · Hierarchical multi-view token pruning.
- [**Think Proprioceptively: Embodied Visual Reasoning for VLA Manipulation (ThinkProprio)**](https://arxiv.org/pdf/2602.06575) · `2026-02` · Early proprioception fusion for aggressive visual token reduction.

<details>
<summary>More selective feature processing papers</summary>

- [**DTP:A SIMPLE YET EFFECTIVE DISTRACTING TOKEN PRUNINGFRAMEWORK FOR VISION-LANGUAGE ACTION MODELS**](https://arxiv.org/pdf/2601.16065) · `2026-01` · Distracting token pruning. <sub>[Code](https://anonymous.4open.science/r/CBD3)</sub>
- [**Token Expand-Merge: Training-Free Token Compression for Vision-Language-Action Models (TEAM-VLA)**](https://arxiv.org/pdf/2512.09927) · `2025-12` · Dynamic token expansion and merging. <sub>[Code](https://github.com/Jasper-aaa/TEAM-VLA)</sub>
- [**COMPRESSOR-VLA: INSTRUCTION-GUIDED VISUAL TOKEN COMPRESSION FOR EFFICIENT ROBOTIC MANIPULATION**](https://arxiv.org/pdf/2511.18950) · `2025-11` · Hybrid visual token compression with semantic distillation.
- [**VLA-Pruner: Temporal-Aware Dual-Level Visual Token Pruning for Efficient Vision-Language-Action Inference**](https://arxiv.org/pdf/2511.16449) · `2025-11` · Dual-objective visual token pruning. <sub>[Code](https://github.com/MINT-SJTU/VLA-Pruner)</sub>
- [**SemanticVLA: Semantic-Aligned Sparsification and Enhancement for Efficient Robotic Manipulation**](https://arxiv.org/pdf/2511.10518) · `2025-11` · Dual visual pruning with hierarchical fusion. <sub>[Code](https://github.com/JiuTian-VL/SemanticVLA) · Venue: AAAI 2026</sub>
- [**ACTION-AWARE DYNAMIC PRUNING FOR EFFICIENT VISION-LANGUAGE-ACTION MANIPULATION (ADP)**](https://arxiv.org/pdf/2509.22093) · `2025-09` · Action-aware dynamic visual token pruning. <sub>Venue: ICLR 2026</sub>
- [**The Better You Learn, The Smarter You Prune: Towards Efficient Vision-language-action Models via Differentiable Token Pruning (LightVLA)**](https://arxiv.org/pdf/2509.12594) · `2025-09` · Dynamic query-based importance estimation. <sub>[Code](https://github.com/LiAutoAD/LightVLA)</sub>
- [**SpecPrune-VLA: Accelerating VLA Models via Action-Aware Self-Speculative Pruning**](https://arxiv.org/pdf/2509.05614) · `2025-09` · Spatial-temporally consistent action-aware pruning.
- [**FastDriveVLA: Efficient End-to-End Driving via Plug-and-Play Reconstruction-based Token Pruning**](https://arxiv.org/pdf/2507.23318) · `2025-07` · Reconstruction-based foreground-aware pruning. <sub>AD</sub>
- [**Think Twice, Act Once: Token-Aware Compression and Action Reuse for Efficient Inference in VLA Models (FlashVLA)**](https://arxiv.org/pdf/2505.21200) · `2025-05` · Action reuse with information-guided visual token pruning. <sub>Sec. 2.2</sub>
</details>

<a id="temporal-sharing-and-reuse"></a>
### 2.2 Temporal Sharing and Reuse

- [**ETA-VLA: Efficient Token Adaptation via Temporal Fusion and Intra-LLM Sparsification for Vision-Language-Action Models**](https://arxiv.org/pdf/2603.25766) · `2026-03` · Temporal fusion plus intra-LLM sparse aggregation for driving inference. <sub>Sec. 2.1 · AD</sub>
- [**Beyond Short-Horizon: VQ-Memory for Robust Long-Horizon Manipulation in Non-Markovian Simulation Benchmarks (VQ-Memory)**](https://arxiv.org/pdf/2603.09513) · `2026-03` · Vector-quantized memory for long-horizon proprioceptive histories.
- [**History-Conditioned Spatio-Temporal Visual Token Pruning for Efficient Vision-Language Navigation (A-MMR)**](https://arxiv.org/pdf/2603.06480) · `2026-03` · Spatio-temporal token pruning for current views and navigation history. <sub>Sec. 2.1 · VLN</sub>
- [**FUTURE-VLA: Forecasting Unified Trajectories Under Real-time Execution**](https://arxiv.org/pdf/2602.15882) · `2026-02` · Temporally adaptive compression for long multi-view histories. <sub>[Code](https://github.com/fan-jj24/FUTURE-VLA)</sub>
- [**Efficient Long-Horizon Vision-Language-Action Models via Static-Dynamic Disentanglement (SD-VLA)**](https://arxiv.org/pdf/2602.03983) · `2026-02` · Static-dynamic token disentanglement with KV reuse.
- [**Learning to Accelerate Vision-Language-Action Models through Adaptive Visual Token Caching**](https://arxiv.org/pdf/2602.00686) · `2026-02` · Task-aware token caching with differentiable selection. <sub>[Code](https://github.com/JiahanFan/LAC) · Sec. 4.2</sub>
- [**KV-Efficient VLA: A Method of Speed up Vision Language Model with RNN-Gated Chunked KV Cache**](https://arxiv.org/pdf/2509.21354) · `2025-09` · Chunk-based KV cache compression with recurrent utility gating.
- [**VLA-Cache: Efficient Vision-Language-Action Manipulation via Adaptive Token Caching**](https://arxiv.org/pdf/2502.02175) · `2025-02` · Visual token KV caching and reuse. <sub>[Code](https://github.com/siyuhsu/vla-cache) · Venue: NeurIPS 2025</sub>

<a id="efficient-action-generation"></a>
## 3. Efficient Action Generation

Reduce redundancy in action representation, decoding, sampling, or reasoning before control.

<a id="raw-action-generation"></a>
### 3.1 Raw Action Generation

- [**FASTER: Value-Guided Sampling for Fast RL**](https://arxiv.org/pdf/2604.19730) · 🔥 New `2026-04` · VLA-adjacent value-guided candidate filtering for diffusion-policy sampling. <sub>[Code](https://github.com/alexanderswerdlow/faster)</sub>
- [**SpanVLA: Efficient Action Bridging and Learning from Negative-Recovery Samples for Vision-Language-Action Model**](https://arxiv.org/pdf/2604.19710) · 🔥 New `2026-04` · Flow-matching action expert bridged from autoregressive VLM reasoning. <sub>AD</sub>
- [**SnapFlow: One-Step Action Generation for Flow-Matching VLAs via Progressive Self-Distillation**](https://arxiv.org/pdf/2604.05656) · `2026-04` · Self-distillation from multi-step flow matching to one-step action generation.
- [**Adaptive Action Chunking at Inference-time for Vision-Language-Action Models (AAC)**](https://arxiv.org/pdf/2604.04161) · `2026-04` · Inference-time action chunking using action entropy. <sub>[Code](https://github.com/junhyukso/SGAC)</sub>
- [**AnchorVLA: Anchored Diffusion for Efficient End-to-End Mobile Manipulation**](https://arxiv.org/pdf/2604.01567) · `2026-04` · Anchored diffusion action head with a truncated schedule. <sub>[Code](https://github.com/jason-lim26/AnchorVLA)</sub>

<details>
<summary>More raw action generation papers</summary>

- [**Fast-dVLA: Accelerating Discrete Diffusion VLA to Real-Time Performance**](https://arxiv.org/pdf/2603.25661) · `2026-03` · Block-wise discrete diffusion with KV-cache reuse. <sub>Sec. 4.2</sub>
- [**FASTER: Rethinking Real-Time Flow VLAs**](https://arxiv.org/pdf/2603.19199) · `2026-03` · Horizon-aware flow action sampling. <sub>Sec. 4.2</sub>
- [**ProbeFlow: Training-Free Adaptive Flow Matching for Vision-Language-Action Models**](https://arxiv.org/pdf/2603.17850) · `2026-03` · Adaptive flow solver for redundant decoding evaluations.
- [**Unifying Language-Action Understanding and Generation for Autonomous Driving (LinkVLA)**](https://arxiv.org/pdf/2603.01441) · `2026-03` · Unified language-action codebook with coarse-to-fine decoding. <sub>AD</sub>
- [**Global Prior Meets Local Consistency: Dual-Memory Augmented Vision-Language-Action Model for Efficient Robotic Manipulation (OptimusVLA)**](https://arxiv.org/pdf/2602.20200) · `2026-02` · Retrieved task-level action priors shorten denoising.
- [**ActionCodec: What Makes for Good Action Tokenizers**](https://arxiv.org/pdf/2602.15397) · `2026-02` · Optimization-oriented action tokenizer.
- [**Recurrent-Depth VLA: Implicit Test-Time Compute Scaling of Vision-Language-Action Models via Latent Iterative Reasoning (RD-VLA)**](https://arxiv.org/pdf/2602.07845) · `2026-02` · Recurrent latent refinement with adaptive stopping. <sub>[Code](https://github.com/rd-vla/rd-vla) · Sec. 1.2</sub>
- [**FASTER: TOWARD EFFICIENT AUTOREGRESSIVE VISION LANGUAGE ACTION MODELING VIA NEURAL ACTION TOKENIZATION**](https://arxiv.org/pdf/2512.04952) · `2025-12` · High-compression action tokenizer with block-wise autoregressive decoding. <sub>Venue: ICLR 2026</sub>
- [**MM-ACT: Learn from Multimodal Parallel Generation to Act**](https://arxiv.org/pdf/2512.00975) · `2025-12` · Parallel action decoding. <sub>[Code](https://github.com/HHYHRHY/MM-ACT) · Sec. 4.2</sub>
- [**AsyncVLA: Asynchronous Flow Matching for Vision-Language-Action Models**](https://arxiv.org/pdf/2511.14148) · `2025-11` · Flow matching with action-aware non-uniform scheduling. <sub>[Code](https://github.com/YuhuaJiang2002/AsyncVLA)</sub>
- [**UNIFIED DIFFUSION VLA: VISION-LANGUAGE-ACTION MODEL VIA JOINT DISCRETE DENOISING DIFFUSION PROCESS**](https://arxiv.org/pdf/2511.01718) · `2025-11` · Joint diffusion denoising for image-action generation. <sub>[Code](https://github.com/OpenHelix-Team/UD-VLA) · Venue: ICLR 2026</sub>
- [**OMNISAT: COMPACT ACTION TOKEN, FASTER AUTOREGRESSION**](https://arxiv.org/pdf/2510.09667) · `2025-10` · Residual-quantized action tokenizer.
- [**DISCRETE DIFFUSION VLA: BRINGING DISCRETE DIFFUSION TO ACTION DECODING IN VISION-LANGUAGE-ACTION POLICIES**](https://arxiv.org/pdf/2508.20072) · `2025-08` · Discrete diffusion decoder with adaptive parallel refinement.
- [**NinA: Normalizing Flows in Action. Training VLA Models with Normalizing Flows**](https://arxiv.org/pdf/2508.16845) · `2025-08` · One-shot action sampling with normalizing flows. <sub>[Code](https://github.com/dunnolab/NinA)</sub>
- [**EdgeVLA: Efficient Vision-Language-Action Models**](https://arxiv.org/pdf/2507.14049) · `2025-07` · End-effector prediction with a smaller language model. <sub>[Code](https://github.com/kscalelabs/evla) · Sec. 1.1</sub>
- [**VOTE: Vision-Language-Action Optimization with Trajectory Ensemble Voting**](https://arxiv.org/pdf/2507.05116) · `2025-07` · Low-token action generation with ensemble voting. <sub>[Code](https://github.com/LukeLIN-web/VOTE) · Sec. 4.2</sub>
- [**Real-Time Execution of Action Chunking Flow Policies (RTC)**](https://arxiv.org/pdf/2506.07339) · `2025-06` · Real-time chunking with freeze-and-inpaint. <sub>[Code](https://github.com/LukeLIN-web/VOTE) · Venue: NeurIPS 2025</sub>
- [**FAST: Efficient Action Tokenization for Vision-Language-Action Models**](https://arxiv.org/pdf/2501.09747) · `2025-01` · Frequency-space action tokenization.
</details>

<a id="reasoning-aware-action-generation"></a>
### 3.2 Reasoning-Aware Action Generation

- [**OneVL: One-Step Latent Reasoning and Planning with Vision-Language Explanation**](https://arxiv.org/pdf/2604.18486) · 🔥 New `2026-04` · One-step latent CoT for single-pass trajectory prediction. <sub>AD</sub>
- [**DualCoT-VLA: Visual-Linguistic Chain of Thought via Parallel Reasoning for Vision-Language-Action Models**](https://arxiv.org/pdf/2603.22280) · `2026-03` · Parallel visual-linguistic Chain-of-Thought.
- [**DynVLA: Learning World Dynamics for Action Reasoning in Autonomous Driving**](https://arxiv.org/pdf/2603.11041) · `2026-03` · Compact future dynamics tokens before action prediction. <sub>AD</sub>
- [**Latent Reasoning VLA: Latent Thinking and Prediction for Vision-Language-Action Models (LaRA-VLA)**](https://www.arxiv.org/pdf/2602.01166) · `2026-02` · Latent reasoning instead of explicit CoT. <sub>[Code](https://github.com/LoveJu1y/LaRA-VLA)</sub>
- [**Fast-ThinkAct: Efficient Vision-Language-Action Reasoning via Verbalizable Latent Planning**](https://arxiv.org/pdf/2601.09708) · `2026-01` · Chain-of-thought distillation for compact reasoning.
- [**LaST0: Latent Spatio-Temporal Chain-of-Thought for Robotic Vision-Language-Action Model**](https://arxiv.org/pdf/2601.05248) · `2026-01` · Latent spatio-temporal CoT. <sub>Sec. 1.3</sub>
- [**Latent Chain-of-Thought World Modeling for End-to-End Autonomous Driving (LCDrive)**](https://arxiv.org/pdf/2512.10226) · `2025-12` · Action-aligned latent chain-of-thought reasoning. <sub>AD</sub>
- [**ThinkAct: Vision-Language-Action Reasoning via Reinforced Visual Latent Planning**](https://arxiv.org/pdf/2507.16815) · `2025-07` · Reinforced visual latent planning. <sub>Venue: NeurIPS 2025 · Sec. 1.3</sub>
- [**Training Strategies for Efficient Embodied Reasoning (ECoT-Lite)**](https://arxiv.org/pdf/2505.08243) · `2025-05` · Robot reasoning recipes for faster CoT-based VLA inference.

<a id="efficient-training-and-inference"></a>
## 4. Efficient Training and Inference

Optimize how VLA models are learned, executed, compressed, deployed, or evaluated.

<a id="training-efficiency-techniques"></a>
### 4.1 Training Efficiency Techniques

#### Adaptation-Efficient Learning

- [**TwinRL-VLA: Digital Twin-Driven Reinforcement Learning for Real-World Robotic Manipulation (TwinRL)**](https://arxiv.org/pdf/2602.09023) · `2026-02` · Digital twin-guided RL for efficient real-world exploration.
- [**RL-VLA3: REINFORCEMENT LEARNING VLA AC-CELERATING VIA FULL ASYNCHRONISM**](https://arxiv.org/pdf/2602.05765) · `2026-02` · Fully asynchronous RL training pipeline.
- [**FT-NCFM: An Influence-Aware Data Distillation Framework for Efficient VLA Models**](https://arxiv.org/pdf/2511.16233) · `2025-11` · Influence-aware generative data distillation. <sub>Venue: AAAI 2026</sub>
- [**VITA-VLA: Efficiently Teaching Vision-Language Models to Act via Action Expert Distillation**](https://arxiv.org/pdf/2510.09607) · `2025-10` · Action expert distillation into a VLM. <sub>[Code](https://github.com/Tencent/VITA)</sub>

#### Distillation and Compression-Oriented Optimization

- [**DA-PTQ: Drift-Aware Post-Training Quantization for Efficient Vision-Language-Action Models**](https://arxiv.org/pdf/2604.11572) · 🔥 New `2026-04` · Drift-aware PTQ with cross-space compensation and mixed precision.
- [**DyQ-VLA: Temporal-Dynamic-Aware Quantization for Embodied Vision-Language-Action Models**](https://arxiv.org/pdf/2603.07904) · `2026-03` · Dynamic quantization using kinematic sensitivity.
- [**QuantVLA: Scale-Calibrated Post-Training Quantization for Vision-Language-Action Models**](https://arxiv.org/pdf/2602.20309) · `2026-02` · Scale-calibrated PTQ for low-bit deployment. <sub>[Code](https://github.com/AIoT-MLSys-Lab/QuantVLA)</sub>
- [**Shallow-π: Knowledge Distillation for Flow-based VLAs**](https://arxiv.org/pdf/2601.20262) · `2026-01` · Knowledge distillation for reduced-depth flow-based VLA models. <sub>Sec. 1.2</sub>
- [**ActDistill: General Action-Guided Self-Derived Distillation for Efficient Vision-Language-Action Models**](https://arxiv.org/pdf/2511.18082) · `2025-11` · Action-guided distillation with a dynamically routed student. <sub>[Code](https://github.com/gooogleshanghai/ActDistill) · Sec. 1.2</sub>
- [**BitVLA: 1-bit Vision-Language-Action Models for Robotics Manipulation**](https://arxiv.org/pdf/2506.07530) · `2025-06` · Distillation-aware low-bit compression. <sub>[Code](https://github.com/ustcwhy/BitVLA)</sub>

<a id="inference-efficiency-techniques"></a>
### 4.2 Inference Efficiency Techniques

#### Runtime Decoding and Execution

- [**A1: A Fully Transparent Open-Source, Adaptive and Efficient Truncated Vision-Language-Action Model**](https://arxiv.org/pdf/2604.05672) · `2026-04` · Budget-aware adaptive inference for backbone and flow-matching head. <sub>[Code](https://github.com/ATeam-Research/A1)</sub>
- [**StreamingVLA: Streaming Vision-Language-Action Model with Action Flow Matching and Adaptive Early Observation**](https://arxiv.org/pdf/2603.28565) · `2026-03` · Overlaps observation, action generation, and execution. <sub>Sec. 3.1</sub>
- [**HeiSD: Hybrid Speculative Decoding for Embodied Vision-Language-Action Models with Kinematic Awareness**](https://arxiv.org/pdf/2603.17573) · `2026-03` · Hybrid speculative decoding with kinematic boundary selection.
- [**KERV: Kinematic-Rectified Speculative Decoding for Embodied VLA Models**](https://arxiv.org/pdf/2603.01581) · `2026-03` · Kinematic-rectified speculative decoding.
- [**VLA Knows Its Limits (AutoHorizon)**](https://arxiv.org/pdf/2602.21445) · `2026-02` · Dynamic action-horizon adjustment.

<details>
<summary>More runtime decoding and execution papers</summary>

- [**Xiaomi-Robotics-0: An Open-Sourced Vision-Language-Action Model with Real-Time Execution**](https://arxiv.org/pdf/2602.12684) · `2026-02` · Deployment-aligned action chunk rollout. <sub>[Code](https://github.com/XiaomiRobotics/Xiaomi-Robotics-0)</sub>
- [**DynamicVLA: A Vision-Language-Action Model for Dynamic Object Manipulation**](https://arxiv.org/pdf/2601.22153) · `2026-01` · Latent-aware action streaming. <sub>[Code](https://github.com/hzxie/DynamicVLA) · Sec. 1.1</sub>
- [**ActionFlow: A Pipelined Action Acceleration for Vision Language Models on Edge**](https://arxiv.org/pdf/2512.20276) · `2025-12` · Pipelined scheduling with unified KV buffer.
- [**DeeAD: Dynamic Early Exit of Vision-Language Action for Efficient Autonomous Driving**](https://arxiv.org/pdf/2511.20720) · `2025-11` · Action-guided early exit with adaptive layer skipping. <sub>AD</sub>
- [**Spec-VLA: Speculative Decoding for Vision-Language-Action Models with Relaxed Acceptance**](https://arxiv.org/pdf/2507.22424) · `2025-07` · Speculative decoding with relaxed acceptance. <sub>[Code](https://github.com/PineTreeWss/SpecVLA) · Venue: EMNLP 2025</sub>
- [**CEED-VLA: Consistency Vision-Language-Action Model with Early-Exit Decoding**](https://arxiv.org/pdf/2506.13725) · `2025-06` · Multi-token prediction with early-exit decoding. <sub>[Code](https://github.com/OpenHelix-Team/CEED-VLA) · Venue: NeurIPS 2025 · Sec. 4.1</sub>
- [**SP-VLA: A joint model scheduling and token-pruning approach for VLA model acceleration**](https://arxiv.org/pdf/2506.12723) · `2025-06` · Action-aware scheduling with spatio-semantic token pruning. <sub>Venue: ICLR 2026 · Sec. 2.1</sub>
- [**Fast ECoT: Efficient Embodied Chain-of-Thought via Thoughts Reuse**](https://arxiv.org/pdf/2506.07639) · `2025-06` · Reasoning cache reuse and parallel generation. <sub>Venue: ICRA 2026 · Sec. 2.2</sub>
- [**Accelerating VLA Models Integrated with Action Chunking via Parallel Decoding (PD-VLA)**](https://arxiv.org/pdf/2503.02310) · `2025-03` · Fixed-point decoding for action-chunked VLA models.
- [**Fine-Tuning VLA Models: Optimizing Speed and Success (OFT)**](https://arxiv.org/pdf/2502.19645) · `2025-02` · Fine-tuning recipe with parallel decoding and action chunking. <sub>[Code](https://github.com/moojink/openvla-oft) · Venue: RSS 2025</sub>
</details>

#### Deployment, Compression, and Scheduling

- [**Realtime-VLA V2: Learning to Run VLAs Fast, Smooth, and Accurate**](https://arxiv.org/pdf/2603.26360) · `2026-03` · Deployment-oriented system with calibration, planning, control, and speed selection. <sub>[Code](https://github.com/dexmal/realtime-vla-v2)</sub>
- [**RAPID: Redundancy-Aware and Compatibility-Optimal Edge-Cloud Partitioned Inference for Diverse VLA Models**](https://arxiv.org/pdf/2603.07949) · `2026-03` · Edge-cloud partitioned inference.
- [**LiteVLA-Edge: Quantized On-Device Multimodal Control for Embedded Robotics**](https://arxiv.org/pdf/2603.03380) · `2026-03` · On-device VLA pipeline with 4-bit quantization. <sub>Sec. 4.1</sub>
- [**HBVLA: Pushing 1-Bit Post-Training Quantization for Vision-Language-Action Models**](https://arxiv.org/pdf/2602.13710) · `2026-02` · Hessian-guided 1-bit binarization.
- [**QVLA: Not All Channels Are Equal in Vision-Language-Action Model's Quantization**](https://arxiv.org/pdf/2602.03782) · `2026-02` · Channel-wise mixed-bit quantization. <sub>[Code](https://github.com/AutoLab-SAI-SJTU/QVLA) · Venue: ICLR 2026</sub>
- [**Don’t Run with Scissors: Pruning Breaks VLA Models but They Can Be Recovered (GLUESTICK)**](https://arxiv.org/pdf/2510.08464) · `2025-10` · Recovery for pruned VLA models.
- [**SQAP-VLA: A Synergistic Quantization-Aware Pruning Framework**](https://arxiv.org/pdf/2509.09090) · `2025-09` · Quantization-aware visual token pruning. <sub>[Code](https://github.com/ecdine/SQAP-VLA) · Sec. 2.1</sub>
- [**EfficientVLA: Training-Free Acceleration and Compression for Vision-Language-Action Models**](https://www.arxiv.org/pdf/2506.10100) · `2025-06` · Layer pruning, visual token selection, and temporal reuse. <sub>Venue: NeurIPS 2025 · Sec. 2.1; 2.2</sub>

#### Efficiency Analysis and Embodied Metrics

- [**From Inference Efficiency to Embodied Efficiency: Revisiting Efficiency Metrics for Vision-Language-Action Models**](https://arxiv.org/pdf/2603.19131) · `2026-03` · System-level analysis of embodied efficiency beyond conventional inference metrics.
- [**How Fast Can I Run My VLA? Demystifying VLA Inference Performance with VLA-Perf**](https://arxiv.org/pdf/2602.18397) · `2026-02` · Analytical performance modeling for real-time VLA inference.

---

<a id="related-surveys"></a>
## Related Surveys

- **[EFFICIENT VISION-LANGUAGE-ACTION MODELS FOR EMBODIED MANIPULATION: A SYSTEMATIC SURVEY](https://arxiv.org/pdf/2510.17111)** *(This work)*
- **[A Survey on Efficient Vision-Language-Action Models](https://arxiv.org/pdf/2510.24795)**

<a id="citation"></a>
## Citation

If you find this survey or resource list helpful, please consider citing our work:

```bibtex
@misc{guan2025efficientvisionlanguageactionmodelsembodied,
      title={Efficient Vision-Language-Action Models for Embodied Manipulation: A Systematic Survey},
      author={Weifan Guan and Qinghao Hu and Aosheng Li and Jian Cheng},
      year={2025},
      eprint={2510.17111},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={[https://arxiv.org/abs/2510.17111](https://arxiv.org/abs/2510.17111)}
}
```

<a id="star-history"></a>
## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=guanweifan/awesome-efficient-vla&type=Timeline)](#fig:starhistory)

<a id="classification-logic"></a>
## Classification Logic

To keep this repository consistent and principled, we classify each paper by its dominant efficiency mechanism: what part of the VLA stack it changes, compresses, reuses, schedules, or evaluates.

The decision tree below is a lightweight diagnostic tool rather than a rigid rule. If a paper spans multiple mechanisms, assign the primary category by the main efficiency bottleneck addressed, and use `Sec.`, `AD`, or `VLN` tags for secondary relevance or domain scope.

This appendix makes the classification transparent and reproducible for future additions to the repository.

```pgsql
START
│
├─ Q1: Is the main intervention the model organization itself?
│      (backbone size, routing depth, slow-fast decomposition)
│      │
│      ├─ Fixed smaller / efficient backbone → 1.1 Static Backbone Selection
│      ├─ Dynamic routing / layer skipping / built-in early exit → 1.2 Dynamic Computation Pathways
│      └─ Slow + fast cooperative policy loops → 1.3 Dual-system Design
│
├─ Q2: Is the main intervention visual or temporal feature processing?
│      │
│      ├─ Spatial token pruning / merging / compression / salience selection
│      │        → 2.1 Selective Feature Processing
│      └─ History fusion / temporal token reuse / KV or feature caching
│               → 2.2 Temporal Sharing and Reuse
│
├─ Q3: Is the main intervention action modeling?
│      │
│      ├─ Action tokenizer, action chunking, diffusion / flow / sampling schedule,
│      │   one-step or parallel action decoder
│      │        → 3.1 Raw Action Generation
│      └─ Text / latent CoT, visual subgoal, world-model reasoning before action
│               → 3.2 Reasoning-Aware Action Generation
│
└─ Q4: Otherwise, is it mainly a learning, compression, deployment, or analysis tool?
       │
       ├─ Training / adaptation / compression-oriented optimization:
       │   PEFT, distillation, data selection, RL fine-tuning, PTQ / QAT calibration
       │        → 4.1 Training Efficiency Techniques
       └─ Runtime / deployment / systems / evaluation:
           speculative or parallel decoding, async / streaming execution,
           pipelining, action reuse, scheduling, edge deployment, metrics
                → 4.2 Inference Efficiency Techniques
```
