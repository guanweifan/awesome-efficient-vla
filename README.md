# Awesome-Efficient-VLA [![Awesome](https://awesome.re/badge.svg)](https://awesome.re)

[![arXiv](https://img.shields.io/badge/arXiv-2510.17111-b31b1b.svg)](https://arxiv.org/pdf/2510.17111)
[![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-green.svg)](https://github.com/guanweifan/awesome-efficient-vla/commits/main/)
[![PR's Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=flat)](https://github.com/guanweifan/awesome-efficient-vla/pulls)

<p align="center">
  <img src="./imgs/overview.png" width="100%" height="80%">
</p>

📖 Survey: A live-updated hub for **[Efficient VLA for Embodied Manipulation](https://arxiv.org/pdf/2510.17111)** across architecture, perception, action & pipeline.

🚀 Updates: Weekly / Bi-weekly. Contributions & ⭐ are welcome to help researchers navigate the field!

---

## 🔥 Latest Updates (2026-03-09)

- **Architecture:** [DySL-VLA](#dynamic-computation-pathways), [TacMamba](#dual-system-design), [Act-Think-Abstain](#dynamic-computation-pathways)
- **Perception:** [FUTURE-VLA](#temporal-sharing-and-reuse), [BFA++](#selective-feature-processing), [A-MMR](#temporal-sharing-and-reuse)
- **Action:** [ActionCodec](#raw-action-generation), [OptimusVLA](#raw-action-generation), [AutoHorizon](#raw-action-generation), [LinkVLA](#raw-action-generation)
- **Training:** [QuantVLA](#training-efficiency-techniques)
- **Inference:** [VLA-Perf](#inference-efficiency-techniques), [KERV](#inference-efficiency-techniques), [LiteVLA-Edge](#inference-efficiency-techniques)

See the full list in the corresponding sections below.

---

## 🧭 Table of Contents


* [Related Surveys](#related-surveys)
* [Paper List by Taxonomy](#paper-list-by-taxonomy)
    * [1. Efficient Model Architecture](#efficient-model-architecture)
        * [1.1 Static Backbone Selection](#static-backbone-selection)
        * [1.2 Dynamic Computation Pathways](#dynamic-computation-pathways)
        * [1.3 Dual-system Design](#dual-system-design)
    * [2. Efficient Perception Feature](#efficient-perception-feature)
        * [2.1 Selective Feature Processing](#selective-feature-processing)
        * [2.2 Temporal Sharing and Reuse](#temporal-sharing-and-reuse)
    * [3. Efficient Action Generation](#efficient-action-generation)
        * [3.1 Raw Action Generation](#raw-action-generation)
        * [3.2 Reasoning-Aware Action Generation](#reasoning-aware-action-generation)
    * [4. Efficient Training and Inference](#efficient-training-and-inference)
        * [4.1 Training Efficiency Techniques](#training-efficiency-techniques)
        * [4.2 Inference Efficiency Techniques](#inference-efficiency-techniques)
* [Citation](#citation)
* [Star History](#star-history)
* [Appendix: Classification Logic](#appendix)

---

<a id="related-surveys"></a>
## 📚 Related Surveys

- **[EFFICIENT VISION-LANGUAGE-ACTION MODELS FOR EMBODIED MANIPULATION: A SYSTEMATIC SURVEY](https://arxiv.org/pdf/2510.17111)** *(This work)*
- **[A Survey on Efficient Vision-Language-Action Models](https://arxiv.org/pdf/2510.24795)**

---

<a id="paper-list-by-taxonomy"></a>
## 🗂️ Paper List by Taxonomy

The following sections organize papers by their primary efficiency mechanism.

> **Tag Notes**
> - `Secondary`: this paper is also relevant to another taxonomy category.
> - `Domain: AD (Autonomous Driving)`: this paper targets driving scenarios, included for transferable efficiency methods.
> - `Domain: VLN (Vision-Language-Navigation)`: this paper targets vision-language navigation, included for transferable efficiency methods.

<a id="efficient-model-architecture"></a>
## 🏗️ Efficient Model Architecture

Reduce structural redundancy inside the model itself.
This category modifies the computation graph or module organization of VLA models to lower capacity redundancy — either by shrinking the backbone, dynamically adapting depth, or splitting reasoning and control into separate systems.

<a id="static-backbone-selection"></a>
## 🦴 1.1 Static Backbone Selection
> Lower the baseline cost by adopting smaller or efficiency-oriented backbones.
The model structure is fixed and lightweight by design, reducing latency and memory at all timesteps without runtime adaptation. The trade-off is reduced peak capacity on complex tasks.

- [**Evo-1: Lightweight Vision-Language-Action Model with Preserved Semantic Alignment**](https://arxiv.org/pdf/2511.04555) [![Star](https://img.shields.io/github/stars/MINT-SJTU/Evo-1.svg?style=social&label=Star)](https://github.com/MINT-SJTU/Evo-1)

  (`2025-11`) Compact multimodal backbone with a cross-modulated diffusion transformer. <sub>Sec. 4.1</sub>

- [**RetoVLA: Reusing Register Tokens for Spatial Reasoning in Vision-Language-Action Models**](https://arxiv.org/pdf/2509.21243)

  (`2025-09`) Reuses Vision Transformer register tokens for spatial reasoning.

- [**SmolVLA: A vision-language-action model for affordable and efficient robotics**](https://arxiv.org/pdf/2506.01844) [![Star](https://img.shields.io/github/stars/huggingface/lerobot.svg?style=social&label=Star)](https://github.com/huggingface/lerobot)

  (`2025-06`) Single-GPU training with asynchronous inference. <sub>Sec. 4.2</sub>

- [**NORA: A SMALL OPEN-SOURCED GENERALIST VISION-LANGUAGE ACTION MODEL FOR EMBODIED TASKS**](https://arxiv.org/pdf/2504.19854) [![Star](https://img.shields.io/github/stars/declare-lab/nora.svg?style=social&label=Star)](https://github.com/declare-lab/nora)

  (`2025-04`) Qwen-2.5-VL-3B backbone with a FAST+ tokenizer.

- [**TinyVLA: Towards Fast, Data-Efficient Vision-Language-Action Models for Robotic Manipulation**](https://arxiv.org/pdf/2409.12514) [![Star](https://img.shields.io/github/stars/liyaxuanliyaxuan/TinyVLA.svg?style=social&label=Star)](https://github.com/liyaxuanliyaxuan/TinyVLA) [![Publish](https://img.shields.io/badge/Conference-RAL%202025-blue)]()

  (`2024-09`) High-speed multimodal backbone with a diffusion policy decoder. <sub>Sec. 4.1</sub>

- [**RoboMamba: Efficient Vision-Language-Action Model for Robotic Reasoning and Manipulation**](https://arxiv.org/pdf/2406.04339) [![Star](https://img.shields.io/github/stars/lmzpai/roboMamba.svg?style=social&label=Star)](https://github.com/lmzpai/roboMamba) [![Publish](https://img.shields.io/badge/Conference-NeurIPS%202024-blue)]()

  (`2024-06`) Mamba state-space backbone for linear-complexity reasoning and efficient inference. <sub>Sec. 4.1</sub>



<a id="dynamic-computation-pathways"></a>
## 🔀 1.2 Dynamic Computation Pathways
> Retain a large-capacity backbone but reduce runtime cost by dynamically selecting computation paths.
Methods in this category introduce layer skipping, routing, or built-in early-exit mechanisms that adapt depth or module usage conditioned on input difficulty.

- [**Act, Think or Abstain: Complexity-Aware Adaptive Inference for Vision-Language-Action Models**](https://arxiv.org/pdf/2603.05147)

  🔥 New (`2026-03`) Adaptive routing between direct acting, deeper reasoning, and abstention based on task complexity.

- [**DySL-VLA: Efficient Vision-Language-Action Model Inference via Dynamic-Static Layer-Skipping for Robot Manipulation**](https://arxiv.org/pdf/2602.22896) [![Star](https://img.shields.io/github/stars/PKU-SEC-Lab/DYSL_VLA.svg?style=social&label=Star)](https://github.com/PKU-SEC-Lab/DYSL_VLA)

  🔥 New (`2026-02`) Dynamic-static layer skipping that prioritizes action-critical layers during inference. <sub>Sec. 4.1</sub>

- [**Environment-Aware Adaptive Pruning with Interleaved Inference Orchestration for Vision-Language-Action Models**](https://arxiv.org/pdf/2602.00780)

  (`2026-02`) Environment-aware adaptive channel pruning with interleaved orchestration. <sub>Sec. 4.2</sub>

- [**Recurrent-Depth VLA: Implicit Test-Time Compute Scaling of Vision-Language-Action Models via Latent Iterative Reasoning**](https://arxiv.org/pdf/2602.07845) [![Star](https://img.shields.io/github/stars/rd-vla/rd-vla.svg?style=social&label=Star)](https://github.com/rd-vla/rd-vla)

  (`2026-02`) Recurrent latent refinement with adaptive stopping for compute-aware VLA inference. <sub>Sec. 3.2</sub>

- [**NANOVLA: ROUTING DECOUPLED VISION-LANGUAGE UNDERSTANDING FOR NANO-SIZED GENERALIST ROBOTIC POLICIES**](https://arxiv.org/pdf/2510.25122)

  (`2025-10`) Lightweight VLA with dynamic routing. <sub>Sec. 4.2</sub>

- [**FLOWER: Democratizing Generalist Robot Policies with Efficient Vision-Language-Action Flow Policies**](https://arxiv.org/pdf/2509.04996) [![Star](https://img.shields.io/github/stars/intuitive-robots/flower_vla_calvin.svg?style=social&label=Star)](https://github.com/intuitive-robots/flower_vla_calvin) [![Publish](https://img.shields.io/badge/Conference-CoRL%202025-blue)]()

  (`2025-09`) Intermediate-modality fusion with LLM layer pruning.

- [**MoLe-VLA: Dynamic Layer-skipping Vision Language Action Model via Mixture-of-Layers for Efficient Robot Manipulation**](https://arxiv.org/pdf/2503.20384) [![Star](https://img.shields.io/github/stars/RoyZry98/MoLe-VLA-Pytorch.svg?style=social&label=Star)](https://github.com/RoyZry98/MoLe-VLA-Pytorch) [![Publish](https://img.shields.io/badge/Conference-AAAI%202026-blue)]()

  (`2025-03`) Mixture-of-Layers VLA with a spatial-temporal router for dynamic LLM layer activation.

- [**DeeR-VLA: Dynamic Inference of Multimodal Large Language Models for Efficient Robot Execution**](https://arxiv.org/pdf/2411.02359) [![Star](https://img.shields.io/github/stars/yueyang130/DeeR-VLA.svg?style=social&label=Star)](https://github.com/yueyang130/DeeR-VLA) [![Publish](https://img.shields.io/badge/Conference-NeurIPS%202024-blue)]()

  (`2024-11`) Multi-exit MLLM-based VLA with resource-aware early termination. <sub>Sec. 4.2</sub>



<a id="dual-system-design"></a>
## ⚖️ 1.3 Dual-system Design
> Decompose the VLA policy into two cooperating modules operating at different frequencies.
A slow, high-capacity reasoning component provides guidance to a fast, lightweight reactive controller, balancing deliberation and real-time execution.

- [**TacMamba: A Tactile History Compression Adapter Bridging Fast Reflexes and Slow VLA Reasoning**](https://arxiv.org/pdf/2603.01700)

  🔥 New (`2026-03`) Tactile history compression that aligns high-frequency tactile control with slower visual reasoning.

- [**StreamVLA: Breaking the Reason-Act Cycle via Completion-State Gating**](https://arxiv.org/pdf/2602.01100)

  (`2026-02`) Lock-and-gated selective slow reasoning. <sub>Sec. 4.2</sub>

- [**LaST0: Latent Spatio-Temporal Chain-of-Thought for Robotic Vision–Language–Action Model**](https://arxiv.org/pdf/2601.05248)

  (`2026-01`) Latent spatio-temporal CoT with a mixture-of-transformers design. <sub>Sec. 3.2</sub>

- [**Video2Act: A Dual-System Video Diffusion Policy with Robotic Spatio-Motional Modeling**](https://arxiv.org/pdf/2512.03044) [![Star](https://img.shields.io/github/stars/jiayueru/Video2Act.svg?style=social&label=Star)](https://github.com/jiayueru/Video2Act)

  (`2025-12`) Dual-system VLA with slow VDM reasoning.

- [**MOTVLA: A VISION-LANGUAGE-ACTION MODEL WITH UNIFIED FAST-SLOW REASONING**](https://arxiv.org/pdf/2510.18337)

  (`2025-10`) Mixture-of-Transformers VLA with unified fast-slow reasoning.

- [**ThinkAct: Vision-Language-Action Reasoning via Reinforced Visual Latent Planning**](https://arxiv.org/pdf/2507.16815) [![Publish](https://img.shields.io/badge/Conference-NeurIPS%202025-blue)]()

  (`2025-07`) Reinforced visual latent planning that bridges reasoning and action execution.

- [**Fast-in-Slow: A Dual-System Foundation Model Unifying Fast Manipulation within Slow Reasoning**](https://arxiv.org/pdf/2506.01953) [![Star](https://img.shields.io/github/stars/CHEN-H01/Fast-in-Slow.svg?style=social&label=Star)](https://github.com/CHEN-H01/Fast-in-Slow) [![Publish](https://img.shields.io/badge/Conference-NeurIPS%202025-blue)]()

  (`2025-06`) Fast execution module within VLM-based reasoning model.

- [**OPENHELIX: A Short Survey, Empirical Analysis, and Open-Source Dual-System VLA Model for Robotic Manipulation**](https://arxiv.org/pdf/2505.03912) [![Star](https://img.shields.io/github/stars/OpenHelix-Team/OpenHelix.svg?style=social&label=Star)](https://github.com/OpenHelix-Team/OpenHelix)

  (`2025-05`) Systematic structural evaluation with a low-cost design.

- [**Hume: Introducing System-2 Thinking in Visual-Language-Action Model**](https://arxiv.org/pdf/2505.21432) [![Star](https://img.shields.io/github/stars/hume-vla/hume.svg?style=social&label=Star)](https://github.com/hume-vla/hume)

  (`2025-05`) Value-guided slow thinking with lightweight reactive action denoising.

- [**HiRT: Enhancing Robotic Control with Hierarchical Robot Transformers**](https://arxiv.org/pdf/2410.05273) [![Publish](https://img.shields.io/badge/Conference-CoRL%202024-blue)]()

  (`2024-10`) Low-frequency VLM reasoning with a high-frequency vision-based control policy.

- [**TOWARDS SYNERGISTIC, GENERALIZED AND EFFICIENT DUAL-SYSTEM FOR ROBOTIC MANIPULATION**](https://arxiv.org/pdf/2410.08001)

  (`2024-10`) Generalist reasoning with a lightweight specialist diffusion policy.


---

<a id="efficient-perception-feature"></a>
## 📷 Efficient Perception Feature

Reduce spatial and temporal redundancy in visual representations.
Since visual tokens dominate attention cost and KV memory, this category focuses on shrinking, filtering, or reusing perceptual features without modifying the core model architecture.

<a id="selective-feature-processing"></a>
## ✂️ 2.1 Selective Feature Processing
> Compress or prune visual tokens before they are consumed by the policy.
Methods selectively retain task-relevant spatial information (foreground, geometry, semantics) to reduce attention cost while preserving critical signals.

- [**BFA++: Hierarchical Best-Feature-Aware Token Prune for Multi-View Vision Language Action Model**](https://arxiv.org/pdf/2602.20566)

  🔥 New (`2026-02`) Hierarchical multi-view token pruning that removes redundant regions and camera views.

- [**Think Proprioceptively: Embodied Visual Reasoning for VLA Manipulation**](https://arxiv.org/pdf/2602.06575)

  (`2026-02`) Early proprioception fusion for aggressive visual token reduction.

- [**HiST-VLA: A Hierarchical Spatio-Temporal Vision-Language-Action Model for End-to-End Autonomous Driving**](https://arxiv.org/pdf/2602.13329)

  (`2026-02`) Token fusion-based sparsification for hierarchical trajectory planning. <sub>AD</sub>

- [**DTP:A SIMPLE YET EFFECTIVE DISTRACTING TOKEN PRUNINGFRAMEWORK FOR VISION-LANGUAGE ACTION MODELS**](https://arxiv.org/pdf/2601.16065)

  (`2026-01`) Distracting token pruning to remove task-irrelevant visual tokens. <sub>[Code](https://anonymous.4open.science/r/CBD3)</sub>

- [**Token Expand-Merge: Training-Free Token Compression for Vision-Language-Action Models**](https://arxiv.org/pdf/2512.09927) [![Star](https://img.shields.io/github/stars/Jasper-aaa/TEAM-VLA.svg?style=social&label=Star)](https://github.com/Jasper-aaa/TEAM-VLA)

  (`2025-12`) Dynamic token expansion and merging.

- [**SemanticVLA: Semantic-Aligned Sparsification and Enhancement for Efficient Robotic Manipulation**](https://arxiv.org/pdf/2511.10518) [![Star](https://img.shields.io/github/stars/JiuTian-VL/SemanticVLA.svg?style=social&label=Star)](https://github.com/JiuTian-VL/SemanticVLA) [![Publish](https://img.shields.io/badge/Conference-AAAI%202026-blue)]()

  (`2025-11`) Dual visual pruning with hierarchical fusion for sparsified VLA perception.

- [**VLA-Pruner: Temporal-Aware Dual-Level Visual Token Pruning for Efficient Vision-Language-Action Inference**](https://arxiv.org/pdf/2511.16449) [![Star](https://img.shields.io/github/stars/MINT-SJTU/VLA-Pruner.svg?style=social&label=Star)](https://github.com/MINT-SJTU/VLA-Pruner)

  (`2025-11`) Dual-objective visual token pruning aligned with semantic understanding.

- [**COMPRESSOR-VLA: INSTRUCTION-GUIDED VISUAL TOKEN COMPRESSION FOR EFFICIENT ROBOTIC MANIPULATION**](https://arxiv.org/pdf/2511.18950)

  (`2025-11`) Hybrid visual token compression with semantic distillation.

- [**The Better You Learn, The Smarter You Prune: Differentiable Token Pruning for VLA**](https://arxiv.org/pdf/2509.12594) [![Star](https://img.shields.io/github/stars/LiAutoAD/LightVLA.svg?style=social&label=Star)](https://github.com/LiAutoAD/LightVLA)

  (`2025-09`) Adaptive visual token pruning via dynamic query-based importance estimation.

- [**ACTION-AWARE DYNAMIC PRUNING FOR EFFICIENT VISION-LANGUAGE-ACTION MANIPULATION**](https://arxiv.org/pdf/2509.22093) [![Publish](https://img.shields.io/badge/Conference-ICLR%202026-blue)]()

  (`2025-09`) Action-aware dynamic visual token pruning with trajectory-conditioned gating.

- [**SpecPrune-VLA: Accelerating VLA Models via Action-Aware Self-Speculative Pruning**](https://arxiv.org/pdf/2509.05614)

  (`2025-09`) Spatial-temporally consistent two-level visual token pruning with action-aware control.

- [**FastDriveVLA: Efficient End-to-End Driving via Plug-and-Play Reconstruction-based Token Pruning**](https://arxiv.org/pdf/2507.23318)

  (`2025-07`) Reconstruction-based foreground-aware visual token pruning. <sub>AD</sub>

- [**EfficientVLA: Training-Free Acceleration and Compression for Vision-Language-Action Models**](https://www.arxiv.org/pdf/2506.10100) [![Publish](https://img.shields.io/badge/Conference-NeurIPS%202025-blue)]()

  (`2025-06`) Holistic VLA acceleration via layer pruning and task-aware visual token selection. <sub>Sec. 2.2</sub>

- [**Think Twice, Act Once: Token-Aware Compression and Action Reuse for Efficient Inference in VLA Models**](https://arxiv.org/pdf/2505.21200)

  (`2025-05`) Action reuse with information-guided visual token pruning. <sub>Sec. 2.2</sub>

- [**OTTER: A Vision-Language-Action Model with Text-Aware Visual Feature Extraction**](https://arxiv.org/pdf/2503.03734) [![Star](https://img.shields.io/github/stars/Max-Fu/otter.svg?style=social&label=Star)](https://github.com/Max-Fu/otter) [![Publish](https://img.shields.io/badge/Conference-ICML%202025-blue)]()

  (`2025-03`) Text-aware selective visual feature extraction.



<a id="temporal-sharing-and-reuse"></a>
## ♻️ 2.2 Temporal Sharing and Reuse
> Exploit temporal consistency to avoid recomputing stable information across timesteps.
This includes feature fusion, KV reuse/compression, or temporal token caching mechanisms that reduce redundant computation in sequential decision-making.

- [**History-Conditioned Spatio-Temporal Visual Token Pruning for Efficient Vision-Language Navigation**](https://arxiv.org/pdf/2603.06480)

  🔥 New (`2026-03`) Training-free spatio-temporal token pruning for current views and navigation history. <sub>Sec. 2.1 · VLN</sub>

- [**FUTURE-VLA: Forecasting Unified Trajectories Under Real-time Execution**](https://arxiv.org/pdf/2602.15882) [![Star](https://img.shields.io/github/stars/fan-jj24/FUTURE-VLA.svg?style=social&label=Star)](https://github.com/fan-jj24/FUTURE-VLA)

  🔥 New (`2026-02`) Temporally adaptive compression for long multi-view histories with constant-latency execution.

- [**Learning to Accelerate Vision-Language-Action Models through Adaptive Visual Token Caching**](https://arxiv.org/pdf/2602.00500) [![Star](https://img.shields.io/github/stars/JiahanFan/LAC.svg?style=social&label=Star)](https://github.com/JiahanFan/LAC)

  (`2026-02`) Task-aware token caching with differentiable selection. <sub>Sec. 4.2</sub>

- [**Efficient Long-Horizon Vision-Language-Action Models via Static-Dynamic Disentanglement**](https://arxiv.org/pdf/2602.03983)

  (`2026-02`) Static-dynamic token disentanglement with KV reuse.

- [**KV-Efficient VLA: A Method of Speed up Vision Language Model with RNN-Gated Chunked KV Cache**](https://arxiv.org/pdf/2509.21354)

  (`2025-09`) Chunk-based KV cache compression with recurrent utility gating.

- [**TTF-VLA: Temporal Token Fusion via Pixel-Attention Integration for VLA Models**](https://arxiv.org/pdf/2508.19257) [![Star](https://img.shields.io/github/stars/PKU-XLab/TTF-VLA.svg?style=social&label=Star)](https://github.com/PKU-XLab/TTF-VLA)

  (`2025-08`) Temporal token fusion with selective historical feature integration.

- [**VLA-Cache: Efficient Vision-Language-Action Manipulation via Adaptive Token Caching**](https://arxiv.org/pdf/2502.02175) [![Star](https://img.shields.io/github/stars/siyuhsu/vla-cache.svg?style=social&label=Star)](https://github.com/siyuhsu/vla-cache) [![Publish](https://img.shields.io/badge/Conference-NeurIPS%202025-blue)]()

  (`2025-02`) Visual token KV caching and reuse.


---

<a id="efficient-action-generation"></a>
## 🤖 Efficient Action Generation

Reduce redundancy in action modeling.
Instead of modifying perception or architecture, this category focuses on how actions are represented, generated, or reasoned about — affecting latency, planning quality, and output dimensionality.

<a id="raw-action-generation"></a>
## 🕹️ 3.1 Raw Action Generation
> Improve efficiency by redesigning action representation or decoding mechanisms without inserting explicit reasoning steps.
This includes action tokenizers, discrete/flow-based decoders, block-wise or parallel decoding for actions, and compact action parameterizations.

- [**Unifying Language-Action Understanding and Generation for Autonomous Driving**](https://arxiv.org/pdf/2603.01441)

  🔥 New (`2026-03`) Unified language-action codebook with coarse-to-fine decoding for efficient driving control. <sub>AD</sub>

- [**VLA Knows Its Limits**](https://arxiv.org/pdf/2602.21445)

  🔥 New (`2026-02`) Dynamic action-horizon adjustment using attention-derived execution limits.

- [**Global Prior Meets Local Consistency: Dual-Memory Augmented Vision-Language-Action Model for Efficient Robotic Manipulation**](https://arxiv.org/pdf/2602.20200)

  🔥 New (`2026-02`) Dual-memory generative policy that shortens the denoising path with retrieved task-level action priors.

- [**ActionCodec: What Makes for Good Action Tokenizers**](https://arxiv.org/pdf/2602.15397)

  🔥 New (`2026-02`) Optimization-oriented action tokenizer that improves temporal overlap and vocabulary efficiency.

- [**MM-ACT: Learn from Multimodal Parallel Generation to Act**](https://arxiv.org/pdf/2512.00975) [![Star](https://img.shields.io/github/stars/HHYHRHY/MM-ACT.svg?style=social&label=Star)](https://github.com/HHYHRHY/MM-ACT)

  (`2025-12`) Parallel action decoding. <sub>Sec. 4.2</sub>

- [**FASTER: TOWARD EFFICIENT AUTOREGRESSIVE VISION LANGUAGE ACTION MODELING VIA NEURAL ACTION TOKENIZATION**](https://arxiv.org/pdf/2512.04952) [![Publish](https://img.shields.io/badge/Conference-ICLR%202026-blue)]()

  (`2025-12`) High-compression action tokenizer with block-wise autoregressive decoding.

- [**AsyncVLA: Asynchronous Flow Matching for Vision-Language-Action Models**](https://arxiv.org/pdf/2511.14148) [![Star](https://img.shields.io/github/stars/YuhuaJiang2002/AsyncVLA.svg?style=social&label=Star)](https://github.com/YuhuaJiang2002/AsyncVLA)

  (`2025-11`) Flow matching with action-aware non-uniform scheduling.

- [**UNIFIED DIFFUSION VLA: VISION-LANGUAGE-ACTION MODEL VIA JOINT DISCRETE DENOISING DIFFUSION PROCESS**](https://arxiv.org/pdf/2511.01718) [![Star](https://img.shields.io/github/stars/OpenHelix-Team/UD-VLA.svg?style=social&label=Star)](https://github.com/OpenHelix-Team/UD-VLA) [![Publish](https://img.shields.io/badge/Conference-ICLR%202026-blue)]()

  (`2025-11`) Joint diffusion denoising for synchronous image-action generation.

- [**OMNISAT: COMPACT ACTION TOKEN, FASTER AUTOREGRESSION**](https://arxiv.org/pdf/2510.09667)

  (`2025-10`) Residual-quantized action tokenizer (OmniSAT) for shortening autoregressive VLA action sequences.

- [**NinA: Normalizing Flows in Action. Training VLA Models with Normalizing Flows**](https://arxiv.org/pdf/2508.16845) [![Star](https://img.shields.io/github/stars/dunnolab/NinA.svg?style=social&label=Star)](https://github.com/dunnolab/NinA)

  (`2025-08`) Action decoder enabling one-shot sampling for fast VLA inference.

- [**DISCRETE DIFFUSION VLA: BRINGING DISCRETE DIFFUSION TO ACTION DECODING IN VISION-LANGUAGE-ACTION POLICIES**](https://arxiv.org/pdf/2508.20072)

  (`2025-08`) Discrete diffusion decoder with adaptive parallel refinement.

- [**EdgeVLA: Efficient Vision-Language-Action Models**](https://arxiv.org/pdf/2507.14049) [![Star](https://img.shields.io/github/stars/kscalelabs/evla.svg?style=social&label=Star)](https://github.com/kscalelabs/evla)

  (`2025-07`) End-effector prediction with a smaller language model. <sub>Sec. 1.1</sub>

- [**VOTE: Vision-Language-Action Optimization with Trajectory Ensemble Voting**](https://arxiv.org/pdf/2507.05116) [![Star](https://img.shields.io/github/stars/LukeLIN-web/VOTE.svg?style=social&label=Star)](https://github.com/LukeLIN-web/VOTE)

  (`2025-07`) Low-token action generation with a voting-based inference ensemble. <sub>Sec. 4.2</sub>

- [**Real-Time Execution of Action Chunking Flow Policies**](https://arxiv.org/pdf/2506.07339) [![Star](https://img.shields.io/github/stars/LukeLIN-web/VOTE.svg?style=social&label=Star)](https://github.com/LukeLIN-web/VOTE) [![Publish](https://img.shields.io/badge/Conference-NeurIPS%202025-blue)]()

  (`2025-06`) Real-time chunking with a freeze-and-inpaint strategy.

- [**FAST: Efficient Action Tokenization for Vision-Language-Action Models**](https://arxiv.org/pdf/2501.09747)

  (`2025-01`) Frequency-space action tokenization (FAST).

- [**Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware**](https://arxiv.org/pdf/2304.13705) [![Star](https://img.shields.io/github/stars/tonyzhaozh/aloha.svg?style=social&label=Star)](https://github.com/tonyzhaozh/aloha) [![Publish](https://img.shields.io/badge/Conference-RSS%202023-blue)]()

  (`2023-04`) Action Chunking with Transformers for multi-step action generation.



<a id="reasoning-aware-action-generation"></a>
<a id="text-reasoning"></a>
<a id="visual-reasoning"></a>
## 💡 3.2 Reasoning-Aware Action Generation
Reasoning can take the form of textual CoT, latent reasoning, visual subgoals, or world-model rollouts, and may be optimized via caching or latent compression.

- [**Latent Reasoning VLA: Latent Thinking and Prediction for Vision-Language-Action Models**](https://www.arxiv.org/pdf/2602.01166) [![Star](https://img.shields.io/github/stars/LoveJu1y/LaRA-VLA.svg?style=social&label=Star)](https://github.com/LoveJu1y/LaRA-VLA)

  (`2026-02`) Reasoning replacing explicit CoT.

- [**Fast-ThinkAct: Efficient Vision-Language-Action Reasoning via Verbalizable Latent Planning**](https://arxiv.org/pdf/2601.09708)

  (`2026-01`) Chain-of-thought distillation for compact, low-latency reasoning.

- [**Latent Chain-of-Thought World Modeling for End-to-End Autonomous Driving**](https://arxiv.org/pdf/2512.10226)

  (`2025-12`) Action-aligned latent chain-of-thought reasoning. <sub>AD</sub>

- [**MEMER: SCALING UP MEMORY FOR ROBOT CONTROL VIA EXPERIENCE RETRIEVAL**](https://arxiv.org/pdf/2510.20328) [![Publish](https://img.shields.io/badge/Conference-ICLR%202026-blue)]()

  (`2025-10`) Memory-aware high-level policy selecting relevant keyframes to guide low-level execution. <sub>Sec. 2.2</sub>

- [**Training Strategies for Efficient Embodied Reasoning**](https://arxiv.org/pdf/2505.08243)

  (`2025-05`) Robot reasoning recipes enabling faster CoT-based VLA inference.

- [**CoT-VLA: Visual Chain-of-Thought Reasoning for Vision-Language-Action Models**](https://arxiv.org/pdf/2503.22020) [![Publish](https://img.shields.io/badge/Conference-CVPR%202025-blue)]()

  (`2025-03`) Visual chain-of-thought reasoning via future frame prediction before action generation.


---

<a id="efficient-training-and-inference"></a>
## 🛠️ Efficient Training and Inference

Optimize how the model is trained or executed, without fundamentally altering its structure or representation.
This category addresses optimization redundancy and execution scheduling rather than architectural or modeling changes.

<a id="training-efficiency-techniques"></a>
## 📈 4.1 Training Efficiency Techniques
> Reduce adaptation cost through improved optimization strategies.
Includes parameter-efficient fine-tuning (PEFT), distillation, data distillation/selection, quantization-aware training and other methods that preserve performance under reduced training cost.

- [**QuantVLA: Scale-Calibrated Post-Training Quantization for Vision-Language-Action Models**](https://arxiv.org/pdf/2602.20309) [![Star](https://img.shields.io/github/stars/AIoT-MLSys-Lab/QuantVLA.svg?style=social&label=Star)](https://github.com/AIoT-MLSys-Lab/QuantVLA)

  🔥 New (`2026-02`) Scale-calibrated post-training quantization for low-bit VLA deployment.

- [**QVLA: Not All Channels Are Equal in Vision-Language-Action Model's Quantization**](https://arxiv.org/pdf/2602.03782) [![Star](https://img.shields.io/github/stars/AutoLab-SAI-SJTU/QVLA.svg?style=social&label=Star)](https://github.com/AutoLab-SAI-SJTU/QVLA) [![Publish](https://img.shields.io/badge/Conference-ICLR%202026-blue)]()

  (`2026-02`) Channel-wise mixed-bit quantization for efficient deployment.

- [**RL-VLA3: REINFORCEMENT LEARNING VLA AC-CELERATING VIA FULL ASYNCHRONISM**](https://arxiv.org/pdf/2602.05765)

  (`2026-02`) RL training pipeline for high-throughput VLA policy optimization.

- [**TwinRL-VLA: Digital Twin-Driven Reinforcement Learning for Real-World Robotic Manipulation**](https://arxiv.org/pdf/2602.09023)

  (`2026-02`) Digital twin-guided RL for efficient real-world exploration.

- [**HBVLA: Pushing 1-Bit Post-Training Quantization for Vision-Language-Action Models**](https://arxiv.org/pdf/2602.13710)

  (`2026-02`) Hessian-guided 1-bit binarization framework.

- [**Shallow-π: Knowledge Distillation for Flow-based VLAs**](https://arxiv.org/pdf/2601.20262)

  (`2026-01`) Transformer depth reduction for flow-based VLA models. <sub>Sec. 1.2</sub>

- [**Towards Accessible Physical AI: LoRA-Based Fine-Tuning of VLA Models for Real-World Robot Control**](https://arxiv.org/pdf/2512.11921)

  (`2025-12`) Resource-efficient fine-tuning of large VLA models for low-cost deployment.

- [**FT-NCFM: An Influence-Aware Data Distillation Framework for Efficient VLA Models**](https://arxiv.org/pdf/2511.16233) [![Publish](https://img.shields.io/badge/Conference-AAAI%202026-blue)]()

  (`2025-11`) Influence-aware generative data distillation for compact VLA training sets.

- [**ActDistill: General Action-Guided Self-Derived Distillation for Efficient Vision-Language-Action Models**](https://arxiv.org/pdf/2511.18082) [![Star](https://img.shields.io/github/stars/gooogleshanghai/ActDistill.svg?style=social&label=Star)](https://github.com/gooogleshanghai/ActDistill)

  (`2025-11`) Distillation with a dynamically routed lightweight student. <sub>Sec. 1.2</sub>

- [**VITA-VLA: Efficiently Teaching Vision-Language Models to Act via Action Expert Distillation**](https://arxiv.org/pdf/2510.09607) [![Star](https://img.shields.io/github/stars/Tencent/VITA.svg?style=social&label=Star)](https://github.com/Tencent/VITA)

  (`2025-10`) Transfer of action modeling from small policy to VLM.

- [**BitVLA: 1-bit Vision-Language-Action Models for Robotics Manipulation**](https://arxiv.org/pdf/2506.07530) [![Star](https://img.shields.io/github/stars/ustcwhy/BitVLA.svg?style=social&label=Star)](https://github.com/ustcwhy/BitVLA)

  (`2025-06`) Distillation-aware compression for memory-efficient VLA deployment.

- [**Saliency-Aware Quantized Imitation Learning for Efficient Robotic Control**](https://arxiv.org/pdf/2505.15304)

  (`2025-05`) Quantization-aware training for low-bit efficient VLA deployment.

- [**Fine-Tuning VLA Models: Optimizing Speed and Success**](https://arxiv.org/pdf/2502.19645) [![Star](https://img.shields.io/github/stars/moojink/openvla-oft.svg?style=social&label=Star)](https://github.com/moojink/openvla-oft) [![Publish](https://img.shields.io/badge/Conference-RSS%202025-blue)]()

  (`2025-02`) Fine-tuning recipe with parallel decoding and action chunking. <sub>Sec. 4.2</sub>



<a id="inference-efficiency-techniques"></a>
## ⚡ 4.2 Inference Efficiency Techniques
> Improve runtime latency through decoding or system-level scheduling optimizations, without modifying the model structure.
Includes speculative decoding, parallel decoding, pipelining, chunk scheduling, action reuse, runtime early-exit policies, and KV scheduling strategies.

- [**LiteVLA-Edge: Quantized On-Device Multimodal Control for Embedded Robotics**](https://arxiv.org/pdf/2603.03380)

  🔥 New (`2026-03`) On-device VLA pipeline with 4-bit quantization and GPU-accelerated runtime execution. <sub>Sec. 4.1</sub>

- [**KERV: Kinematic-Rectified Speculative Decoding for Embodied VLA Models**](https://arxiv.org/pdf/2603.01581)

  🔥 New (`2026-03`) Kinematic-rectified speculative decoding with adaptive acceptance thresholds.

- [**How Fast Can I Run My VLA? Demystifying VLA Inference Performance with VLA-Perf**](https://arxiv.org/pdf/2602.18397)

  🔥 New (`2026-02`) Analytical performance modeling for real-time VLA inference across architectures and execution modes.

- [**Xiaomi-Robotics-0: An Open-Sourced Vision-Language-Action Model with Real-Time Execution**](https://arxiv.org/pdf/2602.12684) [![Star](https://img.shields.io/github/stars/XiaomiRobotics/Xiaomi-Robotics-0.svg?style=social&label=Star)](https://github.com/XiaomiRobotics/Xiaomi-Robotics-0)

  (`2026-02`) Execution training with deployment-aligned action chunk rollout.

- [**DynamicVLA: A Vision-Language-Action Model for Dynamic Object Manipulation**](https://arxiv.org/pdf/2601.22153) [![Star](https://img.shields.io/github/stars/hzxie/DynamicVLA.svg?style=social&label=Star)](https://github.com/hzxie/DynamicVLA)

  (`2026-01`) Latent-aware action streaming for dynamic inference. <sub>Sec. 1.1</sub>

- [**ActionFlow: A Pipelined Action Acceleration for Vision Language Models on Edge**](https://arxiv.org/pdf/2512.20276)

  (`2025-12`) Pipelined scheduling with a unified KV buffer.

- [**DeeAD: Dynamic Early Exit of Vision-Language Action for Efficient Autonomous Driving**](https://arxiv.org/pdf/2511.20720)

  (`2025-11`) Action-guided early exit with adaptive layer skipping. <sub>AD</sub>

- [**Don’t Run with Scissors: Pruning Breaks VLA Models but They Can Be Recovered**](https://arxiv.org/pdf/2510.08464)

  (`2025-10`) Weight-space interpolation recovery.

- [**SQAP-VLA: A Synergistic Quantization-Aware Pruning Framework**](https://arxiv.org/pdf/2509.09090) [![Star](https://img.shields.io/github/stars/ecdine/SQAP-VLA.svg?style=social&label=Star)](https://github.com/ecdine/SQAP-VLA)

  (`2025-09`) Quantization-aware visual token pruning for training-free holistic VLA inference acceleration. <sub>Sec. 2.1</sub>

- [**Spec-VLA: Speculative Decoding for Vision-Language-Action Models with Relaxed Acceptance**](https://arxiv.org/pdf/2507.22424) [![Star](https://img.shields.io/github/stars/PineTreeWss/SpecVLA.svg?style=social&label=Star)](https://github.com/PineTreeWss/SpecVLA) [![Publish](https://img.shields.io/badge/Conference-EMNLP%202025-blue)]()

  (`2025-07`) Speculative decoding with relaxed acceptance.

- [**SP-VLA: A joint model scheduling and token-pruning approach for VLA model acceleration**](https://arxiv.org/pdf/2506.12723) [![Publish](https://img.shields.io/badge/Conference-ICLR%202026-blue)]()

  (`2025-06`) Action-aware model scheduling with spatio-semantic dual token pruning. <sub>Sec. 2.1</sub>

- [**Fast ECoT: Efficient Embodied Chain-of-Thought via Thoughts Reuse**](https://arxiv.org/pdf/2506.07639) [![Publish](https://img.shields.io/badge/Conference-ICRA%202026-blue)]()

  (`2025-06`) Embodied CoT acceleration via reasoning cache reuse and parallel generation. <sub>Sec. 2.2</sub>

- [**CEED-VLA: Consistency Vision-Language-Action Model with Early-Exit Decoding**](https://arxiv.org/pdf/2506.13725) [![Star](https://img.shields.io/github/stars/OpenHelix-Team/CEED-VLA.svg?style=social&label=Star)](https://github.com/OpenHelix-Team/CEED-VLA) [![Publish](https://img.shields.io/badge/Conference-NeurIPS%202025-blue)]()

  (`2025-06`) Multi-token prediction with early-exit decoding. <sub>Sec. 4.1</sub>

- [**Accelerating VLA Models Integrated with Action Chunking via Parallel Decoding**](https://arxiv.org/pdf/2503.02310)

  (`2025-03`) Fixed-point decoding for accelerating action-chunked VLA models without retraining.



---

<a id="citation"></a>
## 📚 Citation

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
## ⭐ Star History

[![Star History Chart](https://api.star-history.com/svg?repos=guanweifan/awesome-efficient-vla&type=Timeline)](#fig:starhistory)

---

<a id="appendix"></a>
## 📎 Appendix: Classification Logic
To keep this repository consistent and principled, we organize efficient VLA methods based on where the core efficiency mechanism intervenes in the system — architecture, perception, action generation, or training/inference paradigm.

The decision tree below serves as a lightweight diagnostic tool.
It is not meant to be rigid, but to clarify boundaries, reduce ambiguity, and ensure that methods with similar mechanistic characteristics are grouped together rather than mixed by surface-level similarities.

This appendix makes the classification transparent and reproducible for future additions to the repository.
``` pgsql
START
│
├─ Q1: Does the core method change the model’s computation semantics?
│      (architecture / representations / action-generation mechanism)
│
│   ├─ NO  → Ch.4 (Paradigm-level efficiency)
│   │       │
│   │       ├─ Q1a: Mainly reduces training/adaptation cost?
│   │       │      (PEFT/LoRA, distillation, data distillation/selection, QAT, RL fine-tuning)
│   │       │        → 4.1 Training Efficiency Techniques
│   │       │
│   │       └─ Q1b: Mainly reduces inference latency via execution/decoding/scheduling
│   │              WITHOUT changing model structure?
│   │              (speculative/parallel decoding, pipelining, chunk scheduling, action reuse,
│   │               runtime early-exit as a stopping rule, KV scheduling)
│   │                → 4.2 Inference Efficiency Techniques
│   │
│   └─ YES → Ch.1–3 (Structural efficiency)
│           │
│           ├─ Q2: Does it insert an explicit reasoning stage before actions?
│           │      (text CoT, latent CoT, visual subgoals, world-model rollout)
│           │        → 3.2 Reasoning-Aware Action Generation
│           │
│           ├─ Q3: Does it redesign action representation or the action decoder itself?
│           │      (action tokenizer, VQ/FAST-like codes, flow/one-shot heads,
│           │       discrete diffusion / parallel action decoding as a generation mechanism)
│           │        → 3.1 Raw Action Generation
│           │
│           ├─ Q4: Is the main mechanism about reducing spatial visual tokens?
│           │      (token pruning/merging/compression/foreground selection)
│           │        → 2.1 Selective Feature Processing
│           │
│           ├─ Q5: Is the main mechanism about reusing information over time?
│           │      (temporal token fusion, KV cache compression/reuse, feature caching)
│           │        → 2.2 Temporal Sharing and Reuse
│           │
│           └─ Q6: Otherwise, it targets model capacity / computation paths
│                  │
│                  ├─ Fixed smaller backbone by design → 1.1 Static Backbone Selection
│                  ├─ Dynamic routing / layer skipping / built-in early-exit → 1.2 Dynamic Computation Pathways
│                  └─ Slow+Fast cooperative policy loops → 1.3 Dual-system Design

```
