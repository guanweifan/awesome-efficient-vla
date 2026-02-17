# Awesome-Efficient-VLA [![Awesome](https://awesome.re/badge.svg)](https://awesome.re)

[![arXiv](https://img.shields.io/badge/arXiv-2510.17111-b31b1b.svg)](https://arxiv.org/pdf/2510.17111)
[![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-green.svg)](https://GitHub.com/Naereen/StrapDown.js/graphs/commit-activity)
[![PR's Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=flat)](http://makeapullrequest.com)

<p align="center">
  <img src="./imgs/overview.png" width="100%" height="80%">
</p>

📖 Survey: A live-updated hub for **[Efficient VLA for Embodied Manipulation](https://arxiv.org/pdf/2510.17111)** across architecture, perception, action & pipeline.

🚀 Updates: Weekly / Bi-weekly. Contributions & ⭐ are welcome to help researchers navigate the field!

---

## 🔥 Latest Updates (2026-02-17)

- **Architecture:** [EcoVLA](#dynamic-computation-pathways), [RD-VLA](#dynamic-computation-pathways), [StreamVLA](#dual-system-design)
- **Perception:** [LAC](#temporal-sharing-and-reuse), [SD-VLA](#temporal-sharing-and-reuse), [HiST-VLA](#selective-feature-processing), [ThinkProprio](#selective-feature-processing)
- **Reasoning:** [LaRA-VLA](#reasoning-aware-action-generation)
- **Training:** [QVLA](#training-efficiency-techniques), [RL-VLA3](#training-efficiency-techniques), [TwinRL](#training-efficiency-techniques), [HBVLA](#training-efficiency-techniques)
- **Inference:** [DynamicVLA](#inference-efficiency-techniques), [Xiaomi-Robotics-0](#inference-efficiency-techniques)

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

<a id="efficient-model-architecture"></a>
## 🏗️ Efficient Model Architecture

Reduce structural redundancy inside the model itself.
This category modifies the computation graph or module organization of VLA models to lower capacity redundancy — either by shrinking the backbone, dynamically adapting depth, or splitting reasoning and control into separate systems.

<a id="static-backbone-selection"></a>
## 🦴 1.1 Static Backbone Selection
> Lower the baseline cost by adopting smaller or efficiency-oriented backbones.
The model structure is fixed and lightweight by design, reducing latency and memory at all timesteps without runtime adaptation. The trade-off is reduced peak capacity on complex tasks.

| Title | Date | Key Idea | Resources |
|:--|:--:|:--|:--:|
| [![Star](https://img.shields.io/github/stars/MINT-SJTU/Evo-1.svg?style=social&label=Star)](https://github.com/MINT-SJTU/Evo-1) <br> [**Evo-1: Lightweight Vision-Language-Action Model with Preserved Semantic Alignment**](https://arxiv.org/pdf/2511.04555) <br><sub>Secondary: `4.1 Training Efficiency Techniques`</sub> | 2025-11 | Lightweight VLA built on a compact multimodal backbone with cross-modulated diffusion transformer and optimized integration module. | [Code](https://github.com/MINT-SJTU/Evo-1) |
| [**RetoVLA: Reusing Register Tokens for Spatial Reasoning in Vision-Language-Action Models**](https://arxiv.org/pdf/2509.21243) | 2025-09 | Lightweight VLA reusing Vision Transformer register tokens to enhance spatial reasoning without increasing model size. | [Website](https://www.youtube.com/watch?v=2CseBR-snZg) |
| [![Star](https://img.shields.io/github/stars/huggingface/lerobot.svg?style=social&label=Star)](https://github.com/huggingface/lerobot) <br> [**SmolVLA: A vision-language-action model for affordable and efficient robotics**](https://arxiv.org/pdf/2506.01844) <br><sub>Secondary: `4.2 Inference Efficiency Techniques`</sub> | 2025-06 | Small-scale VLA designed for single-GPU training with asynchronous inference and chunked action generation. | [Code](https://github.com/huggingface/lerobot) / [Website](https://huggingface.co/blog/smolvla) |
| [![Star](https://img.shields.io/github/stars/declare-lab/nora.svg?style=social&label=Star)](https://github.com/declare-lab/nora) <br> [**NORA: A SMALL OPEN-SOURCED GENERALIST VISION-LANGUAGE ACTION MODEL FOR EMBODIED TASKS**](https://arxiv.org/pdf/2504.19854) | 2025-04 | 3B-parameter VLA built on Qwen-2.5-VL-3B backbone with FAST+ tokenizer for efficient action sequence modeling. | [Code](https://github.com/declare-lab/nora) / [Website](https://declare-lab.github.io/nora) |
| [![Star](https://img.shields.io/github/stars/liyaxuanliyaxuan/TinyVLA.svg?style=social&label=Star)](https://github.com/liyaxuanliyaxuan/TinyVLA) [![Publish](https://img.shields.io/badge/Conference-RAL%202025-blue)]() <br> [**TinyVLA: Towards Fast, Data-Efficient Vision-Language-Action Models for Robotic Manipulation**](https://arxiv.org/pdf/2409.12514) <br><sub>Secondary: `4.1 Training Efficiency Techniques`</sub> | 2024-09 | Compact VLA initialized with high-speed multimodal backbone and diffusion policy decoder for efficient learning and inference. | [Code](https://github.com/liyaxuanliyaxuan/TinyVLA) / [Website](https://tiny-vla.github.io/) |
| [![Star](https://img.shields.io/github/stars/lmzpai/roboMamba.svg?style=social&label=Star)](https://github.com/lmzpai/roboMamba) [![Publish](https://img.shields.io/badge/Conference-NeurIPS%202024-blue)]() <br> [**RoboMamba: Efficient Vision-Language-Action Model for Robotic Reasoning and Manipulation**](https://arxiv.org/pdf/2406.04339) <br><sub>Secondary: `4.1 Training Efficiency Techniques`</sub> | 2024-06 | VLA built on Mamba state-space backbone for linear-complexity reasoning and efficient inference. | [Code](https://github.com/lmzpai/roboMamba) / [Website](https://sites.google.com/view/robomamba-web) |


<a id="dynamic-computation-pathways"></a>
## 🔀 1.2 Dynamic Computation Pathways
> Retain a large-capacity backbone but reduce runtime cost by dynamically selecting computation paths.
Methods in this category introduce layer skipping, routing, or built-in early-exit mechanisms that adapt depth or module usage conditioned on input difficulty.

| Title | Date | Key Idea | Resources |
|:--|:--:|:--|:--:|
| [**Environment-Aware Adaptive Pruning with Interleaved Inference Orchestration for Vision-Language-Action Models**](https://arxiv.org/pdf/2602.00780) <br><sub>Secondary: `4.2 Inference Efficiency Techniques`</sub> | 2026-02 | Training-free environment-aware adaptive channel pruning with interleaved orchestration for efficient VLA inference. | - |
| [![Star](https://img.shields.io/github/stars/rd-vla/rd-vla.svg?style=social&label=Star)](https://github.com/rd-vla/rd-vla) <br> [**Recurrent-Depth VLA: Implicit Test-Time Compute Scaling of Vision-Language-Action Models via Latent Iterative Reasoning**](https://arxiv.org/pdf/2602.07845) <br><sub>Secondary: `3.2 Reasoning-Aware Action Generation`</sub> | 2026-02 | Recurrent latent iterative refinement with adaptive stopping for compute-adaptive VLA inference. | [Code](https://github.com/rd-vla/rd-vla) / [Website](https://rd-vla.github.io/) |
| [**NANOVLA: ROUTING DECOUPLED VISION-LANGUAGE UNDERSTANDING FOR NANO-SIZED GENERALIST ROBOTIC POLICIES**](https://arxiv.org/pdf/2510.25122) <br><sub>Secondary: `4.2 Inference Efficiency Techniques`</sub> | 2025-10 | Lightweight VLA with dynamic routing and vision-language decoupling for efficient edge deployment. | - |
| [![Star](https://img.shields.io/github/stars/intuitive-robots/flower_vla_calvin.svg?style=social&label=Star)](https://github.com/intuitive-robots/flower_vla_calvin) [![Publish](https://img.shields.io/badge/Conference-CoRL%202025-blue)]() <br> [**FLOWER: Democratizing Generalist Robot Policies with Efficient Vision-Language-Action Flow Policies**](https://arxiv.org/pdf/2509.04996) | 2025-09 | Intermediate-modality fusion with LLM layer pruning and modular Global-AdaLN conditioning for compact diffusion VLA. | [Code](https://github.com/intuitive-robots/flower_vla_calvin) / [Website](https://intuitive-robots.github.io/flower_vla/) |
| [![Star](https://img.shields.io/github/stars/RoyZry98/MoLe-VLA-Pytorch.svg?style=social&label=Star)](https://github.com/RoyZry98/MoLe-VLA-Pytorch) [![Publish](https://img.shields.io/badge/Conference-AAAI%202026-blue)]() <br> [**MoLe-VLA: Dynamic Layer-skipping Vision Language Action Model via Mixture-of-Layers for Efficient Robot Manipulation**](https://arxiv.org/pdf/2503.20384) | 2025-03 | Mixture-of-Layers VLA with spatial-temporal router for dynamic LLM layer activation. | [Code](https://github.com/RoyZry98/MoLe-VLA-Pytorch/) / [Website](https://sites.google.com/view/mole-vla) |
| [![Star](https://img.shields.io/github/stars/yueyang130/DeeR-VLA.svg?style=social&label=Star)](https://github.com/yueyang130/DeeR-VLA) [![Publish](https://img.shields.io/badge/Conference-NeurIPS%202024-blue)]() <br> [**DeeR-VLA: Dynamic Inference of Multimodal Large Language Models for Efficient Robot Execution**](https://arxiv.org/pdf/2411.02359) <br><sub>Secondary: `4.2 Inference Efficiency Techniques`</sub> | 2024-11 | Multi-exit MLLM-based VLA with dynamic early termination conditioned on resource constraints. | [Code](https://github.com/yueyang130/DeeR-VLA) |


<a id="dual-system-design"></a>
## ⚖️ 1.3 Dual-system Design
> Decompose the VLA policy into two cooperating modules operating at different frequencies.
A slow, high-capacity reasoning component provides guidance to a fast, lightweight reactive controller, balancing deliberation and real-time execution.

| Title | Date | Key Idea | Resources |
|:--|:--:|:--|:--:|
| [**StreamVLA: Breaking the Reason-Act Cycle via Completion-State Gating**](https://arxiv.org/pdf/2602.01100) <br><sub>Secondary: `4.2 Inference Efficiency Techniques`</sub> | 2026-02 | Dual-system VLA with lock-and-gated selective slow reasoning to reduce redundant multimodal computation. | - |
| [**LaST0: Latent Spatio-Temporal Chain-of-Thought for Robotic Vision–Language–Action Model**](https://arxiv.org/pdf/2601.05248) <br><sub>Secondary: `3.2 Reasoning-Aware Action Generation`</sub> | 2026-01 | Dual-system VLA with latent spatio-temporal CoT and mixture-of-transformers for efficient reasoning and high-frequency action. | - |
| [![Star](https://img.shields.io/github/stars/jiayueru/Video2Act.svg?style=social&label=Star)](https://github.com/jiayueru/Video2Act) <br> [**Video2Act: A Dual-System Video Diffusion Policy with Robotic Spatio-Motional Modeling**](https://arxiv.org/pdf/2512.03044) | 2025-12 | Asynchronous dual-system VLA with slow VDM reasoning and fast DiT action head. | [Code](https://github.com/jiayueru/Video2Act) / [Website](https://video2act.github.io/) |
| [**MOTVLA: A VISION-LANGUAGE-ACTION MODEL WITH UNIFIED FAST-SLOW REASONING**](https://arxiv.org/pdf/2510.18337) | 2025-10 | Mixture-of-Transformers VLA with unified fast–slow reasoning via generalist VLM and domain-specific expert. | [Website](https://motvla.github.io/MoTVLA-website/) |
| [![Publish](https://img.shields.io/badge/Conference-NeurIPS%202025-blue)]() <br> [**ThinkAct: Vision-Language-Action Reasoning via Reinforced Visual Latent Planning**](https://arxiv.org/pdf/2507.16815) | 2025-07 | Dual-system VLA with reinforced visual latent planning bridging reasoning and action execution. | [Website](https://jasper0314-huang.github.io/thinkact-vla/) |
| [![Star](https://img.shields.io/github/stars/CHEN-H01/Fast-in-Slow.svg?style=social&label=Star)](https://github.com/CHEN-H01/Fast-in-Slow) [![Publish](https://img.shields.io/badge/Conference-NeurIPS%202025-blue)]() <br> [**Fast-in-Slow: A Dual-System Foundation Model Unifying Fast Manipulation within Slow Reasoning**](https://arxiv.org/pdf/2506.01953) | 2025-06 | Unified dual-system VLA embedding fast execution module within VLM-based reasoning model. | [Code](https://github.com/CHEN-H01/Fast-in-Slow) / [Website](https://fast-in-slow.github.io/) |
| [![Star](https://img.shields.io/github/stars/OpenHelix-Team/OpenHelix.svg?style=social&label=Star)](https://github.com/OpenHelix-Team/OpenHelix) <br> [**OPENHELIX: A Short Survey, Empirical Analysis, and Open-Source Dual-System VLA Model for Robotic Manipulation**](https://arxiv.org/pdf/2505.03912) | 2025-05 | Open-source dual-system VLA with systematic structural evaluation and low-cost design for practical deployment. | [Code](https://github.com/OpenHelix-Team/OpenHelix) / [Website](https://openhelix-robot.github.io/) |
| [![Star](https://img.shields.io/github/stars/hume-vla/hume.svg?style=social&label=Star)](https://github.com/hume-vla/hume) <br> [**Hume: Introducing System-2 Thinking in Visual-Language-Action Model**](https://arxiv.org/pdf/2505.21432) | 2025-05 | Dual-system VLA with value-guided slow thinking and lightweight reactive action denoising. | [Code](https://github.com/hume-vla/hume) / [Website](https://hume-vla.github.io/) |
| [![Publish](https://img.shields.io/badge/Conference-CoRL%202024-blue)]() <br> [**HiRT: Enhancing Robotic Control with Hierarchical Robot Transformers**](https://arxiv.org/pdf/2410.05273) | 2024-10 | Hierarchical VLA with low-frequency VLM reasoning and high-frequency vision-based control policy. | - |
| [**TOWARDS SYNERGISTIC, GENERALIZED AND EFFICIENT DUAL-SYSTEM FOR ROBOTIC MANIPULATION**](https://arxiv.org/pdf/2410.08001) | 2024-10 | Synergistic dual-system VLA combining generalist reasoning and lightweight specialist diffusion policy for efficient control. | [Website](https://robodual.github.io/) |

---

<a id="efficient-perception-feature"></a>
## 📷 Efficient Perception Feature

Reduce spatial and temporal redundancy in visual representations.
Since visual tokens dominate attention cost and KV memory, this category focuses on shrinking, filtering, or reusing perceptual features without modifying the core model architecture.

<a id="selective-feature-processing"></a>
## ✂️ 2.1 Selective Feature Processing
> Compress or prune visual tokens before they are consumed by the policy.
Methods selectively retain task-relevant spatial information (foreground, geometry, semantics) to reduce attention cost while preserving critical signals.

| Title | Date | Key Idea | Resources |
|:--|:--:|:--|:--:|
| [**Think Proprioceptively: Embodied Visual Reasoning for VLA Manipulation**](https://arxiv.org/pdf/2602.06575) | 2026-02 | Text-tokenized early proprioception fusion enabling aggressive visual token reduction in VLA. | - |
| [**HiST-VLA: A Hierarchical Spatio-Temporal Vision-Language-Action Model for End-to-End Autonomous Driving**](https://arxiv.org/pdf/2602.13329) <br><sub>Domain: `AD (Autonomous Driving)`</sub> | 2026-02 | Dynamic token fusion-based sparsification for efficient hierarchical VLA trajectory planning. | - |
| [**DTP:A SIMPLE YET EFFECTIVE DISTRACTING TOKEN PRUNINGFRAMEWORK FOR VISION-LANGUAGE ACTION MODELS**](https://arxiv.org/pdf/2601.16065) | 2026-01 | Dynamic distracting token pruning to remove task-irrelevant visual tokens during VLA inference. | [Code](https://anonymous.4open.science/r/CBD3) |
| [![Star](https://img.shields.io/github/stars/Jasper-aaa/TEAM-VLA.svg?style=social&label=Star)](https://github.com/Jasper-aaa/TEAM-VLA) <br> [**Token Expand-Merge: Training-Free Token Compression for Vision-Language-Action Models**](https://arxiv.org/pdf/2512.09927) | 2025-12 | Training-free dynamic token expansion and merging for efficient VLA inference. | [Code](https://github.com/Jasper-aaa/TEAM-VLA) |
| [![Star](https://img.shields.io/github/stars/JiuTian-VL/SemanticVLA.svg?style=social&label=Star)](https://github.com/JiuTian-VL/SemanticVLA) [![Publish](https://img.shields.io/badge/Conference-AAAI%202026-blue)]() <br> [**SemanticVLA: Semantic-Aligned Sparsification and Enhancement for Efficient Robotic Manipulation**](https://arxiv.org/pdf/2511.10518) | 2025-11 | Semantic-aligned dual visual pruning with hierarchical fusion for sparsified VLA perception. | [Code](https://github.com/JiuTian-VL/SemanticVLA) |
| [![Star](https://img.shields.io/github/stars/MINT-SJTU/VLA-Pruner.svg?style=social&label=Star)](https://github.com/MINT-SJTU/VLA-Pruner) <br> [**VLA-Pruner: Temporal-Aware Dual-Level Visual Token Pruning for Efficient Vision-Language-Action Inference**](https://arxiv.org/pdf/2511.16449) | 2025-11 | Dual-objective visual token pruning aligned with semantic understanding and action execution for efficient VLA inference. | [Code](https://github.com/MINT-SJTU/VLA-Pruner) |
| [**COMPRESSOR-VLA: INSTRUCTION-GUIDED VISUAL TOKEN COMPRESSION FOR EFFICIENT ROBOTIC MANIPULATION**](https://arxiv.org/pdf/2511.18950) | 2025-11 | Instruction-conditioned hybrid visual token compression with semantic distillation and spatial refinement for efficient VLA inference. | - |
| [![Star](https://img.shields.io/github/stars/LiAutoAD/LightVLA.svg?style=social&label=Star)](https://github.com/LiAutoAD/LightVLA) <br> [**The Better You Learn, The Smarter You Prune: Differentiable Token Pruning for VLA**](https://arxiv.org/pdf/2509.12594) | 2025-09 | Differentiable adaptive visual token pruning via dynamic query-based importance estimation for efficient VLA inference. | [Code](https://github.com/LiAutoAD/LightVLA) / [Website](https://liauto-research.github.io/LightVLA/) |
| [![Publish](https://img.shields.io/badge/Conference-ICLR%202026-blue)]() <br> [**ACTION-AWARE DYNAMIC PRUNING FOR EFFICIENT VISION-LANGUAGE-ACTION MANIPULATION**](https://arxiv.org/pdf/2509.22093) | 2025-09 | Action-aware dynamic visual token pruning with trajectory-conditioned gating for efficient VLA inference. | - |
| [**SpecPrune-VLA: Accelerating VLA Models via Action-Aware Self-Speculative Pruning**](https://arxiv.org/pdf/2509.05614) | 2025-09 | Spatial-temporal consistent two-level visual token pruning with action-aware control for efficient VLA inference. | - |
| [**FastDriveVLA: Efficient End-to-End Driving via Plug-and-Play Reconstruction-based Token Pruning**](https://arxiv.org/pdf/2507.23318) <br><sub>Domain: `AD (Autonomous Driving)`</sub> | 2025-07 | Reconstruction-based foreground-aware visual token pruning for efficient VLA inference in autonomous driving. | - |
| [![Publish](https://img.shields.io/badge/Conference-NeurIPS%202025-blue)]() <br> [**EfficientVLA: Training-Free Acceleration and Compression for Vision-Language-Action Models**](https://www.arxiv.org/pdf/2506.10100) <br><sub>Secondary: `2.2 Temporal Sharing and Reuse`</sub> | 2025-06 | Training-free holistic VLA acceleration via layer pruning, task-aware visual token selection, and temporal feature reuse in diffusion head. | - |
| [**Think Twice, Act Once: Token-Aware Compression and Action Reuse for Efficient Inference in VLA Models**](https://arxiv.org/pdf/2505.21200) <br><sub>Secondary: `2.2 Temporal Sharing and Reuse`</sub> | 2025-05 | Training-free action reuse with information-guided visual token pruning for efficient VLA inference. | - |
| [![Star](https://img.shields.io/github/stars/Max-Fu/otter.svg?style=social&label=Star)](https://github.com/Max-Fu/otter) [![Publish](https://img.shields.io/badge/Conference-ICML%202025-blue)]() <br> [**OTTER: A Vision-Language-Action Model with Text-Aware Visual Feature Extraction**](https://arxiv.org/pdf/2503.03734) | 2025-03 | Text-aware selective visual feature extraction to preserve pretrained VLM alignment in VLA. | [Code](https://github.com/Max-Fu/otter) / [Website](https://ottervla.github.io/) |


<a id="temporal-sharing-and-reuse"></a>
## ♻️ 2.2 Temporal Sharing and Reuse
> Exploit temporal consistency to avoid recomputing stable information across timesteps.
This includes feature fusion, KV reuse/compression, or temporal token caching mechanisms that reduce redundant computation in sequential decision-making.

| Title | Date | Key Idea | Resources |
|:--|:--:|:--|:--:|
| [![Star](https://img.shields.io/github/stars/JiahanFan/LAC.svg?style=social&label=Star)](https://github.com/JiahanFan/LAC) <br> [**Learning to Accelerate Vision-Language-Action Models through Adaptive Visual Token Caching**](https://arxiv.org/pdf/2602.00500) <br><sub>Secondary: `4.2 Inference Efficiency Techniques`</sub> | 2026-02 | Learnable task-aware token caching with differentiable selection for efficient VLA inference. | [Code](https://github.com/JiahanFan/LAC) |
| [**Efficient Long-Horizon Vision-Language-Action Models via Static-Dynamic Disentanglement**](https://arxiv.org/pdf/2602.03983) | 2026-02 | Static-dynamic token disentanglement with KV reuse for efficient long-horizon VLA inference. | - |
| [**KV-Efficient VLA: A Method of Speed up Vision Language Model with RNN-Gated Chunked KV Cache**](https://arxiv.org/pdf/2509.21354) | 2025-09 | Chunk-based KV cache compression with recurrent utility gating for scalable VLA inference. | - |
| [![Star](https://img.shields.io/github/stars/PKU-XLab/TTF-VLA.svg?style=social&label=Star)](https://github.com/PKU-XLab/TTF-VLA) <br> [**TTF-VLA: Temporal Token Fusion via Pixel-Attention Integration for VLA Models**](https://arxiv.org/pdf/2508.19257) | 2025-08 | Training-free temporal token fusion with selective historical feature integration for robust VLA inference. | [Code](https://github.com/PKU-XLab/TTF-VLA) |
| [![Star](https://img.shields.io/github/stars/siyuhsu/vla-cache.svg?style=social&label=Star)](https://github.com/siyuhsu/vla-cache) [![Publish](https://img.shields.io/badge/Conference-NeurIPS%202025-blue)]() <br> [**VLA-Cache: Efficient Vision-Language-Action Manipulation via Adaptive Token Caching**](https://arxiv.org/pdf/2502.02175) | 2025-02 | Adaptive visual token KV caching and reuse across frames for training-free VLA acceleration. | [Code](https://github.com/siyuhsu/vla-cache) / [Website](https://vla-cache.github.io/) |

---

<a id="efficient-action-generation"></a>
## 🤖 Efficient Action Generation

Reduce redundancy in action modeling.
Instead of modifying perception or architecture, this category focuses on how actions are represented, generated, or reasoned about — affecting latency, planning quality, and output dimensionality.

<a id="raw-action-generation"></a>
## 🕹️ 3.1 Raw Action Generation
> Improve efficiency by redesigning action representation or decoding mechanisms without inserting explicit reasoning steps.
This includes action tokenizers, discrete/flow-based decoders, block-wise or parallel decoding for actions, and compact action parameterizations.

| Title | Date | Key Idea | Resources |
|:--|:--:|:--|:--:|
| [![Star](https://img.shields.io/github/stars/HHYHRHY/MM-ACT.svg?style=social&label=Star)](https://github.com/HHYHRHY/MM-ACT) <br> [**MM-ACT: Learn from Multimodal Parallel Generation to Act**](https://arxiv.org/pdf/2512.00975) <br><sub>Secondary: `4.2 Inference Efficiency Techniques`</sub> | 2025-12 | Unified multimodal VLA with parallel action decoding for efficient generation. | [Code](https://github.com/HHYHRHY/MM-ACT) |
| [![Publish](https://img.shields.io/badge/Conference-ICLR%202026-blue)]() <br> [**FASTER: TOWARD EFFICIENT AUTOREGRESSIVE VISION LANGUAGE ACTION MODELING VIA NEURAL ACTION TOKENIZATION**](https://arxiv.org/pdf/2512.04952) | 2025-12 | Learnable high-compression action tokenizer with block-wise autoregressive decoding for efficient VLA. | - |
| [![Star](https://img.shields.io/github/stars/YuhuaJiang2002/AsyncVLA.svg?style=social&label=Star)](https://github.com/YuhuaJiang2002/AsyncVLA) <br> [**AsyncVLA: Asynchronous Flow Matching for Vision-Language-Action Models**](https://arxiv.org/pdf/2511.14148) | 2025-11 | Asynchronous flow matching with action-aware non-uniform scheduling and selective refinement for efficient VLA action generation. | [Code](https://github.com/YuhuaJiang2002/AsyncVLA) |
| [![Star](https://img.shields.io/github/stars/OpenHelix-Team/UD-VLA.svg?style=social&label=Star)](https://github.com/OpenHelix-Team/UD-VLA) [![Publish](https://img.shields.io/badge/Conference-ICLR%202026-blue)]() <br> [**UNIFIED DIFFUSION VLA: VISION-LANGUAGE-ACTION MODEL VIA JOINT DISCRETE DENOISING DIFFUSION PROCESS**](https://arxiv.org/pdf/2511.01718) | 2025-11 | Unified joint diffusion denoising (JD3P) for synchronous image–action generation with faster inference. | [Code](https://github.com/OpenHelix-Team/UD-VLA/tree/main) / [Website](https://irpn-eai.github.io/UD-VLA.github.io/) |
| [**OMNISAT: COMPACT ACTION TOKEN, FASTER AUTOREGRESSION**](https://arxiv.org/pdf/2510.09667) | 2025-10 | Compact residual-quantized action tokenizer (OmniSAT) for shortening autoregressive VLA action sequences. | [Website](https://annoymoushh.github.io/) |
| [![Star](https://img.shields.io/github/stars/dunnolab/NinA.svg?style=social&label=Star)](https://github.com/dunnolab/NinA) <br> [**NinA: Normalizing Flows in Action. Training VLA Models with Normalizing Flows**](https://arxiv.org/pdf/2508.16845) | 2025-08 | Normalizing Flow action decoder enabling one-shot sampling for fast VLA inference. | [Code](https://github.com/dunnolab/NinA/) |
| [**DISCRETE DIFFUSION VLA: BRINGING DISCRETE DIFFUSION TO ACTION DECODING IN VISION-LANGUAGE-ACTION POLICIES**](https://arxiv.org/pdf/2508.20072) | 2025-08 | Unified discrete diffusion decoder with adaptive parallel refinement for efficient VLA action generation. | - |
| [![Star](https://img.shields.io/github/stars/kscalelabs/evla.svg?style=social&label=Star)](https://github.com/kscalelabs/evla) <br> [**EdgeVLA: Efficient Vision-Language-Action Models**](https://arxiv.org/pdf/2507.14049) <br><sub>Secondary: `1.1 Static Backbone Selection`</sub> | 2025-07 | Non-autoregressive end-effector prediction and small language model substitution for faster VLA inference. | [Code](https://github.com/kscalelabs/evla) |
| [![Star](https://img.shields.io/github/stars/LukeLIN-web/VOTE.svg?style=social&label=Star)](https://github.com/LukeLIN-web/VOTE) <br> [**VOTE: Vision-Language-Action Optimization with Trajectory Ensemble Voting**](https://arxiv.org/pdf/2507.05116) <br><sub>Secondary: `4.2 Inference Efficiency Techniques`</sub> | 2025-07 | Parallel low-token action generation with voting-based inference ensemble for ultra-fast VLA deployment. | [Code](https://github.com/LukeLIN-web/VOTE) |
| [![Star](https://img.shields.io/github/stars/LukeLIN-web/VOTE.svg?style=social&label=Star)](https://github.com/LukeLIN-web/VOTE) [![Publish](https://img.shields.io/badge/Conference-NeurIPS%202025-blue)]() <br> [**Real-Time Execution of Action Chunking Flow Policies**](https://arxiv.org/pdf/2506.07339) | 2025-06 | Asynchronous real-time chunking with freeze-and-inpaint strategy for latency-robust VLA inference. | [Code](https://github.com/LukeLIN-web/VOTE) / [Website](https://www.pi.website/research/real_time_chunking) |
| [**FAST: Efficient Action Tokenization for Vision-Language-Action Models**](https://arxiv.org/pdf/2501.09747) | 2025-01 | Frequency-space action tokenization (FAST) for efficient autoregressive VLA training on high-frequency control. | [Website](https://www.pi.website/research/fast) |
| [![Star](https://img.shields.io/github/stars/tonyzhaozh/aloha.svg?style=social&label=Star)](https://github.com/tonyzhaozh/aloha) [![Publish](https://img.shields.io/badge/Conference-RSS%202023-blue)]() <br> [**Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware**](https://arxiv.org/pdf/2304.13705) | 2023-04 | Action Chunking with Transformers (ACT) for temporally consistent multi-step action generation. | [Code](https://github.com/tonyzhaozh/aloha) / [Website](https://tonyzhaozh.github.io/aloha/) |


<a id="reasoning-aware-action-generation"></a>
<a id="text-reasoning"></a>
<a id="visual-reasoning"></a>
## 💡 3.2 Reasoning-Aware Action Generation
Reasoning can take the form of textual CoT, latent reasoning, visual subgoals, or world-model rollouts, and may be optimized via caching or latent compression.

| Title | Date | Key Idea | Resources |
|:--|:--:|:--|:--:|
| [![Star](https://img.shields.io/github/stars/LoveJu1y/LaRA-VLA.svg?style=social&label=Star)](https://github.com/LoveJu1y/LaRA-VLA) <br> [**Latent Reasoning VLA: Latent Thinking and Prediction for Vision-Language-Action Models**](https://www.arxiv.org/pdf/2602.01166) | 2026-02 | Latent embodied reasoning replacing explicit CoT for low-latency VLA control. | [Code](https://github.com/LoveJu1y/LaRA-VLA) / [Website](https://loveju1y.github.io/Latent-Reasoning-VLA/) |
| [**Fast-ThinkAct: Efficient Vision-Language-Action Reasoning via Verbalizable Latent Planning**](https://arxiv.org/pdf/2601.09708) | 2026-01 | Latent chain-of-thought distillation for compact and low-latency reasoning in VLA. | [Website](https://jasper0314-huang.github.io/fast-thinkact/) |
| [**Latent Chain-of-Thought World Modeling for End-to-End Autonomous Driving**](https://arxiv.org/pdf/2512.10226) <br><sub>Domain: `AD (Autonomous Driving)`</sub> | 2025-12 | Key Idea: Action-aligned latent chain-of-thought reasoning for efficient end-to-end driving VLA. | - |
| [![Publish](https://img.shields.io/badge/Conference-ICLR%202026-blue)]() <br> [**MEMER: SCALING UP MEMORY FOR ROBOT CONTROL VIA EXPERIENCE RETRIEVAL**](https://arxiv.org/pdf/2510.20328) <br><sub>Secondary: `2.2 Temporal Sharing and Reuse`</sub> | 2025-10 | Hierarchical VLA with memory-aware high-level policy selecting relevant keyframes to guide low-level execution. | [Website](https://jen-pan.github.io/memer/) |
| [**Training Strategies for Efficient Embodied Reasoning**](https://arxiv.org/pdf/2505.08243) | 2025-05 | Lightweight alternative robot reasoning recipes enabling faster CoT-based VLA inference. | [Website](https://ecot-lite.github.io/) |
| [![Publish](https://img.shields.io/badge/Conference-CVPR%202025-blue)]() <br> [**CoT-VLA: Visual Chain-of-Thought Reasoning for Vision-Language-Action Models**](https://arxiv.org/pdf/2503.22020) | 2025-03 | Explicit visual chain-of-thought reasoning via future frame prediction before action generation. | [Website](https://cot-vla.github.io/) |

---

<a id="efficient-training-and-inference"></a>
## 🛠️ Efficient Training and Inference

Optimize how the model is trained or executed, without fundamentally altering its structure or representation.
This category addresses optimization redundancy and execution scheduling rather than architectural or modeling changes.

<a id="training-efficiency-techniques"></a>
## 📈 4.1 Training Efficiency Techniques
> Reduce adaptation cost through improved optimization strategies.
Includes parameter-efficient fine-tuning (PEFT), distillation, data distillation/selection, quantization-aware training and other methods that preserve performance under reduced training cost.

| Title | Date | Key Idea | Resources |
|:--|:--:|:--|:--:|
| [![Star](https://img.shields.io/github/stars/AutoLab-SAI-SJTU/QVLA.svg?style=social&label=Star)](https://github.com/AutoLab-SAI-SJTU/QVLA) [![Publish](https://img.shields.io/badge/Conference-ICLR%202026-blue)]() <br> [**QVLA: Not All Channels Are Equal in Vision-Language-Action Model's Quantization**](https://arxiv.org/pdf/2602.03782) | 2026-02 | Action-centric channel-wise mixed-bit quantization for efficient and robust VLA deployment. | [Code](https://github.com/AutoLab-SAI-SJTU/QVLA) |
| [**RL-VLA3: REINFORCEMENT LEARNING VLA AC-CELERATING VIA FULL ASYNCHRONISM**](https://arxiv.org/pdf/2602.05765) | 2026-02 | Fully-asynchronous RL training pipeline for high-throughput VLA policy optimization. | - |
| [**TwinRL-VLA: Digital Twin-Driven Reinforcement Learning for Real-World Robotic Manipulation**](https://arxiv.org/pdf/2602.09023) | 2026-02 | Digital twin-guided collaborative online RL for efficient real-world VLA exploration. | [Website](https://sites.google.com/view/twinrl/twinrl) |
| [**HBVLA: Pushing 1-Bit Post-Training Quantization for Vision-Language-Action Models**](https://arxiv.org/pdf/2602.13710) | 2026-02 | Policy-aware Hessian-guided 1-bit binarization framework for efficient VLA deployment. | - |
| [**Shallow-π: Knowledge Distillation for Flow-based VLAs**](https://arxiv.org/pdf/2601.20262) <br><sub>Secondary: `1.2 Dynamic Computation Pathways`</sub> | 2026-01 | Key Idea: Knowledge distillation-based transformer depth reduction for flow-based VLA models. | [Website](https://icsl-jeon.github.io/shallow-pi/) |
| [**Towards Accessible Physical AI: LoRA-Based Fine-Tuning of VLA Models for Real-World Robot Control**](https://arxiv.org/pdf/2512.11921) | 2025-12 | LoRA- and quantization-based resource-efficient fine-tuning of large VLA models for low-cost deployment. | - |
| [![Publish](https://img.shields.io/badge/Conference-AAAI%202026-blue)]() <br> [**FT-NCFM: An Influence-Aware Data Distillation Framework for Efficient VLA Models**](https://arxiv.org/pdf/2511.16233) | 2025-11 | Data-centric generative data distillation (FT-NCFM) for efficient VLA training with compact coresets. | - |
| [![Star](https://img.shields.io/github/stars/gooogleshanghai/ActDistill.svg?style=social&label=Star)](https://github.com/gooogleshanghai/ActDistill) <br> [**ActDistill: General Action-Guided Self-Derived Distillation for Efficient Vision-Language-Action Models**](https://arxiv.org/pdf/2511.18082) <br><sub>Secondary: `1.2 Dynamic Computation Pathways`</sub> | 2025-11 | Action-guided distillation with dynamically routed lightweight student for efficient VLA inference. | [Code](https://github.com/gooogleshanghai/ActDistill) |
| [![Star](https://img.shields.io/github/stars/Tencent/VITA.svg?style=social&label=Star)](https://github.com/Tencent/VITA) <br> [**VITA-VLA: Efficiently Teaching Vision-Language Models to Act via Action Expert Distillation**](https://arxiv.org/pdf/2510.09607) | 2025-10 | Distillation-based transfer of action modeling from small policy to VLM for efficient VLA training. | [Code](https://github.com/Tencent/VITA/tree/VITA-VLA) / [Website](https://ltbai.github.io/VITA-VLA/) |
| [![Star](https://img.shields.io/github/stars/ustcwhy/BitVLA.svg?style=social&label=Star)](https://github.com/ustcwhy/BitVLA) <br> [**BitVLA: 1-bit Vision-Language-Action Models for Robotics Manipulation**](https://arxiv.org/pdf/2506.07530) | 2025-06 | Ternary 1-bit Vision-Language-Action model with distillation-aware compression for memory-efficient deployment. | [Code](https://github.com/ustcwhy/BitVLA) |
| [**Saliency-Aware Quantized Imitation Learning for Efficient Robotic Control**](https://arxiv.org/pdf/2505.15304) | 2025-05 | Saliency-aware quantization-aware training for low-bit efficient VLA deployment. | [Website](https://aiha-lab.github.io/sqil/) |
| [![Star](https://img.shields.io/github/stars/moojink/openvla-oft.svg?style=social&label=Star)](https://github.com/moojink/openvla-oft) [![Publish](https://img.shields.io/badge/Conference-RSS%202025-blue)]() <br> [**Fine-Tuning VLA Models: Optimizing Speed and Success**](https://arxiv.org/pdf/2502.19645) <br><sub>Secondary: `4.2 Inference Efficiency Techniques`</sub> | 2025-02 | Optimized fine-tuning recipe with parallel decoding and action chunking for high-throughput VLA adaptation. | [Code](https://github.com/moojink/openvla-oft) / [Website](https://openvla-oft.github.io/) |


<a id="inference-efficiency-techniques"></a>
## ⚡ 4.2 Inference Efficiency Techniques
> Improve runtime latency through decoding or system-level scheduling optimizations, without modifying the model structure.
Includes speculative decoding, parallel decoding, pipelining, chunk scheduling, action reuse, runtime early-exit policies, and KV scheduling strategies.

| Title | Date | Key Idea | Resources |
|:--|:--:|:--|:--:|
| [![Star](https://img.shields.io/github/stars/XiaomiRobotics/Xiaomi-Robotics-0.svg?style=social&label=Star)](https://github.com/XiaomiRobotics/Xiaomi-Robotics-0) <br> [**Xiaomi-Robotics-0: An Open-Sourced Vision-Language-Action Model with Real-Time Execution**](https://arxiv.org/pdf/2602.12684) | 2026-02 | Asynchronous execution training and deployment-aligned action chunk rollout for real-time VLA control. | [Code](https://github.com/XiaomiRobotics/Xiaomi-Robotics-0) / [Website](https://xiaomi-robotics-0.github.io/) |
| [![Star](https://img.shields.io/github/stars/hzxie/DynamicVLA.svg?style=social&label=Star)](https://github.com/hzxie/DynamicVLA) <br> [**DynamicVLA: A Vision-Language-Action Model for Dynamic Object Manipulation**](https://arxiv.org/pdf/2601.22153) <br><sub>Secondary: `1.1 Static Backbone Selection`</sub> | 2026-01 | Continuous inference with latent-aware action streaming for low-latency dynamic VLA control. | [Code](https://github.com/hzxie/DynamicVLA) / [Website](https://www.infinitescript.com/project/dynamic-vla/) |
| [**ActionFlow: A Pipelined Action Acceleration for Vision Language Models on Edge**](https://arxiv.org/pdf/2512.20276) | 2025-12 | Cross-request pipelined scheduling with unified KV buffer for real-time VLA inference on edge devices. | [Website](https://anonymous.4open.science/r/ActionFlow-1D47/README.md) |
| [**DeeAD: Dynamic Early Exit of Vision-Language Action for Efficient Autonomous Driving**](https://arxiv.org/pdf/2511.20720) <br><sub>Domain: `AD (Autonomous Driving)`</sub> | 2025-11 | Training-free action-guided early-exit with adaptive layer skipping for faster VLA inference. | - |
| [**Don’t Run with Scissors: Pruning Breaks VLA Models but They Can Be Recovered**](https://arxiv.org/pdf/2510.08464) | 2025-10 | Post-pruning weight-space interpolation recovery for efficient and safe VLA inference. | [Website](https://gluestick-vla.github.io/) |
| [![Star](https://img.shields.io/github/stars/ecdine/SQAP-VLA.svg?style=social&label=Star)](https://github.com/ecdine/SQAP-VLA) <br> [**SQAP-VLA: A Synergistic Quantization-Aware Pruning Framework**](https://arxiv.org/pdf/2509.09090) <br><sub>Secondary: `2.1 Selective Feature Processing`</sub> | 2025-09 | Joint quantization-aware visual token pruning for training-free holistic VLA inference acceleration. | [Code](https://github.com/ecdine/SQAP-VLA) |
| [![Star](https://img.shields.io/github/stars/PineTreeWss/SpecVLA.svg?style=social&label=Star)](https://github.com/PineTreeWss/SpecVLA) [![Publish](https://img.shields.io/badge/Conference-EMNLP%202025-blue)]() <br> [**Spec-VLA: Speculative Decoding for Vision-Language-Action Models with Relaxed Acceptance**](https://arxiv.org/pdf/2507.22424) | 2025-07 | Speculative decoding with relaxed acceptance for accelerated VLA action generation. | [Code](https://github.com/PineTreeWss/SpecVLA) |
| [![Publish](https://img.shields.io/badge/Conference-ICLR%202026-blue)]() <br> [**SP-VLA: A joint model scheduling and token-pruning approach for VLA model acceleration**](https://arxiv.org/pdf/2506.12723) <br><sub>Secondary: `2.1 Selective Feature Processing`</sub> | 2025-06 | Action-aware model scheduling with spatio-semantic dual token pruning for efficient sequential VLA inference. | - |
| [![Publish](https://img.shields.io/badge/Conference-ICRA%202026-blue)]() <br> [**Fast ECoT: Efficient Embodied Chain-of-Thought via Thoughts Reuse**](https://arxiv.org/pdf/2506.07639) <br><sub>Secondary: `2.2 Temporal Sharing and Reuse`</sub> | 2025-06 | Inference-time acceleration of Embodied CoT via reasoning cache reuse, parallel generation, and asynchronous scheduling. | - |
| [![Star](https://img.shields.io/github/stars/OpenHelix-Team/CEED-VLA.svg?style=social&label=Star)](https://github.com/OpenHelix-Team/CEED-VLA) [![Publish](https://img.shields.io/badge/Conference-NeurIPS%202025-blue)]() <br> [**CEED-VLA: Consistency Vision-Language-Action Model with Early-Exit Decoding**](https://arxiv.org/pdf/2506.13725) <br><sub>Secondary: `4.1 Training Efficiency Techniques`</sub> | 2025-06 | Consistency-distilled multi-token prediction with early-exit decoding for accelerated VLA inference. | [Code](https://github.com/OpenHelix-Team/CEED-VLA) / [Website](https://irpn-eai.github.io/CEED-VLA/) |
| [**Accelerating VLA Models Integrated with Action Chunking via Parallel Decoding**](https://arxiv.org/pdf/2503.02310) | 2025-03 | Parallel fixed-point decoding for accelerating action-chunked VLA models without retraining. | - |


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
