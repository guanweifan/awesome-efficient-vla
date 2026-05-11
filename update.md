# Efficient VLA Update Notes

## At a glance

- Total papers covered in this update: **9**
- Main themes in this batch:
  - online RL and parameter-efficient VLA adaptation
  - cloud-edge, XPU, and edge-runtime deployment efficiency
  - faster action generation and adaptive test-time compute
  - efficient VLA data augmentation and world-model visual bandwidth compression

## RLT

- **Title:** RL Token: Bootstrapping Online RL with Vision-Language-Action Models
- **Short Name:** RLT
- **Link:** https://arxiv.org/pdf/2604.23073
- **Primary Category:** 4.1 Training Efficiency Techniques
- **Core Idea:** Expose a compact RL token from a pretrained VLA so that a small actor-critic head can perform sample-efficient online RL fine-tuning without updating the full model.
- **Why this category:** This paper fits Efficient VLA as a training- and adaptation-efficiency work. Its main efficiency mechanism is not reducing VLA inference latency, visual tokens, KV cache, or denoising steps; instead, it uses a compact RL token as an information bottleneck so that a small actor-critic can run online RL on top of a frozen or mostly frozen VLA representation. The reported 3x speedup is better read as task-execution throughput rather than model inference acceleration, so 4.1 is the best primary category.

## AsyncShield

- **Title:** AsyncShield: A Plug-and-Play Edge Adapter for Asynchronous Cloud-based VLA Navigation
- **Short Name:** AsyncShield
- **Link:** https://arxiv.org/pdf/2604.24086
- **Primary Category:** 4.2 Inference Efficiency Techniques
- **Tag:** VLN
- **Core Idea:** A lightweight edge-side asynchronous adapter realigns delayed cloud VLA navigation intents through geometric re-projection and safe local control, enabling robust cloud-based VLA deployment under network latency.
- **Why this category:** This is a system- and deployment-side inference-efficiency paper. It targets cloud-based VLA inference latency, network jitter, and asynchronous execution, then uses an edge adapter, temporal pose buffer, kinematic re-projection, and high-frequency safety control to reduce execution mismatch without fine-tuning the cloud VLA. It is not visual-token pruning or action-decoding acceleration, so 4.2 is the best primary category.

## DP-Cache / V-AEFusion

- **Title:** Characterizing Vision-Language-Action Models across XPUs: Constraints and Acceleration for On-Robot Deployment
- **Short Name:** DP-Cache / V-AEFusion
- **Link:** https://arxiv.org/pdf/2604.24447
- **Primary Category:** 4.2 Inference Efficiency Techniques
- **Second:** 3.1 Raw Action Generation
- **Core Idea:** Characterize VLA inference across edge accelerators and accelerate on-robot deployment by caching redundant diffusion steps and pipelining the VLM backbone with the Action Expert.
- **Why this category:** This paper directly addresses real-time inference, cost, and energy constraints for on-robot VLA deployment, including a VLA-XPU leaderboard and training-free inference acceleration. The primary contribution is model-hardware co-characterization, edge accelerator analysis, and asynchronous pipelining between the VLM backbone and Action Expert, so 4.2 is the primary category. It also receives 3.1 as a secondary category because DP-Cache skips or caches redundant intermediate denoising steps inside the diffusion action expert.

## CF-VLA

- **Title:** CF-VLA: Efficient Coarse-to-Fine Action Generation for Vision-Language-Action Policies
- **Short Name:** CF-VLA
- **Link:** https://arxiv.org/pdf/2604.24622
- **Code:** https://github.com/EmbodiedAI-RoboTron/CF-VLA
- **Primary Category:** 3.1 Raw Action Generation
- **Core Idea:** Restructure flow-based VLA action generation into coarse action-aware initialization followed by single-step local refinement, reducing low-NFE action sampling latency.
- **Why this category:** This is a typical raw-action-generation efficiency paper. It targets the multi-step sampling cost of flow-based or diffusion-style VLA policies by replacing recovery from Gaussian noise with a coarse-to-fine process: first build an action-aware starting point, then run single-step local refinement. The efficiency gain comes from reducing action sampling steps and function evaluations rather than from system scheduling or training compression, so 3.1 is the best primary category.

## EdgeFM

- **Title:** EdgeFM: Efficient Edge Inference for Vision-Language Models
- **Short Name:** EdgeFM
- **Link:** https://arxiv.org/pdf/2604.27476
- **Primary Category:** 4.2 Inference Efficiency Techniques
- **Core Idea:** Provide a lightweight cross-platform edge inference framework with agent-tuned kernel optimizations and runtime simplification to reduce deployment latency and hardware lock-in for VLM/LLM systems, including end-to-end VLA deployment cases.
- **Why this category:** This is a deployment-oriented Efficient VLA boundary entry. Its core contribution is not VLA-internal visual-token compression, action-generation acceleration, or policy architecture redesign, but an edge inference framework that reduces runtime overhead, optimizes low-level kernels, supports cross-platform deployment, and lowers latency and deployment cost. Since it explicitly includes VLA edge deployment cases, it fits 4.2 as a system-efficiency reference.

## VLA-ATTC

- **Title:** VLA-ATTC: Adaptive Test-Time Compute for VLA Models with Relative Action Critic Model
- **Short Name:** VLA-ATTC
- **Link:** https://arxiv.org/pdf/2605.01194
- **Primary Category:** 3.2 Reasoning-Aware Action Generation
- **Second:** 4.2 Inference Efficiency Techniques
- **Core Idea:** Adaptively trigger lightweight test-time deliberation only under high action uncertainty, using a Relative Action Critic to select among parallel action candidates while preserving high control frequency.
- **Why this category:** This paper is not just about improving reasoning quality; it explicitly introduces adaptive test-time compute for VLA control. The uncertainty-based cognitive clutch triggers deliberation only in difficult states, and the lightweight Relative Action Critic selects among parallel action candidates without applying expensive deliberation at every timestep. The primary category is 3.2 because the method centers on efficient deliberation and reasoning-aware action selection. It also receives 4.2 as a secondary category because it dynamically allocates inference-time compute to preserve real-time control frequency.

## Efficient Video Transfer

- **Title:** Seeing Realism from Simulation: Efficient Video Transfer for Vision-Language-Action Data Augmentation
- **Short Name:** Efficient Video Transfer
- **Link:** https://arxiv.org/pdf/2605.02757
- **Code:** https://github.com/nanfangxiansheng/Seeing-Realism-from-Simulation
- **Primary Category:** 4.1 Training Efficiency Techniques
- **Core Idea:** Accelerate VLA data augmentation by reusing diffusion velocity predictions during video transfer and selecting a compact coreset of high-value simulated trajectories for realistic augmentation.
- **Why this category:** This paper fits Efficient VLA as a training- and data-efficiency work rather than as VLA inference acceleration. It targets the cost of building realistic VLA training data through conditional video transfer, then reduces video diffusion time through velocity caching and limits augmentation to a compact, non-redundant set of high-value simulated trajectories. It is not temporal KV/cache reuse in the VLA policy and does not modify action denoising, so 4.1 is the best primary category.

## VLA-GSE

- **Title:** VLA-GSE: Boosting Parameter-Efficient Fine-Tuning in VLA with Generalized and Specialized Experts
- **Short Name:** VLA-GSE
- **Link:** https://arxiv.org/pdf/2605.06175
- **Code:** https://github.com/YuhuaJiang2002/VLA-GSE
- **Primary Category:** 4.1 Training Efficiency Techniques
- **Core Idea:** Improve VLA adaptation under a fixed trainable-parameter budget by decomposing frozen backbone weights into generalized shared experts and routed specialized experts for parameter-efficient fine-tuning.
- **Why this category:** This is a parameter-efficient VLA fine-tuning paper. Its main goal is to reduce adaptation cost when transferring VLMs to VLA control, using SVD-initialized generalized and specialized experts while updating only a small fraction of the full model parameters. Although the method includes routed experts, the routing serves PEFT capacity rather than runtime layer skipping, early exit, dynamic depth, or inference acceleration, so 4.1 is the best primary category.

## OneWM-VLA

- **Title:** One Token Per Frame: Reconsidering Visual Bandwidth in World Models for VLA Policy
- **Short Name:** OneWM-VLA
- **Link:** https://arxiv.org/pdf/2605.07931
- **Primary Category:** 2.1 Selective Feature Processing
- **Core Idea:** Compress each view in each frame into a single latent world token through Adaptive Attention Pooling, making world-model rollout memory and token cost scalable with planning horizon.
- **Why this category:** This is a boundary entry for world-module visual bandwidth compression in VLA policies. The paper treats high-bandwidth per-frame visual streams as a bottleneck for world-model-augmented VLA planning, then uses Adaptive Attention Pooling to compress each view in each frame into one semantic latent token. This reduces long-horizon rollout memory and token cost, so 2.1 is the best primary category. It is not temporal cache reuse, action-denoising acceleration, or deployment-runtime optimization.
