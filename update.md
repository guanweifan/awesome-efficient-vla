# Efficient VLA Update Notes

## At a glance

- Total papers covered in this update: **19**
- Main themes in this batch:
  - efficient WAM variants that reduce future-imagination cost through compact video experts, latent/image-editing world context, register tokens, or persistent memory
  - faster action generation through block diffusion, residual caching, speed-controllable execution, and one-to-few-step mean-flow policies
  - asynchronous slow-fast architectures for WAM and multimodal VLA control
  - reasoning-efficiency methods that gate language generation or adaptively stop latent reasoning
  - training, compression, and deployment efficiency through multi-chunk supervision, distillation, mixed-precision quantization, layer pruning, and portable C++ runtime support

## TempoVLA

- **Title:** TempoVLA: Learning Speed-Controllable Vision-Language-Action Policies
- **Short Name:** TempoVLA
- **Link:** https://arxiv.org/pdf/2606.06491
- **Primary Category:** 3.1 Raw Action Generation
- **Core Idea:** Enable a single VLA policy to control robot execution speed by retiming demonstrations with Variable-Speed Trajectory Augmentation and conditioning the policy on target speed.
- **Why this category:** This is a boundary Efficient VLA entry focused on embodied execution efficiency rather than model-side compute reduction. TempoVLA does not reduce inference latency, FLOPs, KV cache size, or denoising steps. Its efficiency value comes from speed-controllable action generation: the same policy can accelerate low-risk segments and slow down around contact-sensitive segments. The primary category is 3.1 because the method changes the action trajectory distribution through demonstration retiming and speed-conditioned action chunks. It should not be categorized as 4.2 because it does not introduce a runtime serving scheduler or deployment pipeline.

## TBD-VLA

- **Title:** TBD-VLA: Temporal Block Diffusion Vision Language Action Model
- **Short Name:** TBD-VLA
- **Link:** https://arxiv.org/pdf/2606.07895
- **Code:** https://github.com/TBD-VLA/lerobot
- **Primary Category:** 3.1 Raw Action Generation
- **Second:** 4.2 Inference Efficiency Techniques
- **Core Idea:** Generate discrete action tokens with block diffusion by denoising tokens in parallel within temporal blocks while preserving autoregressive dependencies across blocks.
- **Why this category:** This is a typical discrete-action generation acceleration paper. It targets the latency of next-token autoregressive VLA action decoding by partitioning action sequences into temporal blocks, denoising tokens in parallel inside each block, and preserving temporal dependency across blocks. The primary category is 3.1 because the core mechanism directly changes action-token generation. It also receives 4.2 because its real-time chunking strategy asynchronously prepares the next action chunk while the current chunk is executing. It should not be primarily categorized as 2.2 because prefix KV cache is only an auxiliary decoding acceleration detail rather than the main temporal feature-reuse mechanism.

## vla.cpp

- **Title:** vla.cpp: A Unified Inference Runtime for Vision-Language-Action Models
- **Short Name:** vla.cpp
- **Link:** https://arxiv.org/pdf/2606.08094
- **Code:** https://github.com/VinRobotics/vla.cpp
- **Primary Category:** 4.2 Inference Efficiency Techniques
- **Core Idea:** Build a portable llama.cpp / ggml-based C++ runtime that natively serves flow-matching and diffusion VLA inference patterns across multiple VLA architectures and edge hardware tiers.
- **Why this category:** This is a deployment and inference-runtime entry. The paper targets the mismatch between VLA deployment needs and Python / PyTorch workstation-oriented inference stacks, and supports cached vision-language prefixes, cross-attending action experts, multi-step solvers, GGUF packaging, and hardware tiers from consumer GPUs to embedded modules. The primary category is 4.2 because the efficiency contribution is runtime portability, memory reduction, and edge-oriented inference execution. It should not be categorized as 3.1 because it does not change the action-generation algorithm itself.

## Light-WAM

- **Title:** Light-WAM: Efficient World Action Models with State-Fusion Action Decoding
- **Short Name:** Light-WAM
- **Link:** https://arxiv.org/pdf/2606.08242
- **Code:** https://github.com/L1ziang/Light-WAM
- **Primary Category:** 3.1 Raw Action Generation
- **Second:** 1.1 Static Backbone Selection
- **Core Idea:** Build a lightweight WAM with a compact video backbone, downsampled latent-space video supervision, and a single-forward StateFusionActionExpert for efficient action decoding.
- **Why this category:** This paper fits Efficient VLA / Efficient WAM because it targets the training and inference cost of large generative WAM designs. The primary category is 3.1 because the most direct efficiency mechanism is action decoding: StateFusionActionExpert fuses multi-layer backbone states and predicts action chunks in a single forward pass instead of using a heavier generative action expert or multi-step action decoder. It also receives 1.1 because the compact video backbone and small trainable parameter count are explicit lightweight architecture choices.

## BLUE

- **Title:** BLUE: Toward Better Language Use in Efficient Vision-Language-Action Models for Autonomous Driving
- **Short Name:** BLUE
- **Link:** https://arxiv.org/pdf/2606.08684
- **Code:** https://github.com/George-Ling3/BLUE
- **Primary Category:** 3.2 Reasoning-Aware Action Generation
- **Second:** 1.2 Dynamic Computation Pathways
- **Tag:** Autonomous Driving
- **Core Idea:** Train a lightweight gate on frozen VLA hidden states to decide per frame whether to generate language or directly predict actions, preserving language benefits at lower inference cost.
- **Why this category:** This paper fits Efficient VLA because it targets unnecessary language generation in autonomous-driving VLA policies. BLUE keeps language reasoning only for frames where it improves driving behavior and otherwise routes directly to action prediction. The primary category is 3.2 because the main mechanism is efficient use of language reasoning before action. It also receives 1.2 because the per-frame gate dynamically chooses between the language-generation path and the direct-action path. It should not be primarily categorized as 4.2 because the routing is part of the model behavior rather than an external serving scheduler.

## C3ache

- **Title:** C³ache: Accelerating World Action Models with Cross Inference Chunk Cache
- **Short Name:** C³ache
- **Link:** https://arxiv.org/pdf/2606.08962
- **Primary Category:** 3.1 Raw Action Generation
- **Second:** 2.2 Temporal Sharing and Reuse
- **Core Idea:** Cache and reuse action-expert residuals across consecutive inference chunks at matched denoising steps, exploiting temporal redundancy in smooth WAM rollouts without retraining.
- **Why this category:** This paper fits Efficient VLA / Efficient WAM because it reduces repeated computation in multi-step WAM action generation. The primary category is 3.1 because the cache acts directly on the flow-style action-chunk denoising process, skipping a large fraction of action-expert computation. It also receives 2.2 because the acceleration relies on cross-chunk temporal redundancy and residual reuse across adjacent control chunks. It should not be primarily categorized as 4.2 because the key mechanism is not an external serving framework or scheduler.

## AHA-WAM

- **Title:** AHA-WAM: Asynchronous Horizon-Adaptive World-Action Modeling with Observation-Guided Context Routing
- **Short Name:** AHA-WAM
- **Link:** https://arxiv.org/pdf/2606.09811
- **Code:** https://github.com/serene-sivy/AHA-WAM
- **Primary Category:** 1.3 Dual-system Design
- **Second:** 2.2 Temporal Sharing and Reuse
- **Core Idea:** Decouple WAM inference into a low-frequency video-DiT world planner with reusable rolling KV context and a high-frequency action-DiT executor that routes this context for closed-loop action generation.
- **Why this category:** This paper fits Efficient VLA / Efficient WAM because it removes redundant high-frequency world prediction from closed-loop control. The primary category is 1.3 because the core design is an asynchronous slow planner plus fast executor architecture. It also receives 2.2 because the planner maintains rolling memory and reusable cached context across action chunks. It should not be primarily categorized as 3.1 because the main contribution is not action-token design or denoising-step reduction, but the architectural separation between world planning and action execution.

## Efficient-WAM

- **Title:** Efficient-WAM: A 1B-Parameter World-Action Model with Low-Cost Future Imagination
- **Short Name:** Efficient-WAM / Efficient-WAM-RT
- **Link:** https://arxiv.org/pdf/2606.10040
- **Code:** https://github.com/jiajun613/Efficient-WAM
- **Primary Category:** 1.1 Static Backbone Selection
- **Second:** 3.1 Raw Action Generation
- **Core Idea:** Reduce WAM future-imagination cost with a compact video expert, token-sparse low-resolution future latents, and asymmetric video-action denoising that allocates fewer sampling steps to video than to actions.
- **Why this category:** This is a core Efficient WAM entry. It targets the cost of photorealistic future prediction by shifting toward action-centric future imagination, where the future branch preserves action-relevant geometry, motion, and contact cues rather than high-fidelity video. The primary category is 1.1 because the compact 1B video expert is the foundational efficiency design. It also receives 3.1 because token-sparse future latents and asymmetric denoising reduce the generation budget for future video / action chunks. It should not be primarily categorized as 4.2 because it is a model-design contribution rather than an external runtime.

## Next Forcing

- **Title:** Next Forcing: Causal World Modeling with Multi-Chunk Prediction
- **Short Name:** Next Forcing
- **Link:** https://arxiv.org/pdf/2606.11187
- **Code:** https://github.com/gangweix/next-forcing
- **Primary Category:** 4.1 Training Efficiency Techniques
- **Second:** 3.1 Raw Action Generation
- **Core Idea:** Add lightweight multi-chunk prediction modules to autoregressive video WAMs so future chunks receive dense causal supervision during training and can be predicted in parallel at inference.
- **Why this category:** This paper fits Efficient VLA / Efficient WAM because it targets both slow convergence and slow iterative video prediction in autoregressive world models. The primary category is 4.1 because the main contribution is a training objective with denser multi-horizon supervision. It also receives 3.1 because the same modules can be retained at inference to predict future chunks in parallel and reduce rollout latency. It should not be categorized as 2.2 because it does not rely on KV cache, history compression, or temporal feature reuse.

## DAM-VLA

- **Title:** DAM-VLA: Decoupled Asynchronous Multimodal Vision Language Action Model
- **Short Name:** DAM-VLA
- **Link:** https://arxiv.org/pdf/2606.12105
- **Primary Category:** 1.3 Dual-system Design
- **Second:** 2.2 Temporal Sharing and Reuse
- **Core Idea:** Maintain per-modality latent buffers refreshed at each sensor's native rate and let a high-frequency action head continuously read them, avoiding synchronous oversampling of slow modalities.
- **Why this category:** This paper fits Efficient VLA because it targets the inefficiency of forcing vision, language, proprioception, and force / torque streams into one synchronized processing clock. The primary category is 1.3 because it is an asynchronous multi-rate system design that decouples slow visual / language memory from high-frequency reactive control. It also receives 2.2 because per-modality latent buffers reuse slower modality features across many action-control steps. It should not be primarily categorized as 4.2 because the asynchronous structure is inside the model architecture rather than an external serving pipeline.

## RT-VLA

- **Title:** RT-VLA: Real-Time Vision-Language-Action Models via Knowledge Distillation
- **Short Name:** RT-VLA
- **Link:** https://arxiv.org/pdf/2606.14010
- **Primary Category:** 4.1 Training Efficiency Techniques
- **Tag:** Autonomous Driving
- **Core Idea:** Distill the driving and language-reasoning capabilities of a large SimLingo teacher into a compact student model for real-time autonomous-driving VLA inference.
- **Why this category:** This paper fits Efficient VLA because it reduces autonomous-driving VLA inference cost through teacher-student compression. The primary category is 4.1 because the mechanism is multi-level supervised distillation of visual features, query representations, waypoint predictions, and language logits into a compact student. It should not be categorized as 3.2 because language reasoning is mainly preserved through training-time distillation and post-hoc explanation, not through an online efficient reasoning mechanism.

## WAM4D

- **Title:** WAM4D: Fast 4D World Action Model via Spatial Register Tokens
- **Short Name:** WAM4D
- **Link:** https://arxiv.org/pdf/2606.14048
- **Primary Category:** 2.1 Selective Feature Processing
- **Second:** 4.1 Training Efficiency Techniques
- **Core Idea:** Use lightweight spatial register tokens as training-time future-depth readouts to transfer 4D geometric priors into a WAM, then remove the geometry branch for lightweight action inference.
- **Why this category:** This is a boundary Efficient VLA / Efficient WAM entry. It targets the cost of dense 4D geometry decoding by using a small set of register tokens for training-time geometry supervision and removing the geometry branch at deployment. The primary category is 2.1 because the efficiency mechanism replaces dense spatial geometry representation with compact register-token readouts. It also receives 4.1 because the geometric prior is transferred through training-time supervision. It should not be categorized as 3.1 because it does not directly reduce action denoising steps, action tokenization cost, or action-decoder calls.

## ReactVLA

- **Title:** ReactVLA: Fast and Lightweight Reactive Robot Manipulation via Improved Mean Flow Action Generation
- **Short Name:** ReactVLA
- **Link:** https://arxiv.org/pdf/2606.14255
- **Primary Category:** 3.1 Raw Action Generation
- **Second:** 1.2 Dynamic Computation Pathways
- **Core Idea:** Replace iterative diffusion-style action sampling with improved Mean Flow one-to-few-step action generation, supported by dynamic depth-wise Attention Residual routing for low-latency reactive control.
- **Why this category:** This is a typical action-generation efficiency paper. It targets the latency of diffusion-based VLA policies by replacing multi-step iterative sampling with one-to-few-step finite-interval transport prediction. The primary category is 3.1 because the main speedup comes from reducing the action-generation step budget. It also receives 1.2 because Attention Residual routing dynamically selects useful multimodal intermediate representations across depth. It should not be primarily categorized as 4.2 because it is not an external runtime scheduler.

## AVA-VLA

- **Title:** Think Less, Act Early: Reinforced Latent Reasoning with Early Exit in Vision-Language-Action Models
- **Short Name:** AVA-VLA
- **Link:** https://arxiv.org/pdf/2606.15099
- **Primary Category:** 3.2 Reasoning-Aware Action Generation
- **Second:** 1.2 Dynamic Computation Pathways
- **Core Idea:** Replace explicit textual CoT with RL-denoised latent reasoning and adaptively stop reasoning through an early-exit gate to reduce inference latency.
- **Why this category:** This paper fits Efficient VLA as a reasoning-efficiency entry. It targets the token-by-token latency and error propagation of explicit textual CoT by moving reasoning into continuous latent variables and stopping latent reasoning adaptively when confidence is sufficient. The primary category is 3.2 because the mechanism reduces the cost of intermediate reasoning before action. It also receives 1.2 because early exit creates a dynamic reasoning-depth pathway. It should not be categorized as 3.1 because it does not change action tokenization, action chunking, or diffusion / flow action decoding.

## LaWAM

- **Title:** LaWAM: Latent World Action Models for Efficient Dynamics-Aware Robot Policies
- **Short Name:** LaWAM
- **Link:** https://arxiv.org/pdf/2606.15768
- **Code:** https://github.com/RLinf/LaWAM
- **Primary Category:** 2.1 Selective Feature Processing
- **Second:** 3.1 Raw Action Generation
- **Core Idea:** Replace pixel-space future video generation with compact latent visual subgoals, exposing predictive dynamics to the action policy while avoiding expensive reconstructed future rollouts.
- **Why this category:** This paper fits Efficient VLA / Efficient WAM because it reduces world-model future representation cost. Instead of generating pixel-space future video, LaWAM predicts future observation features in the latent space of a frozen visual foundation model and injects them as compact visual subgoals for action generation. The primary category is 2.1 because the main efficiency mechanism reduces visual bandwidth and avoids redundant pixel reconstruction. It also receives 3.1 because these latent subgoals directly support action-chunk generation.

## ImageWAM

- **Title:** ImageWAM: Do World Action Models Really Need Video Generation, or Just Image Editing?
- **Short Name:** ImageWAM
- **Link:** https://arxiv.org/pdf/2606.19531
- **Code:** https://github.com/yuyangalin/ImageWAM
- **Primary Category:** 2.1 Selective Feature Processing
- **Second:** 3.1 Raw Action Generation
- **Core Idea:** Replace dense future-video generation in WAMs with image-editing KV caches as compact world-action context for lower-cost action prediction.
- **Why this category:** This paper fits Efficient VLA / Efficient WAM because it questions whether WAMs need dense multi-frame future video generation at all. It uses a pretrained image-editing model and feeds the intermediate editing KV caches to a flow-matching action expert without decoding future video. The primary category is 2.1 because the main efficiency mechanism reduces the bandwidth of world-model visual intermediates. It also receives 3.1 because the compact editing context directly supports lower-cost action-chunk prediction. It should not be categorized as 2.2 because the KV caches are not mainly used as cross-time memory or history reuse.

## Mix-QVLA

- **Title:** Mix-QVLA: Task-Evidence-Aware Mixed-Precision Quantization of Vision-Language-Action Models
- **Short Name:** Mix-QVLA
- **Link:** https://arxiv.org/pdf/2606.19565
- **Primary Category:** 4.1 Training Efficiency Techniques
- **Core Idea:** Use task-evidence- and time-aware sensitivity scores to allocate mixed precision across VLA layers, preserving internal decision evidence while reducing memory and BitOps.
- **Why this category:** This is a VLA quantization and compression paper. It targets memory and compute bottlenecks on resource-constrained robot platforms with mixed-precision post-training quantization. The primary category is 4.1 because PTQ, mixed-bit compression, and bit allocation are treated as training / compression efficiency techniques in this taxonomy. It should not be categorized as 2.1 or 3.1 because it does not prune visual tokens or change action-generation mechanics.

## CLP

- **Title:** Finetuning Vision-Language-Action Models Requires Fewer Layers Than You Think
- **Short Name:** CLP
- **Link:** https://arxiv.org/pdf/2606.20246
- **Primary Category:** 1.2 Dynamic Computation Pathways
- **Second:** 4.1 Training Efficiency Techniques
- **Core Idea:** Use a single-pass CKA analysis to identify representationally redundant transformer layers and permanently prune VLA depth before downstream fine-tuning.
- **Why this category:** This paper fits Efficient VLA because it reduces model depth before downstream fine-tuning and inference. The primary category is 1.2 because layer pruning is part of the dynamic-computation / depth-reduction family in this taxonomy, even though CLP itself performs structural pruning rather than per-input adaptive routing. It also receives 4.1 because pruning before fine-tuning reduces adaptation cost. It should not be categorized as 2.1 because it does not perform visual-token pruning or feature compression.

## MemoryWAM

- **Title:** MemoryWAM: Efficient World Action Modeling with Persistent Memory
- **Short Name:** MemoryWAM
- **Link:** https://arxiv.org/pdf/2606.20562
- **Primary Category:** 2.2 Temporal Sharing and Reuse
- **Second:** 4.2 Inference Efficiency Techniques
- **Core Idea:** Use hybrid persistent memory with recent frames, event-boundary anchor frames, and compact gist tokens to preserve long-range WAM context while reducing inference latency and GPU memory.
- **Why this category:** This paper fits Efficient VLA / Efficient WAM because it targets the cost of long-history conditioning. MemoryWAM combines a sliding observation window, event-boundary anchor frames, compact gist tokens, and tailored attention to preserve both recent details and long-range context without full-history attention. The primary category is 2.2 because the main mechanism is temporal memory compression and reuse. It also receives 4.2 because the persistent-memory design reduces deployment-time latency and GPU memory for long-horizon closed-loop inference.
