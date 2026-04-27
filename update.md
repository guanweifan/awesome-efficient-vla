# Efficient VLA Update Notes

## At a glance

- Total papers covered in this update: **5**
- Main themes in this batch:
  - post-training quantization for low-cost VLA deployment
  - latent reasoning and action-generation acceleration for autonomous driving
  - sampling-time acceleration for diffusion-based policies
  - compact backbone design for lightweight VLA models

## DA-PTQ

- **Title:** DA-PTQ: Drift-Aware Post-Training Quantization for Efficient Vision-Language-Action Models
- **Short Name:** DA-PTQ
- **Link:** https://arxiv.org/pdf/2604.11572
- **Primary Category:** 4.1 Training Efficiency Techniques
- **Core Idea:** A drift-aware post-training quantization framework that compensates cross-space representation distortion and allocates mixed precision by minimizing trajectory-level motion errors for efficient VLA deployment.
- **Why this category:** The main efficiency mechanism is post-training quantization. The method uses drift-aware calibration, cross-space representation compensation, and motion-driven mixed-precision allocation to reduce memory and compute under low-bit deployment while mitigating accumulated quantization errors in sequential control. Because the contribution is centered on quantization calibration and precision allocation rather than inference scheduling or a new action-generation paradigm, 4.1 is the best primary category.

## OneVL

- **Title:** OneVL: One-Step Latent Reasoning and Planning with Vision-Language Explanation
- **Short Name:** OneVL
- **Link:** https://arxiv.org/pdf/2604.18486
- **Primary Category:** 3.2 Reasoning-Aware Action Generation
- **Tag:** Autonomous Driving
- **Core Idea:** A one-step latent Chain-of-Thought framework that compresses driving reasoning into compact vision-language latent tokens supervised by language and visual world-model decoders, enabling single-pass trajectory prediction at answer-only latency.
- **Why this category:** The central problem is reasoning-before-action in VLA-based autonomous driving. OneVL replaces explicit autoregressive Chain-of-Thought with compact latent tokens, then supervises those tokens with a language decoder and a visual world-model decoder. At inference time, the auxiliary decoders can be removed and the model can generate the trajectory in a single pass. This makes it a reasoning-aware action generation method, so 3.2 is the best primary category. Its world-model supervision and staged training support the compressed reasoning representation rather than forming an independent training-efficiency or inference-scheduling mechanism.

## SpanVLA

- **Title:** SpanVLA: Efficient Action Bridging and Learning from Negative-Recovery Samples for Vision-Language-Action Model
- **Short Name:** SpanVLA
- **Link:** https://arxiv.org/pdf/2604.19710
- **Primary Category:** 3.1 Raw Action Generation
- **Tag:** Autonomous Driving
- **Core Idea:** An autonomous-driving VLA framework that bridges autoregressive VLM reasoning to a flow-matching action expert initialized by historical trajectories, reducing action generation latency for future trajectory planning.
- **Why this category:** The main problem is action generation latency in autonomous-driving VLA. SpanVLA uses an efficient action bridge to connect VLM perception and reasoning features to a flow-matching action expert, then initializes generation from historical trajectories to produce future trajectories more efficiently. This directly targets the cost of autoregressive action decoding as action length grows, so 3.1 is the best primary category. Its GRPO-based post-training and negative-recovery samples mainly support robustness and long-tail recovery rather than defining a separate efficiency mechanism.

## FASTER

- **Title:** FASTER: Value-Guided Sampling for Fast RL
- **Short Name:** FASTER
- **Link:** https://arxiv.org/pdf/2604.19730
- **Code:** https://github.com/alexanderswerdlow/faster
- **Primary Category:** 3.1 Raw Action Generation
- **Core Idea:** A value-guided candidate filtering method that reduces the computational cost of sampling-based test-time scaling in diffusion-based policies by selecting promising action samples early in the denoising process.
- **Why this category:** If included as Efficient VLA-adjacent work, this paper is closest to 3.1 because it acts directly on diffusion-based action sampling. It reduces the compute cost of multi-candidate action generation by using early value estimates to filter weak candidates before completing the full denoising process. However, it is a general generative-RL and diffusion-policy acceleration method rather than a VLA-specific method, so it is better treated as an adjacent reference for action-generation efficiency than as a core Efficient VLA entry.

## PokeVLA

- **Title:** PokeVLA: Empowering Pocket-Sized Vision-Language-Action Model with Comprehensive World Knowledge Guidance
- **Short Name:** PokeVLA
- **Link:** https://arxiv.org/pdf/2604.20834
- **Primary Category:** 1.1 Static Backbone Selection
- **Core Idea:** A compact VLA framework that pretrains a lightweight embodied-aware VLM and injects manipulation-relevant spatial and semantic representations into action learning for efficient robot manipulation.
- **Why this category:** The main efficiency source is a pocket-sized VLA foundation model built around a compact PokeVLM backbone. The two-stage training process injects world knowledge, spatial grounding, affordance information, and multi-view geometric alignment into action learning, but the efficiency axis is still the static choice of a smaller VLA/VLM backbone. It is not a dynamic routing, token-pruning, or inference-scheduling method, so 1.1 is the best primary category.
