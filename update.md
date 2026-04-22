# Efficient VLA Update Notes

## At a glance

- Total papers covered in this update: **9**
- Main themes in this batch:
  - token pruning and selective visual processing
  - action generation acceleration
  - system-level inference and deployment efficiency
  - temporal compression for long-horizon multi-view inputs

## ETA-VLA

- **Title:** ETA-VLA: Efficient Token Adaptation via Temporal Fusion and Intra-LLM Sparsification for Vision-Language-Action Models
- **Short Name:** ETA-VLA
- **Link:** https://arxiv.org/pdf/2603.25766
- **Primary Category:** 2.2 Temporal Sharing and Reuse
- **Secondary Category:** 2.1 Selective Feature Processing
- **Tag:** Autonomous Driving
- **Core Idea:** A driving VLA framework that compresses redundant historical multi-view information with temporal fusion and dynamically prunes critical visual tokens via an intra-LLM sparse aggregator for efficient inference.
- **Why this category:** The central problem is temporal compute redundancy caused by long multi-frame and multi-view history. The Temporal Fusion Module is the main mechanism for compressing historical information while preserving key motion cues, so 2.2 is the right primary category. At the same time, the paper introduces an intra-LLM sparse aggregator that performs dynamic visual token sparsification, which is a clear token pruning mechanism and justifies 2.1 as a secondary category. Overall, this is a combined temporal compression and spatial sparsification method, but temporal history compression is the stronger organizing axis.

## Realtime-VLA V2

- **Title:** Realtime-VLA V2: Learning to Run VLAs Fast, Smooth, and Accurate
- **Short Name:** Realtime-VLA V2
- **Link:** https://arxiv.org/pdf/2603.26360
- **Code:** https://github.com/dexmal/realtime-vla-v2
- **Primary Category:** 4.2 Inference Efficiency Techniques
- **Core Idea:** A deployment-oriented VLA system report that combines calibration, planning and control, and learning-based execution-speed selection to achieve fast, smooth, and accurate real-robot operation.
- **Why this category:** This paper is best read as a deployment and execution optimization report. Its focus is on how to make a VLA run faster, smoother, and more accurately on real robots by combining calibration, planning and control, and learning-based speed selection. It is not centered on a new backbone, token compression method, or action generation paradigm, so 4.2 is the best fit. It is better understood as a system execution paper than as a paper about one sharply isolated efficiency mechanism.

## StreamingVLA

- **Title:** StreamingVLA: Streaming Vision-Language-Action Model with Action Flow Matching and Adaptive Early Observation
- **Short Name:** StreamingVLA
- **Link:** https://arxiv.org/pdf/2603.28565
- **Primary Category:** 4.2 Inference Efficiency Techniques
- **Secondary Category:** 3.1 Raw Action Generation
- **Core Idea:** A streaming VLA framework that asynchronously overlaps observation, action generation, and execution through action flow matching and adaptive early observation to reduce latency and execution halting.
- **Why this category:** The main problem here is asynchronous scheduling and streaming execution across the VLA pipeline. The paper overlaps observation, action generation, and execution instead of keeping them strictly serial, so its primary contribution belongs to inference and execution optimization. At the same time, the paper removes action chunking and adopts action flow matching, which is a meaningful change to the action generation setup and deserves a secondary 3.1 label.

## VLA-InfoEntropy

- **Title:** VLA-InfoEntropy: A Training-Free Vision-Attention Information Entropy Approach for Vision-Language-Action Models Inference Acceleration and Success
- **Short Name:** VLA-InfoEntropy
- **Link:** https://arxiv.org/pdf/2604.05323
- **Primary Category:** 2.1 Selective Feature Processing
- **Core Idea:** A training-free dynamic token selection framework that uses visual entropy, attention entropy, and timestep cues to shift VLA inference from global perception to locally informative regions for efficient computation.
- **Why this category:** The main problem here is visual information filtering and redundancy reduction. The method evaluates visual tokens with image entropy and attention entropy, then uses timestep cues to focus computation dynamically. In practice, this is a training-free visual token selection and compression method, so 2.1 is the best primary fit. The abstract mentions VLA-Cache-style KV reuse, but the paper's clearest and most central mechanism is still entropy-based visual selection rather than temporal reuse.

## SnapFlow

- **Title:** SnapFlow: One-Step Action Generation for Flow-Matching VLAs via Progressive Self-Distillation
- **Short Name:** SnapFlow
- **Link:** https://arxiv.org/pdf/2604.05656
- **Primary Category:** 3.1 Raw Action Generation
- **Core Idea:** A plug-and-play self-distillation method that compresses multi-step flow-matching denoising into single-step action generation for low-latency VLA inference.
- **Why this category:** The main problem is the action generation and sampling process in flow-matching VLAs. The paper reduces the original iterative denoising process from about ten steps to one-step action generation, directly cutting action decoding latency. Self-distillation is the training method used to get there, but the actual object being changed is the action generation paradigm itself. That makes 3.1 the best fit.

## A1

- **Title:** A1: A Fully Transparent Open-Source, Adaptive and Efficient Truncated Vision-Language-Action Model
- **Short Name:** A1
- **Link:** https://arxiv.org/pdf/2604.05672
- **Code:** https://github.com/ATeam-Research/A1
- **Primary Category:** 4.2 Inference Efficiency Techniques
- **Core Idea:** A budget-aware adaptive inference scheme that jointly accelerates the VLM backbone and flow-matching action head through action-consistency-based early termination and inter-layer truncated denoising warm-starts.
- **Why this category:** The main problem is joint acceleration of the full VLA inference pipeline. The abstract explicitly frames the goal as speeding up both the backbone and the flow-matching action head through budget-aware adaptive inference. Action-consistency-based early termination and inter-layer truncated flow matching are both runtime execution and decoding optimizations, so 4.2 is the best primary category. The paper is more about joint inference control than about a new static backbone design or a new action modeling paradigm.

## AAC

- **Title:** Adaptive Action Chunking at Inference-time for Vision-Language-Action Models
- **Short Name:** AAC
- **Link:** https://arxiv.org/pdf/2604.04161
- **Code:** https://github.com/junhyukso/SGAC
- **Primary Category:** 3.1 Raw Action Generation
- **Core Idea:** An inference-time strategy that adaptively selects action chunk size using action entropy to balance reactivity and consistency in VLA-based manipulation.
- **Why this category:** The main problem is action chunking. The method adaptively chooses chunk size at inference time according to action entropy, which directly changes the way actions are decoded and executed. Even though the method runs at inference time, the object being optimized is still the action chunking mechanism itself, so 3.1 is a better fit than a generic system-efficiency bucket.

## AnchorVLA

- **Title:** AnchorVLA: Anchored Diffusion for Efficient End-to-End Mobile Manipulation
- **Short Name:** AnchorVLA
- **Link:** https://arxiv.org/pdf/2604.01567
- **Code:** https://github.com/jason-lim26/AnchorVLA
- **Primary Category:** 3.1 Raw Action Generation
- **Core Idea:** An anchored diffusion action head that starts denoising near plausible trajectory anchors and uses a truncated diffusion schedule for low-latency multimodal action generation in mobile manipulation.
- **Why this category:** The main problem is diffusion-based action generation. The paper makes the action head start near a plausible solution manifold and then uses a truncated diffusion schedule to reduce iterative denoising cost while keeping multimodal action generation. That is best understood as an action generation change rather than a generic system scheduling trick. The residual correction module mainly stabilizes rollout and does not define the paper's main efficiency axis.

## Tri-Stage Token Pruning Framework

- **Title:** 2D or 3D: Who Governs Salience in VLA Models? — Tri-Stage Token Pruning Framework with Modality Salience Awareness
- **Short Name:** Tri-Stage Token Pruning Framework
- **Link:** https://arxiv.org/pdf/2604.09244
- **Primary Category:** 2.1 Selective Feature Processing
- **Core Idea:** A tri-stage token pruning framework for multi-visual-modal VLA models that adaptively selects 2D and 3D tokens according to modality salience discrepancies and dynamics across preprocessing, semantic synthesis, and action iteration.
- **Why this category:** The main problem here is visual token pruning in multi-modal VLA systems. The method builds a tri-stage pruning pipeline around 2D and 3D modality salience, including candidate selection, semantic grouping, and dynamic adjustment. This is still fundamentally a visual token selection and compression method, so 2.1 is the right primary category. The method does use temporal dynamics from action iteration, but those signals mainly adjust pruning strength rather than implement historical reuse or cache compression.
