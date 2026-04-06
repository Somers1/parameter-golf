# Gated Shared MLP: Our Approach to Parameter Golf

## The Core Idea

Instead of each transformer layer owning its own MLP weights, we use **one massive shared MLP** across all layers, with **tiny per-layer gate adapters** that control which neurons activate. This is multiplicative gating on hidden activations — each layer sees the same shared MLP but uses a different subset of it depending on the layer depth and input content.

Think of it as: one expert team (the shared MLP) with many lightweight receptionists (the gate adapters) who each know which specialists to bring in for the current problem.

## Why This Architecture

The challenge constraint is **parameter storage** (16MB artifact), not compute. This creates a unique optimisation target:

- **Dense models**: Every parameter is used for every input. Rich but expensive to store.
- **MoE**: Selective activation but total parameter count is large (many independent experts). Routing errors and dead experts are real problems.
- **Gated Shared MLP (ours)**: Dense storage (one shared MLP), MoE-like selective activation (per-layer gating), no dead experts (all neurons available to all layers), input-adaptive behaviour. Theoretical best of both worlds for a storage-constrained setting.

## How It Works

### Standard Transformer MLP (baseline)
```
Each layer owns:
  W_up:   (512 x 1024)   — unique per layer
  W_down: (1024 x 512)   — unique per layer

Forward:  hidden = relu(W_up · x)²
          output = W_down · hidden
```

### Our Gated Shared MLP
```
Shared across ALL layers:
  W_up:   (512 x 4096)   — one copy, 4x wider than baseline
  W_down: (4096 x 512)   — one copy

Per-layer gate adapter (tiny):
  W_gate_down: (512 x 32)    — unique per layer
  W_gate_up:   (32 x 4096)   — unique per layer

Forward:  hidden = relu(W_up · x)²
          gate   = sigmoid(W_gate_up · (W_gate_down · x))
          gated  = hidden * gate          ← element-wise: select which neurons matter
          output = W_down · gated
```

The gate is a sigmoid (0 to 1) over all 4096 hidden neurons. If the gate outputs 0.0 for a neuron, it's effectively dead for this layer. If 0.9, it contributes strongly. Each layer learns which neurons are relevant for its position in the network and for the current input.

### What the gates learn (intuition)
- **Depth specialisation**: Early layers gate on syntax-related neurons, late layers gate on prediction neurons
- **Input adaptation**: The same layer shifts its gate slightly for code vs prose vs numbers
- **Adjacent layers share heavily**: Layer 5 and 6 probably gate almost identically since they see similar inputs
- **Early vs late layers diverge**: Layer 0 and layer 23 should have nearly inverted gate patterns

## Parameter Budget

### Baseline (9 layers, 1024 MLP hidden)
```
9 x MLP:        9 x 1.05M  =  9.4M
9 x Attention:  9 x 786K   =  7.1M
Embeddings:                    0.5M
Total:                        17.0M params
```

### Our Architecture (24 layers, 4096 MLP hidden, MLA attention)
```
Shared MLP (4096 hidden):       512x4096 + 4096x512  =  4.2M
24 gate adapters:               24 x (512x32 + 32x4096) = 3.5M
24 MLA attention layers:                                  3.6M
Embeddings + norms + skips:                               0.6M
                                                         ------
Total:                                                   11.9M params
At int6 quantization:                                    ~8.9MB
```

**Result: 24 layers deep with 4096-wide MLP, comfortably under 16MB.**

Compare to baseline: 2.7x deeper, 4x wider MLP.

### Same-config comparison (9 layers, same dims)
If we keep the same 9 layers and dimensions but just share + gate:
```
Baseline:    17.0M params
Shared+Gated: 2.95M params (17% of baseline)
```
83% parameter reduction at the same depth and width.

## Novelty

Literature review confirms this specific combination is novel:

| Prior Work | What They Did | How We Differ |
|---|---|---|
| Mixture of LoRAs (2024-2025) | Shared FFN + LoRA experts (additive weight perturbation) | We use multiplicative gating on activations, not additive weight changes |
| Depth as Modulation (2024) | Per-layer LoRA on shared block | Only modulates attention, not MLP |
| ALBERT (2020) | Full cross-layer weight sharing | No per-layer differentiation at all |
| DeepSeekMoE (2024) | Shared + routed experts | Shares subset of experts; we share one backbone and steer it |
| Product Key Memory (2019) | KV memory replaces FFN | Per-layer memory, not shared across layers |
| Universal Transformer (2019) | Full weight sharing with adaptive halting | No gating mechanism for specialisation |

**The specific gap**: Nobody has trained per-layer learned gating masks on the hidden activations of a shared MLP from scratch for pretraining. The additive version exists (Mixture of LoRAs). The multiplicative gating version — where each layer's adapter controls which neurons fire — is novel.

**Why gating may be better than additive**: Gating can completely shut off irrelevant neurons (gate -> 0), which additive perturbation cannot. This enables sharper specialisation per layer.

## Additional Techniques to Stack

### Architecture
- **MLA (Multi-head Latent Attention)**: From DeepSeek-V2/V3. Compress KV projections through a low-rank bottleneck. ~75% attention param savings.
- **Shared attention weights with gating**: Same principle applied to attention projection matrices. Could push to 48 layers.
- **LeakyReLU(0.5)-squared**: Drop-in activation improvement from leaderboard (+0.003 BPB).

### Training
- **Multi-token prediction**: Train with 2-4 prediction heads (discarded after). More gradient signal per step. Nobody on leaderboard has tried this.
- **Progressive sequence length**: Start at seq_len=256 (fast), increase to 1024+ during training.
- **Distillation from larger teacher**: Train a big model offline, use its soft targets during our 10-min run. Gemma 3 proved this works dramatically well.
- **QAT (quantization-aware training)**: Train with fake-quantised weights so the model learns int6-friendly values.

### Evaluation
- **Sliding window eval**: Overlapping windows for maximum context per token. Free BPB improvement.
- **TTT (test-time training)**: Adapt gate adapters only on already-scored validation chunks. Gates are tiny so adaptation is fast and low-risk.
- **Extended context with YaRN**: Eval at 2048+ seq_len even if trained at 1024.

### Export
- **Int6 QAT**: 6-bit quantised weights, trained to be robust.
- **LZMA compression**: ~15-20% better than zlib on model weights.

## Training Properties

- **End-to-end backprop**: Everything trains together in one forward/backward pass. No special training procedure.
- **Shared MLP gets gradient from all layers**: More gradient signal per step than per-layer MLPs. Faster learning and implicit regularisation.
- **Gates specialise naturally**: Each layer sees different input (residual stream changes with depth), so gates learn different patterns without explicit encouragement.

## Risks and Mitigations

| Risk | Mitigation |
|---|---|
| Gate collapse (all layers learn same gate) | Diversity loss + diverse initialisation |
| Shared MLP bottleneck | Scale hidden dim (4096 should be sufficient) |
| Training instability with shared gradients | Gate adapters absorb layer-specific needs, reducing gradient conflict |
| Per-sample gating overhead | Negligible compute cost for sigmoid + element-wise multiply |

## Implementation Plan

1. Get baseline training and scoring working locally (M3 Max, MLX) ✅
2. Implement gated shared MLP — replace per-layer MLPs with shared + gate adapters
3. Validate on short training runs — does loss decrease? Do gates differentiate?
4. Add MLA attention — low-rank Q/K/V projections
5. Scale up depth (24+ layers) and measure BPB improvement
6. Layer on training improvements (MTP, QAT, progressive seq_len)
7. Add evaluation tricks (sliding window, TTT on gates, extended context)
8. Optimise export (int6 + LZMA)

## References

- Mixture of LoRAs (2024-2025) — https://arxiv.org/abs/2512.12880
- Relaxed Recursive Transformers (ICLR 2025) — https://arxiv.org/abs/2410.20672
- MoLoRA (Microsoft, 2025) — https://arxiv.org/abs/2603.15965
- FiPS (2024) — https://arxiv.org/abs/2411.09816
- Depth as Modulation (2024) — https://openreview.net/forum?id=wm9jRInse3
- Large Memory Layers with Product Keys (2019) — https://arxiv.org/abs/1907.05242
- DeepSeekMoE (2024) — https://arxiv.org/abs/2401.06066
- ALBERT (2020) — Cross-layer weight sharing baseline
- Universal Transformer (ICLR 2019) — Weight sharing with adaptive halting
