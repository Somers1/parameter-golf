# Architecture Optimization Notes

This document explains the changes made to `train_gpt_ours.py` to make the shared-MLP architecture easier to optimize without giving up the main parameter-efficiency ideas.

## Problem

The architecture was already pursuing three useful ideas:

- shared MLP weights across layers
- per-layer gating and low-rank adaptation
- low-rank attention projections

The main optimization failure mode is not that any one of those ideas is obviously wrong. The problem is that the model was forcing too much coordination through shared weights:

- multiple layers push on the same MLP neuron bank
- each layer wants slightly different features
- the only layer-specific escape hatches were a multiplicative gate and LoRA-style adapters
- the shared MLP activations had no explicit per-layer channel calibration
- the optimizer treated shared matrices and block-local specialization too similarly

That creates gradient interference. In a short training run, the optimizer spends too much effort negotiating among layers instead of learning useful features quickly.

## What Changed

### 1. `Q_LATENT=0` and `KV_LATENT=0` now disable those bottlenecks

The attention module now treats latent size `0` as "use a full projection".

Why:

- it makes architecture ablations honest
- `Q_LATENT=512` on a `512`-dim model was not a real "no compression" setting, it was still a factored projection
- now you can test full `Q`, full `KV`, or compressed `KV` directly

### 2. Added per-layer channel modulation for the shared MLP

Each block now has:

- `mlp_pre_scale`
- `mlp_post_scale`

These are per-layer learned vectors over the shared MLP window.

Why:

- they give each layer a cheap private reweighting of the shared neuron slice
- that reduces pressure on the shared MLP weights to serve every layer identically
- this is a direct way to reduce shared-weight interference without paying for a full private MLP

Effectively, each layer gets a different view of the same shared channels.

### 3. Added optional normalization on the shared activation path

The block now supports a `shared_h_norm` toggle, enabled by `SHARED_H_NORM=1`.

This normalizes the shared MLP pre-activation path before the squared-ReLU nonlinearity.

Why:

- shared activations can drift to different scales as multiple layers compete over the same bank
- normalization makes the shared branch more scale-stable
- that usually makes gating and adaptation easier to train

### 4. Changed gate behavior from suppressive-only to residual modulation

The old form was approximately:

- `h = h * sigmoid(gate)`

The new form is:

- `h = h * (1 + tanh(gate))`

Why:

- the old gate could only shrink or preserve activations
- if the gate learned the wrong suppression early, gradients through the shared path could weaken
- the new gate is residual-like and centered around identity
- with zero-init on `gate_up`, the gate starts as an exact no-op

This makes the gate a smoother specialization mechanism rather than a hard bottleneck.

### 5. Added a tiny private residual MLP branch

New optional hyperparameter:

- `PRIVATE_MLP_RANK`

When enabled, each block gets a small private residual path:

- `dim -> private_rank -> dim`

with zero-init on the output projection so it starts as a no-op.

Why:

- LoRA is useful, but it still modifies the shared path indirectly
- a tiny private residual branch gives each layer an immediate private escape hatch
- that reduces the need for the shared bank to fit every layer's special cases
- this is often easier to optimize than forcing all specialization through shared weights

This is deliberately small so the model stays mostly shared.

### 6. Split the optimizer by architectural role, not just tensor shape

The optimizer is now separated into:

- token embedding
- shared MLP matrices
- block-local matrices
- scalar/control params
- per-layer modulation params
- optional LM head
- optional MTP heads

New learning-rate knobs:

- `SHARED_MATRIX_LR`
- `MODULATION_LR`

Why:

- shared MLP weights receive conflicting gradients from every layer
- those shared matrices should move more conservatively
- per-layer gates, adapters, channel scales, and private branches should adapt faster
- treating them the same wastes optimization capacity

The intended behavior is:

- shared weights: slow and stable
- block-local matrices: normal speed
- per-layer modulation: fast adaptation to reduce interference

## Why These Changes Should Help

The guiding idea is:

"keep the cheap shared computation, but make layer-specific behavior easier and more local"

That matters because the hardest part of the current architecture is not raw capacity. It is coordination cost.

The new design tries to reduce that coordination cost in four ways:

1. Per-layer channel scales make shared neurons easier to repurpose locally.
2. Residual gating avoids destructive suppression.
3. Private residual MLPs give each layer a small amount of independent capacity.
4. Optimizer groups let shared weights move cautiously while local modulation learns faster.

## New Environment Variables

- `SHARED_MATRIX_LR`
  - default `0.03`
  - lower LR for `shared_mlp.*`

- `MODULATION_LR`
  - default `0.08`
  - higher LR for gates, adapters, pre/post scales, and private MLP branch

- `PRIVATE_MLP_RANK`
  - default `0`
  - enables a tiny private residual MLP per layer

- `SHARED_H_NORM`
  - default `1`
  - enables normalization on the shared activation path

## Suggested Runs

### Conservative architecture run

Use this first if you want a cleaner, easier-to-optimize architecture:

```bash
Q_LATENT=0 \
KV_LATENT=128 \
GATE_RANK=32 \
ADAPT_RANK=8 \
PRIVATE_MLP_RANK=8 \
SHARED_H_NORM=1 \
NUM_LAYERS=11 \
ATTEND_EVERY=1 \
MLP_WINDOW=1536 \
MLP_OVERLAP=0.6 \
SHARED_MATRIX_LR=0.03 \
MATRIX_LR=0.04 \
MODULATION_LR=0.08 \
SCALAR_LR=0.04 \
TIED_EMBED_LR=0.05 \
MAX_WALLCLOCK_SECONDS=600 \
torchrun --standalone --nproc_per_node=1 train_gpt_ours.py
```

Why:

- full `Q` keeps token routing easy
- moderately compressed `KV` still saves parameters
- small adapters and private branch keep specialization cheap
- overlap stays high enough to share, but not so high that every layer becomes the same

### Stronger specialization run

Use this if the conservative run still looks too constrained:

```bash
Q_LATENT=0 \
KV_LATENT=128 \
GATE_RANK=48 \
ADAPT_RANK=16 \
PRIVATE_MLP_RANK=16 \
SHARED_H_NORM=1 \
NUM_LAYERS=11 \
ATTEND_EVERY=1 \
MLP_WINDOW=1536 \
MLP_OVERLAP=0.55 \
SHARED_MATRIX_LR=0.028 \
MATRIX_LR=0.04 \
MODULATION_LR=0.09 \
SCALAR_LR=0.04 \
TIED_EMBED_LR=0.05 \
MAX_WALLCLOCK_SECONDS=600 \
torchrun --standalone --nproc_per_node=1 train_gpt_ours.py
```

Why:

- more local escape capacity
- slightly lower overlap to reduce layer conflict
- slightly lower shared LR and slightly higher modulation LR

### Shared-MLP-focused control run

Use this to isolate whether the shared MLP design is carrying the improvement:

```bash
Q_LATENT=0 \
KV_LATENT=128 \
GATE_RANK=32 \
ADAPT_RANK=0 \
PRIVATE_MLP_RANK=0 \
SHARED_H_NORM=1 \
NUM_LAYERS=11 \
ATTEND_EVERY=1 \
MLP_WINDOW=1536 \
MLP_OVERLAP=0.6 \
SHARED_MATRIX_LR=0.03 \
MATRIX_LR=0.04 \
MODULATION_LR=0.08 \
SCALAR_LR=0.04 \
TIED_EMBED_LR=0.05 \
MAX_WALLCLOCK_SECONDS=600 \
torchrun --standalone --nproc_per_node=1 train_gpt_ours.py
```

Why:

- this isolates "shared MLP + channel modulation + residual gate" without extra low-rank specialization branches

## What To Watch In Results

When comparing runs, focus on:

- final `val_bpb`
- early training loss slope
- whether training becomes less noisy
- whether the gap to baseline closes without increasing model size too much

If the architecture is getting easier to optimize, you should usually see:

- a better early loss trajectory
- fewer signs of the model "stalling"
- less need for fragile compression tricks

## Expected Direction

The likely best-performing family is:

- full `Q`
- mild or moderate `KV` compression
- shared MLP
- residual gate
- per-layer channel scales
- small private residual branch
- optional small LoRA adapters

That keeps the parameter-efficient core idea while reducing the amount of optimization work needed to specialize each layer.
