# SAE Training Optimization: Diagnosis, Root Causes & Fixes

**Date:** 2026-02-12 to 2026-02-13
**Commits:** `3a3d875`, `a2637f7`, `fa754b7`, `074ab46`
**Deployed to:** k8s-mistudio.hitsai.local

---

## 1. Problem Statement

JumpReLU SAE training (`train_3d`) on LFM2.5-1.2B-Instruct produced catastrophically bad results:

| Metric | Value | Expected |
|--------|-------|----------|
| Loss | 1.2876 | < 0.1 |
| L0 Sparsity | ~1081 features (6.6%) | ~819 features (5%) |
| Dead Neurons | 15,303 / 16,384 (93.4%) | < 1,640 (< 10%) |
| FVU | Not reported | < 0.10 |

Training configuration used:
- Model: LFM2.5-1.2B-Instruct, hidden_dim=2048
- SAE: JumpReLU, latent_dim=16384, layers L4/L9/L14, residual hooks
- sparsity_coeff: 0.0006, learning_rate: 0.0001, batch_size: 512
- weight_decay: 0.01, warmup_steps: 10000, total_steps: 100000
- No sparsity warmup (feature didn't exist)

---

## 2. Root Cause Analysis

Five distinct issues were identified, ordered by impact:

### 2.1 CRITICAL: JumpReLU L0 Loss Was Non-Differentiable (Bug)

**Location:** `backend/src/ml/sparse_autoencoder.py`, JumpReLUSAE.forward(), lines 890-899

**The bug:**
```python
# OLD CODE (non-differentiable)
l0_per_sample = (f != 0).float().sum(dim=-1)  # Heaviside step function
l0_mean = l0_per_sample.mean()
loss_l0 = self.sparsity_coeff * l0_mean
loss_total = loss_reconstruction + loss_l0  # loss_l0 has ZERO gradient
```

**Why it's a bug:** The expression `(f != 0).float()` is a Heaviside step function, which has zero derivative everywhere (except at exactly 0, where it's undefined). When `loss_total.backward()` is called, PyTorch's autograd computes:

```
d(loss_l0)/d(f) = sparsity_coeff * d/d(f)[(f != 0).float().sum().mean()] = 0
```

This means:
1. **The sparsity_coeff parameter had zero effect on training.** Changing it from 0.0006 to any other value would produce identical results.
2. **No gradient signal pushed thresholds to control sparsity.** The JumpReLU thresholds (θ) only received gradients from reconstruction loss via the STE in `JumpReLUFunction.backward()`.
3. **Sparsity was determined entirely by reconstruction dynamics.** The model found it only needed ~0.4% of features for reconstruction and let the rest die.

**Note:** The `JumpReLUFunction.backward()` STE does provide gradients to thresholds from the *reconstruction* loss. The STE computes `grad_threshold = -(grad_output * z * kernel).sum(dim=0)`, but `grad_output` only contains reconstruction gradients (since L0 contributes zero gradient to `f`). The missing term is the direct L0 penalty gradient on thresholds: `-sparsity_coeff * kernel.sum(dim=0)`.

**Evidence:** After fixing only the other 4 issues (sparsity warmup, defaults, etc.) and retraining as `train_b7`:
- Loss improved dramatically: 1.2876 → 0.0353 (other fixes helped reconstruction)
- But dead neurons got WORSE: 93.4% → 98.0% (16,054 / 16,384)
- L0 collapsed to 0.4% (59 features) instead of target 5%
- The sparsity_coeff was confirmed to have no effect

### 2.2 CRITICAL: L0 Penalty Scaled by Raw Count, Not Fraction

**The issue:** Even if L0 were differentiable, the penalty was computed as:
```python
loss_l0 = sparsity_coeff * l0_count  # l0_count ∈ [0, 16384]
```

At 5% target sparsity with 16,384 features:
- l0_count = 0.05 × 16384 = 819
- loss_l0 = 0.0006 × 819 = **0.49**
- reconstruction_loss ≈ **0.035**
- **Ratio: L0 penalty is 14× larger than reconstruction loss**

This means the model is massively penalized for activating features. It's equivalent to setting a very aggressive sparsity target. The equilibrium point is where `sparsity_coeff × l0_count ≈ reconstruction_loss`:
- 0.0006 × l0_count ≈ 0.035 → l0_count ≈ 58 → L0 ≈ 0.35%

This exactly matches the observed 0.4% L0 in `train_b7`.

**Additional problem:** The raw count makes `sparsity_coeff` dependent on `latent_dim`. A value of 0.0006 means very different things for 4K vs 64K features.

### 2.3 CRITICAL: Sigmoid L0 Surrogate Creates Phantom L0 from Dead Features

**Location:** `backend/src/ml/sparse_autoencoder.py`, JumpReLUSAE.forward()

**Discovered:** 2026-02-13, after deploying the sigmoid surrogate fix (commit `fa754b7`). Training `train_a3ace15a` still showed 99.85% dead neurons.

**The bug:** The sigmoid surrogate `σ((z - θ) / ε)` assigns significant activation probability to EVERY feature, including dead ones:

```
For dead features: z ≈ 0, threshold ≈ 0.001, bandwidth = 0.001
sigmoid((0 - 0.001) / 0.001) = sigmoid(-1) = 0.269
```

With 16,384 features and 16,359 dead:
- Dead features contribute: 16359 × 0.269 / 16384 ≈ **0.268**
- Active features contribute: 25 × ~1.0 / 16384 ≈ 0.0015
- Total l0_surrogate_fraction ≈ **0.270** (vs real L0 = 0.0015)
- loss_l0 = 0.4 × 0.270 = **0.108**
- loss_reconstruction = **0.0007**
- **L0 penalty is 154× larger than reconstruction loss!**

This drives the model to suppress ALL encoder outputs to minimize the phantom L0, causing catastrophic sparsity collapse.

**Training `train_a3ace15a` results (with sigmoid surrogate):**

| Step | Loss | L0 | Dead | FVU | Notes |
|------|------|----|------|-----|-------|
| 6900 | 0.0026 | 0.63% | 16,280 | 0.002 | Initial state |
| 10100 | 0.0051 | 1.26% | 14,246 | 0.001 | After resample at 10K |
| 10300 | 0.0051 | 1.26% | 16,170 | 0.001 | Resampled neurons die immediately |
| 15000 | 0.0021 | 0.50% | 16,297 | 0.002 | Pre-resample |
| 15100 | 0.0006 | 0.15% | 14,252 | <0.001 | After resample at 15K |
| 15300 | 0.0006 | 0.15% | 16,355 | <0.001 | **COLLAPSE** — frozen state |
| 19300 | 0.0007 | 0.15% | 16,359 | 0.001 | Nothing changes |

The model found a degenerate equilibrium with only 25 active features. Resampling briefly revived neurons but they died within 200 steps because the phantom L0 penalty killed them.

**Fix (commit `074ab46`):** Replace sigmoid surrogate with `StraightThroughL0` custom autograd function:
- **Forward:** Exact Heaviside `(z > threshold).float()` — dead features contribute exactly 0
- **Backward:** Gaussian kernel STE `K(z-θ) = (1/ε√2π) exp(-(z-θ)²/(2ε²))` — smooth gradients near threshold boundary

The Gaussian kernel ensures:
- Features near threshold (z ≈ θ) get strong gradient to push them active or inactive
- Features far from threshold (z << θ) get near-zero gradient — correctly no pressure
- Dead features with z < 0 get zero L0 gradient (unlike sigmoid which gave 0.269)

### 2.4 HIGH: No Sparsity Warmup

**The issue:** The L1/L0 sparsity penalty was at full strength from step 0. During early training, the model hasn't yet learned meaningful feature representations. Applying full sparsity pressure immediately:
1. Kills features before they can form meaningful representations
2. Creates a "dead neuron" cascade where early dead features never recover
3. The surviving features become polysemantic (encoding multiple concepts)

**Reference:** SAELens and Gemma Scope both use sparsity warmup. The standard approach is to linearly ramp the sparsity penalty from 0 to full over 5,000-10,000 steps.

### 2.4 MEDIUM: Suboptimal Default Hyperparameters

| Parameter | Old Default | Problem | New Default | Rationale |
|-----------|------------|---------|-------------|-----------|
| learning_rate | 0.0001 | 3× too low for JumpReLU; thresholds adapt slowly | 0.0003 | Gemma Scope / SAELens standard |
| weight_decay | 0.01 | Fights decoder norm constraint; degrades reconstruction | 0.0 | SAEs should not use weight decay |
| batch_size | 512 | Noisy gradients; unreliable dead neuron detection | 2048 | More stable; better dead neuron EMA |
| total_steps | 100000 | Correct for batch=512; too many for batch=2048 | 50000 | Same token budget with 4× batch |
| warmup_steps | 10000 | Too long for LR warmup alone | 2000 | Shorter LR warmup; sparsity warmup handles the rest |

### 2.5 LOW: Backend l1_alpha Recommendation Formula Was Wrong

**Location:** `backend/src/services/training_validator.py`, `calculate_recommended_l1_alpha()`

**Old formula:** `5.0 / sqrt(latent_dim / 8192)` → returns 3.5 for 16K features
**Actual need:** `5e-4 * sqrt(16384 / latent_dim)` → returns 5e-4 for 16K features

The old formula was calibrated for a `.mean()` L1 formulation, but the codebase uses `.sum(dim=-1).mean()`. The `.sum()` accumulates across all latent features, producing values ~7,000× larger. This caused misleading backend log warnings.

---

## 3. Fixes Implemented

### Fix 1: Differentiable L0 with STE (commits `fa754b7`, `074ab46`)

**File:** `backend/src/ml/sparse_autoencoder.py`

**Iteration 1 (commit `fa754b7` — FAILED):** Replaced non-differentiable Heaviside with sigmoid surrogate `σ((z - θ) / ε)`. This provided gradients but caused phantom L0 from dead features (see Section 2.3). Training `train_a3ace15a` still collapsed to 99.85% dead neurons.

**Iteration 2 (commit `074ab46` — CORRECT):** Replaced sigmoid surrogate with custom `StraightThroughL0` autograd function:

```python
class StraightThroughL0(Function):
    """Heaviside forward + Gaussian kernel STE backward for L0."""

    @staticmethod
    def forward(ctx, z, threshold, bandwidth=0.001):
        # Exact binary: 1 where z > threshold, 0 otherwise
        # Dead features contribute exactly 0 (no phantom L0)
        indicators = (z > threshold).to(z.dtype)
        ctx.save_for_backward(z, threshold)
        ctx.bandwidth = bandwidth
        return indicators

    @staticmethod
    def backward(ctx, grad_output):
        z, threshold = ctx.saved_tensors
        bandwidth = ctx.bandwidth
        delta = z - threshold
        # Gaussian kernel: smooth gradient near threshold, zero far away
        kernel = torch.exp(-0.5 * (delta / bandwidth) ** 2) / (bandwidth * sqrt(2π))
        grad_z = grad_output * kernel
        grad_threshold = (-grad_output * kernel).sum(dim=0)
        return grad_z, grad_threshold, None
```

Usage in `JumpReLUSAE.forward()`:
```python
# Get pre-activations for L0 gradient
f, z = self.encode(x_normalized, return_pre_activations=True)

# Actual L0 for metrics (non-differentiable)
with torch.no_grad():
    l0_sparsity = (f != 0).float().mean()

# Differentiable L0 via STE (exact forward, smooth backward)
l0_differentiable = StraightThroughL0.apply(z, threshold, bandwidth)
l0_diff_fraction = l0_differentiable.mean(dim=-1).mean()
loss_l0 = self.sparsity_coeff * l0_diff_fraction
```

**Why STE over sigmoid?** The sigmoid `σ(-1) = 0.269` for dead features, inflating L0 by 154×. The STE gives `H(z-θ) = 0` in forward (no phantom), while the Gaussian kernel in backward provides smooth gradients only near the threshold boundary where they're needed.

**Key insight:** The sigmoid relaxation is a standard technique in continuous optimization, but it fails for SAEs because the vast majority of features (>99%) are inactive. The phantom 0.269 contribution per dead feature dominates the loss. STE avoids this by keeping the forward exact.

### Fix 2: Normalized L0 (commit `fa754b7`)

Changed from `.sum(dim=-1)` (raw count) to `.mean(dim=-1)` (fraction) in the L0 surrogate:

```python
# OLD: l0 ∈ [0, 16384] — sensitive to latent_dim
l0_surrogate.sum(dim=-1).mean()

# NEW: l0 ∈ [0, 1] — independent of latent_dim
l0_surrogate.mean(dim=-1).mean()
```

**Rationale:** With raw count, `sparsity_coeff=0.0006` means very different things:
- latent_dim=4096: penalty at 5% = 0.0006 × 205 = 0.12
- latent_dim=16384: penalty at 5% = 0.0006 × 819 = 0.49
- latent_dim=65536: penalty at 5% = 0.0006 × 3277 = 1.97

With fraction, `sparsity_coeff=0.4` gives consistent penalty regardless of width:
- Any latent_dim at 5%: penalty = 0.4 × 0.05 = 0.02

### Fix 3: Recalibrated sparsity_coeff Default (commit `fa754b7`)

**Old default:** 0.0006 (for raw count L0, but was inert anyway)
**New default:** 0.4 (for fraction-based L0)

**Calibration:** At target 5% sparsity, `loss_l0 = 0.4 × 0.05 = 0.02`. Typical reconstruction loss is 0.01-0.05, so the L0 penalty is ~20-100% of reconstruction loss. This gives the model a meaningful incentive to be sparse without overwhelming reconstruction.

**Updated ranges:**
- Schema: `le=10.0` (was `le=1.0`)
- Validator: warns if > 5.0 or < 0.01 (was > 0.01 or < 1e-5)
- UI input: min=0.01, max=10.0, step=0.1 (was min=1e-5, max=0.1, step=1e-4)

### Fix 4: Sparsity Warmup (commit `3a3d875`)

**Files:**
- Schema: Added `sparsity_warmup_steps` field (default 5000)
- Training loop: Linear ramp from 0 to full over N steps
- Frontend: New UI field in Advanced Settings

```python
# Before training loop
sparsity_warmup_steps = hp.get('sparsity_warmup_steps', 0)
base_sparsity_coeffs = {}
for sae_key, model in models.items():
    if architecture_type == 'jumprelu':
        base_sparsity_coeffs[sae_key] = model.sparsity_coeff
    else:
        base_sparsity_coeffs[sae_key] = model.l1_alpha

# Inside training loop (before forward pass)
if sparsity_warmup_steps > 0:
    sparsity_scale = min(1.0, step / sparsity_warmup_steps)
    for sae_key, model_ref in models.items():
        base_coeff = base_sparsity_coeffs[sae_key]
        if architecture_type == 'jumprelu':
            model_ref.sparsity_coeff = base_coeff * sparsity_scale
        else:
            model_ref.l1_alpha = base_coeff * sparsity_scale
```

**Why this works:** `sparsity_coeff` is a regular Python attribute (not nn.Parameter), so direct assignment is read immediately by the next `forward()` call.

### Fix 5: EMA Dead Neuron Detection (commit `3a3d875`)

**File:** `backend/src/workers/training_tasks.py`

Replaced per-batch detection `(z == 0).all(dim=0)` with exponential moving average:

```python
# Before training loop
feature_activation_ema = {}
ema_window_tokens = 50000

# Inside loop after forward pass
with torch.no_grad():
    fired = (z > 0).any(dim=0).float()
    if sae_key not in feature_activation_ema:
        feature_activation_ema[sae_key] = torch.zeros(hp['latent_dim'], device=z.device)
    ema_decay = max(0.0, 1.0 - batch_size / ema_window_tokens)
    feature_activation_ema[sae_key] = feature_activation_ema[sae_key] * ema_decay + fired
    layer_dead_neurons[sae_key] = (feature_activation_ema[sae_key] < 0.01).sum().item()
```

**Why EMA is better:** With batch_size=512 and 16K features at 5% activation, a healthy neuron fires ~26 times per batch. But a neuron with 0.5% activation rate fires ~2.6 times per batch — it appears "dead" in many individual batches but isn't truly dead. The EMA tracks activation frequency over a ~50K token window, giving reliable dead neuron counts.

### Fix 6: Updated Defaults (commit `3a3d875`)

| Parameter | Old | New | Rationale |
|-----------|-----|-----|-----------|
| learning_rate | 0.0001 | 0.0003 | Gemma Scope / SAELens standard |
| weight_decay | 0.01 | 0.0 | SAEs should not use weight decay |
| batch_size | 512 | 2048 | More stable gradients |
| total_steps | 100000 | 50000 | Same token budget (4× batch) |
| warmup_steps | 10000 | 2000 | Shorter LR warmup |
| sparsity_warmup_steps | N/A | 5000 | New feature |
| l1_alpha | 0.045 | 0.0005 | Corrected for .sum().mean() L1 |

### Fix 7: Backend l1_alpha Formula (commit `3a3d875`)

```python
# OLD: calibrated for .mean() L1 formulation
def calculate_recommended_l1_alpha(latent_dim):
    return 5.0 / math.sqrt(latent_dim / 8192)  # Returns 3.5 for 16K

# NEW: calibrated for .sum(dim=-1).mean() L1 formulation
def calculate_recommended_l1_alpha(latent_dim):
    BASE_LATENT_DIM = 16384
    BASE_L1_ALPHA = 5e-4  # SAELens baseline for 16K features
    return BASE_L1_ALPHA * math.sqrt(BASE_LATENT_DIM / latent_dim)  # Returns 5e-4 for 16K
```

### Fix 8: JumpReLU Params in HP Modal (commit `3a3d875`)

**File:** `frontend/src/components/training/TrainingCard.tsx`

The hyperparameters modal now shows JumpReLU-specific fields:
- Sparsity Coeff (L0) — with emerald highlight border for JumpReLU
- L1 Alpha — dimmed with "(unused)" label for JumpReLU
- Initial Threshold, Bandwidth, Normalize Decoder
- Sparsity Warmup Steps

### Fix 9: Weight Decay Label Bug (commit `a2637f7`)

**File:** `frontend/src/components/training/TrainingCard.tsx`

React renders literal `"0"` when using `{0 && <JSX>}` pattern. Changed to `!= null` checks:
```jsx
// OLD: renders "0" without label when weight_decay is 0
{training.hyperparameters.weight_decay && (<div>...</div>)}

// NEW: correctly shows labeled "0" value
{training.hyperparameters.weight_decay != null && (<div>...</div>)}
```

---

## 4. Files Modified

| File | Changes |
|------|---------|
| `backend/src/ml/sparse_autoencoder.py` | Differentiable L0 surrogate, normalized L0, sparsity_coeff default 0.4, encode() returns pre-activations, create_sae() no longer falls back to l1_alpha |
| `backend/src/schemas/training.py` | Added sparsity_warmup_steps field, updated sparsity_coeff bounds (le=10.0) |
| `backend/src/services/training_validator.py` | Fixed l1_alpha formula (5e-4 baseline), JumpReLU sparsity_coeff validation (0.1-2.0 range), sparsity_warmup_steps in quality checks |
| `backend/src/workers/training_tasks.py` | Sparsity warmup implementation, EMA dead neuron detection, updated resampling timing |
| `backend/tests/unit/test_training_validator.py` | Updated formula tests for new l1_alpha values, adjusted warning threshold multiplier |
| `frontend/src/stores/trainingsStore.ts` | Updated defaults: LR, batch, steps, warmup, weight_decay, sparsity_coeff |
| `frontend/src/types/training.ts` | Added sparsity_warmup_steps type |
| `frontend/src/components/panels/TrainingPanel.tsx` | Sparsity warmup UI field, updated sparsity_coeff input (0.01-10.0), updated defaults |
| `frontend/src/components/training/TrainingCard.tsx` | JumpReLU params in HP modal, weight_decay label fix, sparsity warmup display |
| `frontend/src/config/hyperparameterDocs.ts` | Updated sparsity_coeff docs, added sparsity_warmup_steps docs |
| `frontend/src/utils/hyperparameterOptimization.ts` | Updated getRecommendedHyperparameters() |

---

## 5. Recommended Settings for JumpReLU Training

For LFM2.5-1.2B-Instruct (hidden_dim=2048, latent_dim=16384):

| Parameter | Value | Notes |
|-----------|-------|-------|
| architecture_type | jumprelu | |
| sparsity_coeff | **0.4** | Applied to normalized L0 fraction. Increase for sparser, decrease for denser |
| learning_rate | **0.0003** | Gemma Scope / SAELens standard |
| weight_decay | **0.0** | Must be 0 for SAEs |
| batch_size | **2048** | Stable gradients |
| total_steps | **50000** | Same token budget as 100K × batch=512 |
| warmup_steps | **2000** | LR warmup |
| sparsity_warmup_steps | **5000** | Ramp L0 penalty 0 → full |
| initial_threshold | 0.001 | Per-feature JumpReLU threshold init |
| bandwidth | 0.001 | KDE bandwidth for STE gradients |
| normalize_decoder | true | Required for JumpReLU |
| resample_dead_neurons | true | |
| resample_interval | 5000 | |
| grad_clip_norm | 1.0 | |

**Expected outcomes:** L0 ~2-5%, dead neurons <10%, FVU <0.10

---

## 6. Technical Deep Dive: JumpReLU Gradient Flow

### Architecture

```
x → z = W_enc @ x + b_enc     (pre-activations)
z → f = z ⊙ H(z - θ)          (JumpReLU: gated activation)
f → x_hat = W_dec @ f + b_dec  (reconstruction)
```

Where θ are learnable per-feature thresholds stored as `log_threshold` (ensuring positivity via exp).

### Loss Computation (After Fix)

```
L = L_recon + λ × L0_surrogate

L_recon = MSE(x_hat, x)                           # Differentiable through f → x_hat
L0_surrogate = mean(σ((z - θ) / ε))               # Differentiable sigmoid approximation
```

### Gradient Paths to Threshold θ

1. **From reconstruction (via JumpReLU STE):**
   ```
   L_recon → x_hat → f → JumpReLUFunction.backward() → θ
   grad_θ_recon = -(grad_output × z × K(z,θ)).sum(batch)
   ```
   This pushes θ DOWN to activate more features for better reconstruction.

2. **From L0 surrogate (via sigmoid backprop):**
   ```
   L0_surrogate → σ((z-θ)/ε) → θ
   grad_θ_l0 = -λ × σ'((z-θ)/ε) / ε
   ```
   This pushes θ UP to deactivate features for sparser representation.

The balance between these two gradients determines the sparsity level.

### Gradient Paths to Encoder Weights (W_enc, b_enc)

1. **From reconstruction:** `L_recon → x_hat → f → z → W_enc`
2. **From L0 surrogate:** `L0_surrogate → σ((z-θ)/ε) → z → W_enc`

Path 2 was previously missing (L0 was non-differentiable). It teaches the encoder to produce pre-activations that are clearly above or below thresholds, rather than lingering near the boundary.

### Why Sigmoid Not Gaussian?

The JumpReLUFunction backward uses a Gaussian kernel for the STE:
```
K(z,θ) = (1/(ε√2π)) × exp(-(z-θ)²/(2ε²))
```

Our L0 surrogate uses sigmoid:
```
σ((z-θ)/ε) with derivative σ'((z-θ)/ε)/ε
```

Both are bell-shaped curves centered at z=θ with effective range ~3ε. The practical difference is negligible. Sigmoid was chosen because:
1. It's directly available in PyTorch with efficient autograd
2. It naturally outputs values in [0, 1] (like the Heaviside it approximates)
3. No need to modify the JumpReLUFunction custom backward

---

## 7. Training Results Timeline

| Training | Date | Config Issues | Results |
|----------|------|---------------|---------|
| train_3d | 2026-02-12 | No warmup, weight_decay=0.01, LR=1e-4, L0 non-diff | Loss: 1.29, L0: 6.6%, Dead: 93.4% |
| train_b7 | 2026-02-13 | L0 still non-diff (other issues fixed) | Loss: 0.035, L0: 0.4%, Dead: 98.0% |
| train_a3ace15a | 2026-02-13 | Sigmoid L0 phantom (Section 2.3) | Loss: 0.0007, L0: 0.15%, Dead: 99.85% |
| (next) | pending | STE L0 fix (commit `074ab46`) | Expected: L0 ~5%, Dead <10% |

**Key insight from train_b7:** The improved defaults (LR, weight_decay, warmup) dramatically improved reconstruction loss (1.29 → 0.035), confirming those fixes were correct. But dead neurons actually increased because the non-differentiable L0 meant the model converged to a minimal-feature solution purely through reconstruction dynamics.

**Key insight from train_a3ace15a:** The sigmoid surrogate was WORSE than non-differentiable L0 because it created phantom L0 pressure from dead features, overwhelming reconstruction. The model collapsed to using only 25 features (0.15% L0) with 99.85% dead neurons. Resampled neurons died within 200 steps. This demonstrates that a "differentiable relaxation" is not automatically better — the STE approach of keeping forward exact while providing smooth backward gradients is the correct solution for SAEs where most features are inactive.

---

## 8. Testing

- 25/25 training_validator tests pass (formula tests updated for new values)
- 46/46 sparse_autoencoder tests pass (forward/backward, JumpReLU, create_sae)
- 658/658 unit tests pass (167 errors are pre-existing DB setup issues)
- Frontend type-check: clean
- Frontend build: clean (5.23s)

---

## 9. Open Questions / Future Work

1. **Sparsity_coeff tuning per model:** The default 0.4 is calibrated for typical reconstruction loss scales (~0.01-0.05). Models with very different activation norms may need adjustment. Consider auto-calibration based on initial reconstruction loss.

2. **Bandwidth sensitivity:** The STE and JumpReLU both use `bandwidth=0.001`. This controls the gradient "window" around the threshold. The Gaussian kernel `K(z-θ)` has width ~3ε ≈ 0.003 around the threshold. If model pre-activations have very different scales, this may need adjustment.

3. **Interaction between sparsity warmup and STE L0:** With the STE now providing real L0 gradients, the sparsity warmup is even more important — without it, the L0 penalty would immediately push thresholds up before the encoder has learned meaningful representations.

4. **Standard SAE L1 formulation:** The standard SAE uses `.sum(dim=-1).mean()` for L1, which also scales with latent_dim. Consider normalizing this too (`.mean(dim=-1).mean()`) for consistency. This would require recalibrating l1_alpha similarly to what we did for sparsity_coeff.
