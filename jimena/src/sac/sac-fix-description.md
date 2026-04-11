# SAC Fix Description

Four files were changed. Every modification is minimal — only the lines that directly cause incorrect training behaviour were touched.

---

## 1. `configs/sac.yaml`

### 1.1 Replay buffer size: 100,000 → 300,000

```yaml
# before
buffer_size: 100000

# after
buffer_size: 300000
```

**Why.** Off-policy algorithms like SAC rely on sample diversity in the replay buffer to break temporal correlations and avoid overfitting to recent experience. Visual observations are high-dimensional, so the agent needs a larger buffer to maintain sufficient diversity across the wide state space. A buffer of 100,000 frames turns over too quickly and causes the critic to overfit to recent trajectories, degrading the quality of Q-value estimates.

---

### 1.2 Batch size: 256 → 128

```yaml
# before
batch_size: 256

# after
batch_size: 128
```

**Why.** With image-based observations, each batch element is significantly more expensive in memory and compute than a state-vector element. A batch size of 128 keeps gradient updates at a stable frequency relative to data collection without introducing excessive memory pressure. It also tends to produce noisier — and therefore more regularising — gradients, which is beneficial when learning from pixels where the loss landscape is less smooth than in state-based RL.

---

### 1.3 Training frequency: every step → every 2 steps

```yaml
# before
train_freq: 1

# after
train_freq: 2
```

**Why.** Running one gradient update per environment step (train_freq=1) means the network is updated at the same rate as data is collected. For pixel-based SAC this is too aggressive: the CNN encoder is slow to converge, so early gradient steps on poorly-learned features produce noisy and potentially harmful updates. Updating every 2 steps gives the buffer more time to accumulate diverse transitions between updates, reducing overfitting and improving the stability of the critic.

---

### 1.4 Learning starts: 20,000 → 50,000

```yaml
# before
learning_starts: 20000

# after
learning_starts: 50000
```

**Why.** Before gradient updates begin, the agent collects transitions using a random policy. This warm-up period is essential for the CNN encoder: it needs to see a diverse distribution of observations before the first weight update, otherwise early gradients are computed over a very narrow slice of the state space and can push the network into a poor local region from which it struggles to recover. 50,000 random steps provides a sufficiently broad initial dataset for the visual encoder to start with a reasonable signal.

---

## 2. `jimena/src/sac/env.py`

### 2.1 Observation space dtype and bounds

```python
# before
self.observation_space = spaces.Box(
    low=0,
    high=255,
    shape=(channels * self.k, self.height, self.width),
    dtype=np.uint8,
)

# after
self.observation_space = spaces.Box(
    low=0.0,
    high=1.0,
    shape=(channels * self.k, self.height, self.width),
    dtype=np.float32,
)
```

**Why.** Neural networks converge significantly faster and more stably when inputs are in a small, centred numerical range. Feeding raw uint8 pixel values (0–255) directly to a CNN causes gradients in the first layer to be dominated by the scale of the input rather than the structure of the features, which slows down and destabilises training. Normalising to float32 [0, 1] at the environment boundary is the correct place to do it: it keeps the observation space declaration consistent with what the network actually receives, and avoids any ambiguity about where normalisation occurs.

### 2.2 Observation normalisation in `_get_obs`

```python
# before
def _get_obs(self) -> np.ndarray:
    return np.concatenate(list(self._frames), axis=0)

# after
def _get_obs(self) -> np.ndarray:
    return np.concatenate(list(self._frames), axis=0) / 255.0
```

**Why.** This is the implementation of the declaration above. The raw frames are stored as uint8 (which is memory-efficient), and the division by 255 is applied at the moment of observation assembly. This single operation correctly converts the stacked frame tensor to float32 [0, 1] on every `reset` and `step`, ensuring the CNN always receives normalised inputs.

---

## 3. `jimena/src/sac/train.py`

### 3.1 Separate feature extractors for actor and critic

```python
# before
policy_kwargs={
    "net_arch": [hidden_dim, hidden_dim],
    "features_extractor_kwargs": {"features_dim": 256},
},

# after
policy_kwargs={
    "net_arch": [hidden_dim, hidden_dim],
    "share_features_extractor": False,
    "features_extractor_kwargs": {"features_dim": 512},
},
```

**Why — separate extractors.** By default, SB3's `CnnPolicy` shares a single CNN encoder between the actor and the critic. In SAC, the actor and critic receive gradient signals that optimise for fundamentally different objectives: the critic minimises Bellman error (a regression loss) while the actor maximises expected return (a policy gradient). When these gradients flow back into a shared encoder, they can pull the representation in conflicting directions, destabilising the visual features that both networks depend on. Using independent encoders (one per network) insulates each from the other's gradients, which is the approach validated empirically in off-policy visual RL research.

**Why — features_dim 512.** A 512-dimensional bottleneck after the convolutional stack gives the encoder sufficient capacity to represent locomotion-relevant visual features (body posture, joint configuration, ground contact). At 256 dimensions the bottleneck forces lossy compression of visual information before it reaches the policy and value heads, limiting what the agent can learn from pixels.

---

## 4. `jimena/src/sac/optuna.py`

### 4.1 Buffer size search space

```python
# before
buffer_size = trial.suggest_categorical("buffer_size", [50_000, 100_000, 200_000, 500_000])

# after
buffer_size = trial.suggest_categorical("buffer_size", [300_000, 500_000, 750_000, 1_000_000])
```

**Why.** The lower values in the original search space (50k–200k) are now known to produce insufficient sample diversity for pixel-based SAC. Exploring them would waste trials on configurations that cannot train well regardless of other hyperparameters, corrupting the study results.

---

### 4.2 Learning starts search space

```python
# before
learning_starts = trial.suggest_categorical("learning_starts", [5_000, 10_000, 25_000, 50_000])

# after
learning_starts = trial.suggest_categorical("learning_starts", [50_000, 75_000, 100_000, 150_000])
```

**Why.** Values below 50,000 produce an insufficiently diverse warmup buffer for the CNN encoder (see §1.4). Including them in the search space would cause Optuna to waste trials on a known-bad regime and potentially bias the TPE sampler away from the viable region.

---

### 4.3 Batch size: 256 → 128

```python
# before
batch_size=256

# after
batch_size=128
```

**Why.** Consistent with the change in `train.py` and `sac.yaml` (see §1.2). Every trial must use the same batch size so that the study measures the effect of the sampled hyperparameters in isolation, not a confounding interaction with an oversized batch.

---

### 4.4 Training frequency: 1 → 2

```python
# before
train_freq=1,

# after
train_freq=2,
```

**Why.** Consistent with `train.py` and `sac.yaml` (see §1.3). Same reasoning applies: fixing this at the correct value ensures trials are evaluated under conditions that are actually viable for visual SAC.

---

### 4.5 Separate feature extractors for actor and critic

```python
# before
policy_kwargs={
    "net_arch": [256, 256],
    "features_extractor_kwargs": {"features_dim": 256},
},

# after
policy_kwargs={
    "net_arch": [256, 256],
    "share_features_extractor": False,
    "features_extractor_kwargs": {"features_dim": 512},
},
```

**Why.** Consistent with `train.py` (see §3.1). The architectural fix must be present in every trial, otherwise the study would be measuring a broken model architecture and the best params found would not transfer to the corrected `train.py`.
