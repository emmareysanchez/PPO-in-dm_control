# PPO Visual Control Project

This project trains and evaluates a PPO agent for `dm_control` continuous-control tasks from visual observations. It supports:

- training from scratch
- resuming from a checkpoint file
- resuming from the latest checkpoint in a checkpoint directory
- resuming into the same run directories or into fresh run directories
- periodic evaluation without training
- evaluation video export
- TensorBoard logging
- Optuna-based hyperparameter search

The main command-line interfaces are:

- `python -m src.ppo.train --config configs/ppo-train.yaml`
- `python -m src.ppo.search --config configs/ppo-search.yaml`

## Project layout

- `src/ppo/train.py`: training loop, checkpointing, resume logic, TensorBoard logging, periodic evaluation, CLI entrypoint
- `src/ppo/search.py`: Optuna search entrypoint
- `src/ppo/networks.py`: visual encoder, actor, critic
- `src/ppo/environment.py`: environment wrapper and environment interaction utilities
- `src/ppo/reward.py`: explicit reward function wrapper
- `src/ppo/evaluation.py`: evaluation episodes and video export
- `configs/ppo-train.yaml`: fixed training configuration
- `configs/ppo-search.yaml`: Optuna search ranges and search settings
- `update.md`: summary of changes relative to the previous implementation

## Installation

Install the Python dependencies required by the codebase. A typical setup includes at least:

```bash
pip install torch torchvision tensorboard tqdm pyyaml numpy opencv-python imageio optuna
pip install dm-control
```

Depending on your platform, `dm_control` may also require MuJoCo-related system setup.

## Training interface

### Start a new training run

```bash
python -m src.ppo.train --config configs/ppo-train.yaml
```

This starts PPO training from scratch using the values in `configs/ppo-train.yaml`.

### Resume from a specific checkpoint file

```bash
python -m src.ppo.train \
  --config configs/ppo-train.yaml \
  --resume checkpoints/ppo/<run_timestamp>/step_0000010000_episode_000010.pt
```

Use this when you want to continue from an exact checkpoint.

### Resume from the latest checkpoint in a run directory

```bash
python -m src.ppo.train \
  --config configs/ppo-train.yaml \
  --resume checkpoints/ppo/<run_timestamp>/
```

If `--resume` points to a directory, the trainer automatically loads the latest `.pt` checkpoint inside that directory.

### Resume and keep saving into the same run directories

```bash
python -m src.ppo.train \
  --config configs/ppo-train.yaml \
  --resume checkpoints/ppo/<run_timestamp>/
```

In this mode, the trainer keeps using the original run timestamp inferred from the checkpoint path, so logs, videos, and checkpoints continue under the same run.

### Resume but write into new run directories

```bash
python -m src.ppo.train \
  --config configs/ppo-train.yaml \
  --resume checkpoints/ppo/<run_timestamp>/ \
  --new-run
```

In this mode, the trainer loads the model and optimizer states from the checkpoint but creates a fresh timestamped run directory for new logs, videos, and checkpoints.

## Evaluation interface

There is no separate `eval.py` CLI in this project. Evaluation is triggered automatically by the training loop.

Evaluation behavior is controlled by the `evaluation` section of the config:

```yaml
evaluation:
  frequency: 10000
  episodes: 10
  run: "runs/ppo/"
  videos: "videos/ppo/"
  checkpoints: "checkpoints/ppo/"
```

### What evaluation does

At each evaluation phase, the trainer:

- runs `evaluation.episodes` episodes without training
- uses the actor mean as the deterministic evaluation policy
- records evaluation metrics
- saves one video per evaluation episode
- writes evaluation metrics to TensorBoard
- saves a checkpoint
- updates `best.pt` when the new evaluation mean reward is the best so far

### When evaluation runs

Evaluation runs every time the training step count advances by at least `evaluation.frequency` since the previous evaluation. A final evaluation also runs at the end of training.

### Where evaluation videos are saved

For normal training runs:

```text
videos/ppo/<train_start_timestamp>/
```

For Optuna search runs:

```text
videos/ppo-search/<search_start_timestamp>/<trial_run_timestamp>/
```

## Hyperparameter search interface

### Launch an Optuna search

```bash
python -m src.ppo.search --config configs/ppo-search.yaml
```

This launches multiple PPO training runs using Optuna. Each trial samples values from the ranges in `configs/ppo-search.yaml`.

### What the search config contains

The search config contains:

- global experiment settings such as `num_steps` and `trials`
- environment settings
- evaluation settings
- `search_space`, which defines min-max ranges for sampled hyperparameters

Example launch:

```bash
python -m src.ppo.search --config configs/ppo-search.yaml
```

### Search output directory structure

Search runs add an outer timestamp for the whole Optuna session, then an inner timestamp per training trial.

Runs:

```text
runs/ppo-search/<search_start_timestamp>/<trial_run_timestamp>/
```

Videos:

```text
videos/ppo-search/<search_start_timestamp>/<trial_run_timestamp>/
```

Checkpoints:

```text
checkpoints/ppo-search/<search_start_timestamp>/<trial_run_timestamp>/
```

## Output directories

### Standard training

A standard training run uses a single run timestamp:

```text
runs/ppo/<train_start_timestamp>/
videos/ppo/<train_start_timestamp>/
checkpoints/ppo/<train_start_timestamp>/
```

### Search trials

A search run uses two timestamp levels:

```text
runs/ppo-search/<search_start_timestamp>/<trial_run_timestamp>/
videos/ppo-search/<search_start_timestamp>/<trial_run_timestamp>/
checkpoints/ppo-search/<search_start_timestamp>/<trial_run_timestamp>/
```

## What gets saved in `runs`

Each run directory contains:

- a copy of the config file used to launch the run
- `resolved_config.json` with the parsed config actually used
- a full snapshot of `src/ppo`
- TensorBoard event files under `tensorboard/`

## TensorBoard interface

TensorBoard logs are written under:

```text
runs/.../<timestamp>/tensorboard/
```

Launch TensorBoard with:

```bash
tensorboard --logdir runs
```

The trainer logs at least these metrics:

- `train/episode-length`
- `train/episode-distance`
- `train/episode-reward`
- `train/episode-avg-reward`
- `train/episode-avg-speed`
- `train/episode-loss`
- `train/actor-loss`
- `train/critic-loss`
- `train/entropy`
- `eval/mean-reward`
- `eval/mean-length`
- `eval/mean-distance`
- `eval/mean-avg-speed`
- final evaluation counterparts at the end of training

## Progress bar behavior

The main training loop uses `tqdm` and shows moving averages over the last 100 episodes for the most important metrics:

- reward
- episode length
- distance
- average speed
- loss

The progress bar advances in environment steps, not episodes.

## Config interface

### Train config

Launch with:

```bash
python -m src.ppo.train --config configs/ppo-train.yaml
```

Key fields:

- `experiment.num_steps`: total training length in environment steps
- `environment.*`: task, observation shape, frame stack, action repeat, camera, episode length
- `architecture.hidden_dim`: actor/critic hidden layer width after the CNN encoder
- `ppo.*`: PPO hyperparameters
- `evaluation.*`: evaluation and output directory settings

### Search config

Launch with:

```bash
python -m src.ppo.search --config configs/ppo-search.yaml
```

Key fields:

- `experiment.trials`: number of Optuna trials
- `experiment.num_steps`: training budget per trial
- `search_space.*`: min-max search ranges
- `evaluation.*`: periodic evaluation and output roots for the search

## Typical command examples

### 1. Start training from zero

```bash
python -m src.ppo.train --config configs/ppo-train.yaml
```

### 2. Resume from a specific checkpoint

```bash
python -m src.ppo.train \
  --config configs/ppo-train.yaml \
  --resume checkpoints/ppo/20260330_120000/step_0000100000_episode_000120.pt
```

### 3. Resume from the latest checkpoint of a run, keeping the same run directories

```bash
python -m src.ppo.train \
  --config configs/ppo-train.yaml \
  --resume checkpoints/ppo/20260330_120000/
```

### 4. Resume from the latest checkpoint of a run, but save into a fresh run

```bash
python -m src.ppo.train \
  --config configs/ppo-train.yaml \
  --resume checkpoints/ppo/20260330_120000/ \
  --new-run
```

### 5. Launch hyperparameter search

```bash
python -m src.ppo.search --config configs/ppo-search.yaml
```

### 6. Launch TensorBoard

```bash
tensorboard --logdir runs
```

## Checkpoints

Checkpoints are saved during evaluation phases and once at the end of training.

A checkpoint file name looks like:

```text
step_0000100000_episode_000120.pt
```

The checkpoint contains:

- actor weights
- critic weights
- actor optimizer state
- critic optimizer state
- global training step
- episode index
- last evaluation step

The best evaluation checkpoint is also copied to:

```text
best.pt
```

inside the corresponding checkpoint directory.

## Notes

- Evaluation videos are saved as `.mp4` files through `imageio`.
- If MP4 writing fails on your machine, install a compatible ffmpeg backend.
- Search trials are currently launched as fresh runs only.
- Evaluation is integrated into training; there is no standalone evaluation command in this version.
