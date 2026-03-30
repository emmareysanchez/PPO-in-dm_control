# PPO codebase update

This update restructures the single-file prototype into a small PPO package with a command-line interface, resumable runs, evaluation support, checkpoint management, TensorBoard logging, and Optuna-based hyperparameter search.

## Main changes relative to the previous implementation

### 1. Codebase modularization

The original adaptation kept most of the logic inside one script. The new version splits responsibilities into dedicated modules:

* `src/ppo/train.py`: training loop, checkpointing, run management, TensorBoard logging, CLI entry point.
* `src/ppo/networks.py`: policy, value, and encoder networks.
* `src/ppo/environment.py`: visual dm_control wrapper and interaction utilities.
* `src/ppo/reward.py`: explicit reward definition.
* `src/ppo/evaluation.py`: deterministic evaluation and video saving.
* `src/ppo/search.py`: Optuna-driven hyperparameter search.

This makes the project easier to maintain, test, and extend.

### 2. PPO only

The previous configuration mixed PPO with SAC-related settings. All SAC-specific configuration has been removed from the PPO training pipeline.

The new codebase only implements PPO.

### 3. Continuous actions restored

The previous adaptation discretized the action space by mapping categorical decisions to action prototypes. That is no longer used.

The new implementation restores continuous control:

* the policy outputs Gaussian parameters (`mu`, `std`),
* actions are sampled from a Normal distribution,
* actions are clipped to the environment action bounds.

This is much closer to the original `walker.py` PPO implementation.

### 4. Visual observations kept, but made configurable

The visual pipeline is preserved, but it is now wrapped inside `Environment`:

* RGB or grayscale rendering,
* configurable frame stacking,
* configurable camera id,
* configurable action repeat,
* configurable maximum episode length.

The wrapper is responsible for producing stacked observations with shape `[channels * frame_stack, height, width]`.

### 5. Explicit reward function

The previous script implicitly used `timestep.reward`. The new code makes that choice explicit in `src/ppo/reward.py`.

At the moment, the reward is exactly the reward emitted by dm_control, with no extra shaping.

### 6. Training length based on steps

The old code used episode count as the main stopping criterion. The new code switches to `num_steps`, as requested.

This makes training length more stable across runs because it does not depend on how long individual episodes last.

### 7. Evaluation phases and video generation

The previous script did not define a clear evaluation pipeline. The new implementation introduces periodic evaluation:

* evaluation runs every `evaluation.frequency` environment steps,
* each evaluation runs `evaluation.episodes` episodes,
* evaluation uses deterministic actions (policy mean),
* videos are saved for every evaluation episode.

Evaluation outputs are now organized hierarchically:

```
videos/.../<run_timestamp>/<step_zfill>/episode_01.mp4
```

instead of embedding the step in the filename. This improves readability and grouping of evaluation results.

Additionally, video generation has been made explicit and stable:

* frames are padded to satisfy codec constraints (e.g., divisibility by 16),
* this avoids implicit resizing performed by FFmpeg,
* no warnings are produced during video export,
* padding is applied only at save time and does not affect training.

### 8. Run directory management

The new implementation creates timestamped run directories and stores outputs under:

* `runs/.../<timestamp>/`
* `videos/.../<timestamp>/`
* `checkpoints/.../<timestamp>/`

For hyperparameter search, there is an additional timestamp layer:

* `runs/ppo-search/<search_timestamp>/<trial_timestamp>/`
* and the equivalent structure for videos and checkpoints.

### 9. Checkpointing and resume support

The old implementation only saved model weights occasionally. The new code saves full checkpoints with:

* actor weights,
* critic weights,
* optimizer states,
* global step,
* episode index,
* last evaluation step.

The CLI now supports:

* training from scratch,
* resuming from a specific checkpoint,
* resuming from the latest checkpoint of a run directory,
* resuming into the same run directories,
* resuming into a new set of run directories with `--new-run`.

### 10. TensorBoard logging

The new implementation logs the requested metrics to TensorBoard:

* episode length,
* episode distance,
* episode reward,
* rolling average reward,
* episode average speed,
* episode loss.

It also logs additional useful PPO diagnostics:

* actor loss,
* critic loss,
* policy entropy,
* evaluation metrics.

### 11. Progress bar with rolling metrics

The main training loop now uses `tqdm` and shows rolling averages over the last 100 episodes for the most relevant metrics:

* reward,
* episode length,
* distance,
* average speed,
* loss.

### 12. Search entry point with Optuna

A new `src/ppo/search.py` entry point has been added.

It launches multiple PPO runs with Optuna and samples hyperparameters from min-max ranges defined in `configs/ppo-search.yaml`.

This makes the search process reproducible and configurable from YAML instead of hard-coding search ranges.

### 13. Run snapshotting

Each run now stores:

* a copy of the config file,
* a JSON dump of the parsed config,
* a snapshot of the full `src/ppo` directory,
* TensorBoard data.

This improves experiment traceability.

## Summary

The new implementation keeps the useful part of the visual PPO adaptation while removing the categorical action discretization and any SAC-related configuration. The resulting project is a cleaner, fully continuous PPO pipeline with proper experiment management, evaluation, logging, checkpointing, resume support, and hyperparameter search.

---

### PROMPT

```

necesito que reescribas el cï¿½digo con la siguiente estructura:

src/ppo/train.py: defines train loop with evaluation phases from time to time. It must also define the main function to launch the train like this: python -m src.ppo.train --config configs/ppo-train.yaml
src/ppo/networks.py: defines all involved neural nework functions / classes
src/ppo/environment.py: defines all environment related functions / clases (e.g. en Environment class to interact with the environment
src/ppo/reward.py: just defines reward function (use the same that is currently being used by the environment, but make its definition explicit in a function)
src/ppo/evaluation.py: defines any object specifically related to the evaluation

src/ppo/search.py -> defines an interface similar to that of train but instead of just launching one single train run it uses optuna to launch mulitple train runs searching withing the reasonable ranges of hiperparameter values (give me a ppo-search.yaml file that contains the definition of min-max ranges rather than just values; it also must define train trials). This will be launched with: python -m src.ppo.search --config configs/ppo-search.yaml



NOTES:
0. to determine train run length change num_episodes config setting by num_steps
1. evaluation will execute a few episodes without training and save videos of them
2. in the config files you must include a "evaluation" section to define:
 - frequency: from how many to how many train episodes will the 
 - episodes: number of episodes per evaluation phase (10)
 - run: location of "run" directory, containing info about metrics throuhout the episode (runs/ppo/)
 - videos: location of "videos" directory, where the videos of the evaluation episodes will be saved (videos/ppo/)
 - checkpoints: location of the "checkpoints" directory, where the model checkpoints will be saved (checkpoints/ppo/ )
the code wont save de run data/videos/checkpoins directly to those directories but to path/to/dir/<train start timestamp>/
the directory names for ppo-search will be runs/ppo-search/ and equivalent for videos and checkpoints (but instead of saving the content directly under the timestamp, there will be a sencond layer of timestamps (with the the optuna run start-time), and then the content)
(in the case of the videos, there will be an additional layer of directories with the zfilled number of steps at corresponding evaluation phase)
3. runs will contain:
 - a copy of both the config file used and everyting under de src/ppo directory
 - all tensorboard-related info
4. tensorboard will log:
 - episode-length
 - episode-distance
 - episode-reward
 - episode-avg-reward
 - episode-avg-speed
 - episode-loss
 (and any other that you consider useful for monitoring)
5. tambiï¿½n quiero que escibas un update.md describiendo todo lo que se ha cambiado respecto de la implementaciï¿½n de mi compaï¿½era
6. todo en inglï¿½s (el update.md, los comentarios, ...)
7. el proyecto no debe solo ejecutar un entrenamiento desde 0, sino tambiï¿½n retomar entrenamientos (y hacer evaluaciones) con checkpoints ya guardados, insperate en la interfaz de este proyecto anterior usando una DQN en lugar de PPO:

Start training from zero:
python -m src.dqn.train --config configs/dqn.yaml
Resume from a specific checkpoint:
python -m src.dqn.train \
  --config configs/dqn.yaml \
  --resume checkpoints/dqn/Walker2d-v5_20260218_185009/step_5000000_07-36-34.pt
Resume from the latest checkpoint of a run saving videos & checkpoints on the same run directories:
python -m src.dqn.train \
  --config configs/dqn.yaml \
  --resume checkpoints/dqn/Walker2d-v5_20260218_185009/
Resume from the latest checkpoint of a run saving videos & checkpoints on a new run directories:
python -m src.dqn.train \
  --config configs/dqn.yaml \
  --resume checkpoints/dqn/Walker2d-v5_20260218_185009/ \
  --new-run
8. el bucle prindipal lo acompaï¿½arï¿½s de una barra de tqdm, que al mismo tiempo mostrarï¿½ los valores de las mï¿½tricas mï¿½s importantes promediado por los ï¿½ltimos 100 episodios

```
