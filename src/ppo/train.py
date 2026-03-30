from __future__ import annotations

import argparse
import datetime as dt
import json
import shutil
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm

from src.ppo.environment import Environment
from src.ppo.evaluation import Evaluator
from src.ppo.networks import PolicyNetwork, ValueNetwork


@dataclass
class RunDirectories:
    """Filesystem layout for one training run."""

    run_root: Path
    run_dir: Path
    videos_dir: Path
    checkpoints_dir: Path


@dataclass
class TrainState:
    """Mutable training state for checkpointing."""

    global_step: int = 0
    episode_idx: int = 0
    last_eval_step: int = 0


class PPOAgent:
    """PPO agent implementation for continuous control."""

    def __init__(self, config: dict[str, Any], env: Environment) -> None:
        """Initialize the agent.

        Args:
            config: Full experiment configuration.
            env: Environment wrapper.
        """
        self.config: dict[str, Any] = config
        self.device: torch.device = torch.device(
            config["experiment"]["device"] if torch.cuda.is_available() else "cpu"
        )
        ppo_config: dict[str, Any] = config["ppo"]
        hidden_dim: int = int(config["architecture"]["hidden_dim"])

        self.actor: PolicyNetwork = PolicyNetwork(
            input_channels=env.input_channels,
            action_dim=env.action_dim,
            observation_height=env.height,
            observation_width=env.width,
            hidden_dim=hidden_dim,
        ).to(self.device)
        self.critic: ValueNetwork = ValueNetwork(
            input_channels=env.input_channels,
            observation_height=env.height,
            observation_width=env.width,
            hidden_dim=hidden_dim,
        ).to(self.device)

        self.actor_optimizer: torch.optim.Adam = torch.optim.Adam(
            self.actor.parameters(),
            lr=float(ppo_config["actor_lr"]),
        )
        self.critic_optimizer: torch.optim.Adam = torch.optim.Adam(
            self.critic.parameters(),
            lr=float(ppo_config["critic_lr"]),
        )

        self.gamma: float = float(ppo_config["gamma"])
        self.lambd: float = float(ppo_config["lambd"])
        self.k_epochs: int = int(ppo_config["k_epochs"])
        self.eps_clip: float = float(ppo_config["eps_clip"])
        self.entropy_coef: float = float(ppo_config["entropy_coef"])
        self.value_coef: float = float(ppo_config.get("value_coef", 0.5))
        self.max_grad_norm: float = float(ppo_config.get("max_grad_norm", 0.5))

        self.action_low: np.ndarray = env.action_spec.minimum.astype(np.float32)
        self.action_high: np.ndarray = env.action_spec.maximum.astype(np.float32)

    def select_action(self, observation: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Sample an action from the policy.

        Args:
            observation: Stacked observation tensor.

        Returns:
            Sampled action and per-dimension log-probabilities.
        """
        observation_tensor: torch.Tensor = torch.as_tensor(
            observation,
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(0)
        with torch.no_grad():
            mu, std = self.actor(observation_tensor)
            dist = torch.distributions.Normal(mu, std)
            action_tensor: torch.Tensor = dist.sample()
            log_prob_tensor: torch.Tensor = dist.log_prob(action_tensor)
        action: np.ndarray = action_tensor.squeeze(0).cpu().numpy().astype(np.float32)
        action = np.clip(action, self.action_low, self.action_high)
        log_prob: np.ndarray = (
            log_prob_tensor.squeeze(0).cpu().numpy().astype(np.float32)
        )
        return action, log_prob

    def update(self, trajectory: dict[str, list[Any]]) -> dict[str, float]:
        """Run the PPO optimization step.

        Args:
            trajectory: One on-policy trajectory.

        Returns:
            Scalar losses and diagnostics.
        """
        states: torch.Tensor = torch.as_tensor(
            np.array(trajectory["obs"], dtype=np.float32),
            dtype=torch.float32,
            device=self.device,
        )
        actions: torch.Tensor = torch.as_tensor(
            np.array(trajectory["actions"], dtype=np.float32),
            dtype=torch.float32,
            device=self.device,
        )
        rewards: torch.Tensor = torch.as_tensor(
            np.array(trajectory["rewards"], dtype=np.float32),
            dtype=torch.float32,
            device=self.device,
        ).view(-1, 1)
        next_states: torch.Tensor = torch.as_tensor(
            np.array(trajectory["next_obs"], dtype=np.float32),
            dtype=torch.float32,
            device=self.device,
        )
        dones: torch.Tensor = torch.as_tensor(
            np.array(trajectory["dones"], dtype=np.float32),
            dtype=torch.float32,
            device=self.device,
        ).view(-1, 1)
        old_log_probs: torch.Tensor = torch.as_tensor(
            np.array(trajectory["log_probs"], dtype=np.float32),
            dtype=torch.float32,
            device=self.device,
        )

        with torch.no_grad():
            values: torch.Tensor = self.critic(states)
            next_values: torch.Tensor = self.critic(next_states)
            td_target: torch.Tensor = rewards + self.gamma * next_values * (1.0 - dones)
            delta: torch.Tensor = td_target - values

            advantages_list: list[np.ndarray] = []
            gae: np.ndarray = np.zeros(shape=(1,), dtype=np.float32)
            for delta_t, done_t in zip(
                reversed(delta.detach().cpu().numpy()),
                reversed(dones.detach().cpu().numpy()),
            ):
                gae = delta_t + self.gamma * self.lambd * gae * (1.0 - done_t)
                advantages_list.append(gae.copy())
            advantages_list.reverse()
            advantages: torch.Tensor = torch.as_tensor(
                np.array(advantages_list, dtype=np.float32),
                dtype=torch.float32,
                device=self.device,
            )
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            returns: torch.Tensor = advantages + values

        actor_loss_value: float = 0.0
        critic_loss_value: float = 0.0
        entropy_value: float = 0.0
        total_loss_value: float = 0.0

        for _ in range(self.k_epochs):
            mu, std = self.actor(states)
            dist = torch.distributions.Normal(mu, std)
            log_probs: torch.Tensor = dist.log_prob(actions)
            entropy: torch.Tensor = dist.entropy().mean()

            ratio: torch.Tensor = torch.exp(
                log_probs.sum(dim=1, keepdim=True)
                - old_log_probs.sum(dim=1, keepdim=True)
            )
            surrogate_1: torch.Tensor = ratio * advantages
            surrogate_2: torch.Tensor = (
                torch.clamp(
                    ratio,
                    1.0 - self.eps_clip,
                    1.0 + self.eps_clip,
                )
                * advantages
            )

            actor_loss: torch.Tensor = -torch.min(surrogate_1, surrogate_2).mean()
            critic_loss: torch.Tensor = F.mse_loss(
                self.critic(states), returns.detach()
            )
            total_loss: torch.Tensor = (
                actor_loss + self.value_coef * critic_loss - self.entropy_coef * entropy
            )

            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
            self.actor_optimizer.step()
            self.critic_optimizer.step()

            actor_loss_value = float(actor_loss.item())
            critic_loss_value = float(critic_loss.item())
            entropy_value = float(entropy.item())
            total_loss_value = float(total_loss.item())

        return {
            "actor_loss": actor_loss_value,
            "critic_loss": critic_loss_value,
            "entropy": entropy_value,
            "loss": total_loss_value,
        }


def load_config(config_path: str) -> dict[str, Any]:
    """Load a YAML configuration file.

    Args:
        config_path: YAML file path.

    Returns:
        Parsed config dictionary.
    """
    with open(config_path, encoding="utf-8") as file:
        return yaml.safe_load(file)


def timestamp_string() -> str:
    """Generate a filesystem-friendly timestamp.

    Returns:
        Timestamp string.
    """
    return dt.datetime.now().strftime("%Y%m%d_%H%M%S")


def copy_source_tree(destination: Path) -> None:
    """Copy the current src/ppo tree into the run directory.

    Args:
        destination: Destination directory.
    """
    source_dir: Path = Path(__file__).resolve().parent
    target_dir: Path = destination / "source_snapshot"
    if target_dir.exists():
        shutil.rmtree(target_dir)
    shutil.copytree(source_dir, target_dir)


def save_config_copy(
    config: dict[str, Any], config_path: str, destination: Path
) -> None:
    """Save the original config and its parsed content.

    Args:
        config: Parsed configuration.
        config_path: Original config path.
        destination: Destination directory.
    """
    destination.mkdir(parents=True, exist_ok=True)
    shutil.copy2(config_path, destination / Path(config_path).name)
    with open(destination / "resolved_config.json", "w", encoding="utf-8") as file:
        json.dump(config, file, indent=2)


def latest_checkpoint_in_directory(path: Path) -> Path:
    """Find the latest checkpoint inside a directory.

    Args:
        path: Directory containing checkpoint files.

    Returns:
        Latest checkpoint path.
    """
    checkpoint_files: list[Path] = sorted(path.glob("*.pt"))
    if not checkpoint_files:
        raise FileNotFoundError(f"No checkpoint files found in {path}")
    return checkpoint_files[-1]


def resolve_resume_path(resume: str) -> Path:
    """Resolve a resume argument to a checkpoint file.

    Args:
        resume: File or directory path.

    Returns:
        Checkpoint file path.
    """
    resume_path: Path = Path(resume)
    if resume_path.is_dir():
        return latest_checkpoint_in_directory(path=resume_path)
    return resume_path


def infer_run_timestamp_from_checkpoint(checkpoint_path: Path) -> str:
    """Infer the run timestamp from a checkpoint path.

    Args:
        checkpoint_path: Path to a checkpoint file.

    Returns:
        Run timestamp string.
    """
    return checkpoint_path.parent.name


def create_run_directories(
    config: dict[str, Any],
    resume: str | None,
    new_run: bool,
) -> RunDirectories:
    """Prepare all directories for the current training session.

    Args:
        config: Full experiment configuration.
        resume: Optional checkpoint file or run directory.
        new_run: Whether to force creation of a new run directory.

    Returns:
        RunDirectories object.
    """
    run_root_base: Path = Path(config["evaluation"]["run"])
    videos_root_base: Path = Path(config["evaluation"]["videos"])
    checkpoints_root_base: Path = Path(config["evaluation"]["checkpoints"])

    if resume is None or new_run:
        run_timestamp: str = timestamp_string()
    else:
        checkpoint_path: Path = resolve_resume_path(resume=resume)
        run_timestamp = infer_run_timestamp_from_checkpoint(
            checkpoint_path=checkpoint_path
        )

    run_dir: Path = run_root_base / run_timestamp
    videos_dir: Path = videos_root_base / run_timestamp
    checkpoints_dir: Path = checkpoints_root_base / run_timestamp

    run_dir.mkdir(parents=True, exist_ok=True)
    videos_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    return RunDirectories(
        run_root=run_root_base,
        run_dir=run_dir,
        videos_dir=videos_dir,
        checkpoints_dir=checkpoints_dir,
    )


def save_checkpoint(
    agent: PPOAgent,
    state: TrainState,
    checkpoint_dir: Path,
) -> Path:
    """Persist a training checkpoint.

    Args:
        agent: PPO agent.
        state: Training state.
        checkpoint_dir: Destination directory.

    Returns:
        Saved checkpoint path.
    """
    checkpoint_path: Path = checkpoint_dir / (
        f"step_{state.global_step:010d}_episode_{state.episode_idx:06d}.pt"
    )
    torch.save(
        {
            "global_step": state.global_step,
            "episode_idx": state.episode_idx,
            "last_eval_step": state.last_eval_step,
            "actor_state_dict": agent.actor.state_dict(),
            "critic_state_dict": agent.critic.state_dict(),
            "actor_optimizer_state_dict": agent.actor_optimizer.state_dict(),
            "critic_optimizer_state_dict": agent.critic_optimizer.state_dict(),
        },
        checkpoint_path,
    )
    return checkpoint_path


def load_checkpoint(agent: PPOAgent, checkpoint_path: Path) -> TrainState:
    """Restore a training checkpoint.

    Args:
        agent: PPO agent.
        checkpoint_path: Checkpoint file path.

    Returns:
        Restored training state.
    """
    payload: dict[str, Any] = torch.load(
        checkpoint_path,
        map_location=agent.device,
    )
    agent.actor.load_state_dict(payload["actor_state_dict"])
    agent.critic.load_state_dict(payload["critic_state_dict"])
    agent.actor_optimizer.load_state_dict(payload["actor_optimizer_state_dict"])
    agent.critic_optimizer.load_state_dict(payload["critic_optimizer_state_dict"])
    return TrainState(
        global_step=int(payload["global_step"]),
        episode_idx=int(payload["episode_idx"]),
        last_eval_step=int(payload.get("last_eval_step", 0)),
    )


def collect_episode(
    env: Environment,
    agent: PPOAgent,
) -> tuple[dict[str, list[Any]], dict[str, float]]:
    """Collect one training episode.

    Args:
        env: Environment wrapper.
        agent: PPO agent.

    Returns:
        Trajectory dictionary and episode metrics.
    """
    observation, _ = env.reset()
    done: bool = False
    trajectory: dict[str, list[Any]] = {
        "obs": [],
        "actions": [],
        "rewards": [],
        "next_obs": [],
        "dones": [],
        "log_probs": [],
    }

    final_info: dict[str, float] = {
        "episode_step": 0.0,
        "episode_reward": 0.0,
        "episode_distance": 0.0,
        "episode_avg_speed": 0.0,
    }

    while not done:
        action, log_prob = agent.select_action(observation=observation)
        step_result = env.step(action=action)
        trajectory["obs"].append(observation)
        trajectory["actions"].append(action)
        trajectory["rewards"].append(step_result.reward)
        trajectory["next_obs"].append(step_result.observation)
        trajectory["dones"].append(step_result.done)
        trajectory["log_probs"].append(log_prob)
        observation = step_result.observation
        done = step_result.done
        final_info = step_result.info

    metrics: dict[str, float] = {
        "episode_length": float(final_info["episode_step"]),
        "episode_reward": float(final_info["episode_reward"]),
        "episode_distance": float(final_info["episode_distance"]),
        "episode_avg_speed": float(final_info["episode_avg_speed"]),
    }
    return trajectory, metrics


def run_training(
    config: dict[str, Any],
    config_path: str,
    resume: str | None = None,
    new_run: bool = False,
    trial: Any | None = None,
    search_timestamp: str | None = None,
) -> dict[str, float]:
    """Execute one full training run.

    Args:
        config: Parsed configuration.
        config_path: Original config path.
        resume: Optional checkpoint file or run directory.
        new_run: Whether to force a new run directory when resuming.
        trial: Optional Optuna trial.
        search_timestamp: Optional outer timestamp for search runs.

    Returns:
        Summary metrics for the completed run.
    """
    if search_timestamp is not None:
        base_run: Path = Path(config["evaluation"]["run"]) / search_timestamp
        base_videos: Path = Path(config["evaluation"]["videos"]) / search_timestamp
        base_checkpoints: Path = (
            Path(config["evaluation"]["checkpoints"]) / search_timestamp
        )
        config = dict(config)
        config["evaluation"] = dict(config["evaluation"])
        config["evaluation"]["run"] = str(base_run)
        config["evaluation"]["videos"] = str(base_videos)
        config["evaluation"]["checkpoints"] = str(base_checkpoints)

    run_dirs: RunDirectories = create_run_directories(
        config=config,
        resume=resume,
        new_run=new_run,
    )
    copy_source_tree(destination=run_dirs.run_dir)
    save_config_copy(
        config=config, config_path=config_path, destination=run_dirs.run_dir
    )

    writer: SummaryWriter = SummaryWriter(log_dir=str(run_dirs.run_dir / "tensorboard"))
    env: Environment = Environment(config=config)
    agent: PPOAgent = PPOAgent(config=config, env=env)
    evaluator: Evaluator = Evaluator(config=config, videos_dir=run_dirs.videos_dir)

    state: TrainState = TrainState()
    if resume is not None:
        checkpoint_path: Path = resolve_resume_path(resume=resume)
        state = load_checkpoint(agent=agent, checkpoint_path=checkpoint_path)

    num_steps: int = int(config["experiment"]["num_steps"])
    eval_frequency: int = int(config["evaluation"]["frequency"])

    recent_rewards: deque[float] = deque(maxlen=100)
    recent_lengths: deque[float] = deque(maxlen=100)
    recent_distances: deque[float] = deque(maxlen=100)
    recent_speeds: deque[float] = deque(maxlen=100)
    recent_losses: deque[float] = deque(maxlen=100)

    progress: tqdm = tqdm(
        total=num_steps,
        initial=state.global_step,
        desc="PPO training",
        unit="step",
    )

    best_eval_reward: float = -float("inf")

    while state.global_step < num_steps:
        trajectory, episode_metrics = collect_episode(env=env, agent=agent)
        update_metrics: dict[str, float] = agent.update(trajectory=trajectory)
        episode_steps: int = int(episode_metrics["episode_length"])

        state.global_step += episode_steps
        state.episode_idx += 1

        recent_rewards.append(episode_metrics["episode_reward"])
        recent_lengths.append(episode_metrics["episode_length"])
        recent_distances.append(episode_metrics["episode_distance"])
        recent_speeds.append(episode_metrics["episode_avg_speed"])
        recent_losses.append(update_metrics["loss"])

        writer.add_scalar(
            "train/episode-length", episode_metrics["episode_length"], state.global_step
        )
        writer.add_scalar(
            "train/episode-distance",
            episode_metrics["episode_distance"],
            state.global_step,
        )
        writer.add_scalar(
            "train/episode-reward", episode_metrics["episode_reward"], state.global_step
        )
        writer.add_scalar(
            "train/episode-avg-reward",
            float(np.mean(recent_rewards)),
            state.global_step,
        )
        writer.add_scalar(
            "train/episode-avg-speed",
            episode_metrics["episode_avg_speed"],
            state.global_step,
        )
        writer.add_scalar(
            "train/episode-loss", update_metrics["loss"], state.global_step
        )
        writer.add_scalar(
            "train/actor-loss", update_metrics["actor_loss"], state.global_step
        )
        writer.add_scalar(
            "train/critic-loss", update_metrics["critic_loss"], state.global_step
        )
        writer.add_scalar("train/entropy", update_metrics["entropy"], state.global_step)

        progress.update(episode_steps)
        progress.set_postfix(
            reward=f"{np.mean(recent_rewards):.2f}",
            length=f"{np.mean(recent_lengths):.1f}",
            distance=f"{np.mean(recent_distances):.2f}",
            speed=f"{np.mean(recent_speeds):.4f}",
            loss=f"{np.mean(recent_losses):.4f}",
        )

        if state.global_step - state.last_eval_step >= eval_frequency:
            eval_result = evaluator.evaluate(
                actor=agent.actor,
                device=agent.device,
                global_step=state.global_step,
            )
            state.last_eval_step = state.global_step

            writer.add_scalar(
                "eval/mean-reward", eval_result.mean_reward, state.global_step
            )
            writer.add_scalar(
                "eval/mean-length", eval_result.mean_length, state.global_step
            )
            writer.add_scalar(
                "eval/mean-distance", eval_result.mean_distance, state.global_step
            )
            writer.add_scalar(
                "eval/mean-avg-speed", eval_result.mean_avg_speed, state.global_step
            )

            checkpoint_path: Path = save_checkpoint(
                agent=agent,
                state=state,
                checkpoint_dir=run_dirs.checkpoints_dir,
            )

            if eval_result.mean_reward > best_eval_reward:
                best_eval_reward = eval_result.mean_reward
                best_path: Path = run_dirs.checkpoints_dir / "best.pt"
                shutil.copy2(checkpoint_path, best_path)

            if trial is not None:
                trial.report(eval_result.mean_reward, step=state.global_step)
                if trial.should_prune():
                    writer.flush()
                    writer.close()
                    progress.close()
                    raise RuntimeError("Optuna trial pruned")

    final_eval = evaluator.evaluate(
        actor=agent.actor,
        device=agent.device,
        global_step=state.global_step,
    )
    writer.add_scalar(
        "eval/final-mean-reward", final_eval.mean_reward, state.global_step
    )
    writer.add_scalar(
        "eval/final-mean-length", final_eval.mean_length, state.global_step
    )
    writer.add_scalar(
        "eval/final-mean-distance", final_eval.mean_distance, state.global_step
    )
    writer.add_scalar(
        "eval/final-mean-avg-speed", final_eval.mean_avg_speed, state.global_step
    )
    save_checkpoint(agent=agent, state=state, checkpoint_dir=run_dirs.checkpoints_dir)

    progress.close()
    writer.flush()
    writer.close()

    return {
        "final_eval_reward": final_eval.mean_reward,
        "final_eval_length": final_eval.mean_length,
        "final_eval_distance": final_eval.mean_distance,
        "final_eval_avg_speed": final_eval.mean_avg_speed,
    }


def parse_args() -> argparse.Namespace:
    """Parse command line arguments.

    Returns:
        Parsed CLI arguments.
    """
    parser: argparse.ArgumentParser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--new-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    """Launch PPO training from the command line."""
    args: argparse.Namespace = parse_args()
    config: dict[str, Any] = load_config(config_path=args.config)
    run_training(
        config=config,
        config_path=args.config,
        resume=args.resume,
        new_run=args.new_run,
    )


if __name__ == "__main__":
    main()
