from __future__ import annotations
import multiprocessing as mp

if mp.get_start_method(allow_none=True) != "spawn":
    mp.set_start_method("spawn", force=True)

import warnings
from collections import Counter
from stable_baselines3 import A2C
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.callbacks import (
    CheckpointCallback,
    EvalCallback,
    BaseCallback,
)
from gymnasium.wrappers import RecordVideo
from torch.utils.tensorboard import SummaryWriter

from reil_hex_game.hex_engine.hex_env import HexEnv, OpponentWrapper
from reil_hex_game.agents.hex_cnn import HexCNN
from reil_hex_game.agents.rule_based_v4_agent import rule_based_v4_agent

warnings.filterwarnings("ignore", module="pygame.pkgdata")


# ────────────────────────────────────────────────────────────
# Custom TB callback for opponent-strategy counts
# ────────────────────────────────────────────────────────────
class StrategyTBCallback(BaseCallback):
    def __init__(self, log_dir: str, flush_freq: int = 1_000):
        super().__init__()
        self.flush_freq = flush_freq
        self.writer = SummaryWriter(log_dir)
        self.step_counter: Counter[str] = Counter()

    def _on_step(self) -> bool:
        for info in self.locals["infos"]:  # one per vec env
            if (name := info.get("opponent_strategy")):
                self.step_counter[name] += 1

        if self.num_timesteps % self.flush_freq == 0:
            for k, v in self.step_counter.items():
                self.writer.add_scalar(
                    f"strategies/opponent/{k}", v, self.num_timesteps
                )
            self.step_counter.clear()
        return True


# ────────────────────────────────────────────────────────────
# Helper factories
# ────────────────────────────────────────────────────────────
def make_env(
    size: int,
    opponent_fn=None,
    monitor_filename=None,
    render_mode: str | None = None,
):
    """Create a single HexEnv wrapped with optional opponent + monitor."""

    def _f():
        env = HexEnv(size=size, render_mode=render_mode)
        if opponent_fn:
            env = OpponentWrapper(env, opponent_fn)
        return Monitor(env, filename=monitor_filename)

    return _f


def make_eval_env(board_size: int, video_folder: str | None):
    """Single process eval env, optionally recording video."""
    def _factory():
        env = HexEnv(board_size, render_mode="rgb_array")
        env = OpponentWrapper(env, rule_based_v4_agent)
        env = Monitor(env)
        if video_folder:
            env = RecordVideo(
                env,
                video_folder=video_folder,
                episode_trigger=lambda ep: True,
                name_prefix="eval",
            )
        return env
    
    return DummyVecEnv([_factory])

# ────────────────────────────────────────────────────────────
# Main entry called from train_alg_hex.py
# ────────────────────────────────────────────────────────────
def train(args):
    # 1. Training workers (ALL head-less)
    vec_env = SubprocVecEnv(
        [
            make_env(
                size=args.board_size,
                opponent_fn=rule_based_v4_agent
                if i < args.num_envs * args.rule_fraction
                else None,
                render_mode=None,
            )
            for i in range(args.num_envs)
        ]
    )

    # 2. Evaluation env, video only if --video-eval flag
    video_dir = (
        f"{args.save_dir}/{args.run_name}/video" if args.video_eval else None
    )
    eval_env = make_eval_env(args.board_size, video_dir)

    # 3. Model definition
    policy_kwargs = dict(
        features_extractor_class=HexCNN,
        features_extractor_kwargs=dict(features_dim=128),
        share_features_extractor=True,
    )
    log_dir = f"{args.save_dir}/{args.run_name}"

    model = A2C(
        "CnnPolicy",
        vec_env,
        policy_kwargs=policy_kwargs,
        n_steps=128,
        gae_lambda=0.95,
        gamma=0.99,
        ent_coef=0.01,
        vf_coef=0.5,
        learning_rate=7e-4,
        tensorboard_log=log_dir,
        device=args.device,
        verbose=1,
    )

    # 4. Callbacks
    ckpt_cb = CheckpointCallback(
        save_freq=100_000 // args.num_envs,
        save_path=log_dir,
        name_prefix="model",
    )

    eval_cb = EvalCallback(
        eval_env,
        n_eval_episodes=30,
        eval_freq=50_000 // args.num_envs,
        deterministic=True,
        render=False,
        log_path=log_dir,
    )

    strategy_cb = StrategyTBCallback(
        log_dir=log_dir, flush_freq=10_000 // args.num_envs
    )

    # 5. Training loop
    model.learn(
        total_timesteps=args.timesteps,
        callback=[ckpt_cb, eval_cb, strategy_cb],
    )
    model.save(f"{args.save_dir}/{args.run_name}/final")
