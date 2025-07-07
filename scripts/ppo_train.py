# scripts/ppo_train.py
from __future__ import annotations
import multiprocessing as mp
import random
import warnings
from collections import Counter, defaultdict

# ── third-party --------------------------------------------------------
import numpy as np
from gymnasium.wrappers import RecordVideo
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import (
    SubprocVecEnv,
    DummyVecEnv,
    VecVideoRecorder,
)
from torch.utils.tensorboard import SummaryWriter

# ── project imports -----------------------------------------------------
from reil_hex_game.hex_engine.hex_env import HexEnv, OpponentWrapper
from reil_hex_game.hex_engine.side_swap_wrapper import SideSwapWrapper
from reil_hex_game.agents.hex_cnn import HexCNN
from reil_hex_game.agents.rule_based_v3_agent import rule_based_v3_agent
from reil_hex_game.agents.rule_based_v4_agent import rule_based_v4_agent
from reil_hex_game.agents.ppo_opponent_agent import ppo_opponent_agent, set_opponent_model_path

# silence setuptools-81 deprecation noise triggered by pygame
warnings.filterwarnings("ignore", module="pygame.pkgdata")

# ── Windows / CUDA multiprocessing safe-guard --------------------------
if mp.get_start_method(allow_none=True) != "spawn":
    mp.set_start_method("spawn", force=True)

# ───────────────────────────────────────────────────────────────────────
# Utility: random opponent
# ───────────────────────────────────────────────────────────────────────
def random_agent(board, legal_moves):
    """Return a uniformly-random legal coordinate."""
    return random.choice(legal_moves)

# ───────────────────────────────────────────────────────────────────────
# TensorBoard callback (unchanged from A2C version)
# ───────────────────────────────────────────────────────────────────────
class ComprehensiveTBCallback(BaseCallback):
    """
    A custom callback for logging all numeric values from the info dict
    to TensorBoard. It also handles the opponent strategy logging.
    """
    def __init__(self, log_dir: str, flush_freq: int = 1000):
        super().__init__()
        self.log_dir = log_dir
        self.flush_freq = flush_freq
        self.writer = None # Will be set in _on_training_start
        self.info_buffer = defaultdict(list)
        self.strategy_counter = defaultdict(int)

    def _on_training_start(self) -> None:
        if self.writer is None:
            self.writer = SummaryWriter(log_dir=self.log_dir)

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            # Log all numeric values from the info dict
            for key, value in info.items():
                if isinstance(value, (int, float)):
                    self.info_buffer[key].append(value)
            
            # Special handling for opponent strategy (if present)
            if "opponent_strategy" in info:
                self.strategy_counter[info["opponent_strategy"]] += 1

        if self.num_timesteps % self.flush_freq == 0:
            # Log the mean of the buffered info values
            for key, values in self.info_buffer.items():
                if values:
                    self.writer.add_scalar(f"step_info/{key}", np.mean(values), self.num_timesteps)
            
            # Log strategy counts
            for key, value in self.strategy_counter.items():
                self.writer.add_scalar(f"strategies/{key}", value, self.num_timesteps)

            # Clear buffers
            self.info_buffer.clear()
            self.strategy_counter.clear()
            
        return True

    def _on_training_end(self) -> None:
        if self.writer:
            self.writer.close()


# ───────────────────────────────────────────────────────────────────────
# Fast evaluation callback – skips evaluation at t = 0
# (inherits mask-aware logic from MaskableEvalCallback)
# ───────────────────────────────────────────────────────────────────────
class FastEval(MaskableEvalCallback):
    def __init__(self, *a, warmup_steps: int = 10_000, **kw):
        super().__init__(*a, **kw)
        self.warmup_steps = warmup_steps

    def _on_training_start(self) -> None:  # suppress first eval
        self.last_eval_step = -self.warmup_steps


# ───────────────────────────────────────────────────────────────────────
# Env factories
# ───────────────────────────────────────────────────────────────────────
def mask_fn(env: HexEnv):
    """Return Boolean vector - True where the move is legal."""
    mask = env.action_masks()
    assert mask.any(), "Mask is all-False - HexEnv.action_masks() broken?"
    return mask


def make_train_env(board_size: int, opponent_fn, wrapper_type: str):
    """A factory that creates a training environment with the specified wrapper."""
    def _factory():
        env = HexEnv(board_size)
        if wrapper_type == 'opponent':
            env = OpponentWrapper(env, opponent_fn)
        elif wrapper_type == 'two_player':
            env = SideSwapWrapper(env, opponent_fn, prob_start_first=0.5)
        else:
            raise ValueError(f"Unknown wrapper type: {wrapper_type}")
        env = ActionMasker(env, mask_fn)
        return Monitor(env) # Wrap with Monitor to get automatic episode logging
    return _factory


def make_eval_env(board_size: int, video_folder: str | None, video_len: int = 300):
    """Single-worker DummyVecEnv, wrapped for masking + (optional) video."""
    def _factory():
        env = HexEnv(board_size, render_mode="rgb_array")
        # env = OpponentWrapper(env, rule_based_v4_agent)
        env = SideSwapWrapper(env, rule_based_v4_agent, prob_start_first=0.5)
        env = ActionMasker(env, mask_fn)
        return Monitor(env)

    venv = DummyVecEnv([_factory])
    if video_folder:
        venv = VecVideoRecorder(
            venv,
            video_folder=video_folder,
            record_video_trigger=lambda step: True,
            video_length=video_len,
            name_prefix="eval",
        )
    return venv


# ───────────────────────────────────────────────────────────────────────
# Training entry – called from train_alg_hex.py
# ───────────────────────────────────────────────────────────────────────
def train(args):
    # 1. Parse opponent mix ratios (cli has defaults)
    n_envs  = args.num_envs

    if args.opponent_model_path:
        # Self-play mode: all opponents are the loaded PPO model
        print(f"--- Entering SELF-PLAY mode. Opponent: {args.opponent_model_path} ---")
        set_opponent_model_path(args.opponent_model_path)
        opponents = [ppo_opponent_agent] * n_envs
    else:
        # Curriculum mode: mix of rule-based and random opponents
        rule_frac = getattr(args, "rule_fraction", 0.5) # Default to 0.5 if not passed
        n_rules = int(n_envs * rule_frac)
        n_v3 = n_rules // 2
        n_v4 = n_rules - n_v3
        n_random = n_envs - n_rules
        opponents = (
            [rule_based_v3_agent] * n_v3 +
            [rule_based_v4_agent] * n_v4 +
            [random_agent] * n_random
        )
        random.shuffle(opponents)

    # 2. Vectorised training env (every worker already supplies a mask)
    vec_env = SubprocVecEnv(
        [make_train_env(args.board_size, opp, args.wrapper_type) for opp in opponents],
        start_method="spawn",
    )

    # 3. Evaluation env
    video_dir = (
        f"{args.save_dir}/{args.run_name}/video"
        if getattr(args, "video_eval", False)
        else None
    )
    eval_env = make_eval_env(args.board_size, video_dir, video_len=args.video_len)

    # 4. PPO policy & model
    policy_kwargs = dict(
        features_extractor_class=HexCNN,
        features_extractor_kwargs=dict(features_dim=128),
        share_features_extractor=True,
    )
    model = MaskablePPO(
        "CnnPolicy",
        vec_env,
        policy_kwargs=policy_kwargs,
        n_steps=256,                # rollout length (per env)
        batch_size=256,
        n_epochs=4,
        gamma=0.99,
        gae_lambda=0.95,
        ent_coef=0.01,
        vf_coef=0.5,
        learning_rate=2.5e-4,
        max_grad_norm=0.5,
        tensorboard_log=f"{args.save_dir}/{args.run_name}",
        device=args.device,
        verbose=1,
    )

    # 5. Callbacks
    ckpt_cb = CheckpointCallback(
        save_freq=100_000 // n_envs,
        save_path=f"{args.save_dir}/{args.run_name}",
        name_prefix="model",
    )
    eval_cb = FastEval(
        eval_env,
        n_eval_episodes=20,
        eval_freq=max(1, 50_000 // n_envs),
        warmup_steps=10_000,
        deterministic=True,
    )
    strategy_cb = ComprehensiveTBCallback(
        f"{args.save_dir}/{args.run_name}", flush_freq=10_000 // n_envs
    )

    # 6. Train!
    model.learn(
        total_timesteps=args.timesteps,
        callback=[ckpt_cb, eval_cb, strategy_cb],
    )
    model.save(f"{args.save_dir}/{args.run_name}/final")

    vec_env.close()
    eval_env.close()
