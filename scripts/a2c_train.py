# scripts/a2c_train.py
from __future__ import annotations
import multiprocessing as mp
import random
from collections import Counter, defaultdict
import warnings
import numpy as np

# ── mp start-method for Windows / CUDA ─────────────────────────────────
if mp.get_start_method(allow_none=True) != "spawn":
    mp.set_start_method("spawn", force=True)

# ── third-party --------------------------------------------------------
from sb3_contrib.common.wrappers import ActionMasker
from sb3_contrib.common.maskable.utils import get_action_masks
from stable_baselines3 import A2C
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecVideoRecorder
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback, BaseCallback
from gymnasium.wrappers import RecordVideo
from torch.utils.tensorboard import SummaryWriter

# ── project -----------------------------------------------------------
from reil_hex_game.hex_engine.hex_env import HexEnv, OpponentWrapper  # Opponent wrapper
from reil_hex_game.agents.hex_cnn import HexCNN
from reil_hex_game.agents.rule_based_v3_agent import rule_based_v3_agent
from reil_hex_game.agents.rule_based_v4_agent import rule_based_v4_agent

warnings.filterwarnings("ignore", module="pygame.pkgdata")   # silence setuptools 81 spam


# ────────────────────────────────────────────────────────────
# Utility: random opponent
# ────────────────────────────────────────────────────────────
def random_agent(board, legal_moves):
    """Return a uniformly-random legal coordinate."""
    return random.choice(legal_moves)


def _masked_predict(model, obs, env, deterministic=True):
    """
    SB3 `predict()` that never returns an illegal action.
    Works with any vanilla SB3 algorithm.
    """
    masks = get_action_masks(env)                 # bool (n_envs, A)
    act, _ = model.predict(obs, deterministic=deterministic)

    # If the chosen index is illegal, fall back to *first* legal one
    illegal = ~masks[np.arange(len(act)), act]
    if illegal.any():
        fallback = masks.argmax(1)
        act = np.where(illegal, fallback, act)
    return act


# ────────────────────────────────────────────────────────────
# TensorBoard callback for agent/opponent strategies
# ────────────────────────────────────────────────────────────
class StrategyTBCallback(BaseCallback):
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


class FastEval(EvalCallback):
    """Skip the t=0 evaluation and use far fewer episodes in the early game."""
    def __init__(self, *a, warmup_steps: int = 10_000, **kw):
        super().__init__(*a, **kw)
        self.warmup_steps = warmup_steps

    def _on_training_start(self) -> None:          # suppress first eval
        self.last_eval_step = -self.warmup_steps

    def _run_eval(self):
        from collections import deque
        returns, lengths = deque(), deque()

        for _ in range(self.n_eval_episodes):
            obs, _ = self.eval_env.reset()
            done = [False]
            ep_ret, ep_len = 0.0, 0
            while not done[0]:
                action = _masked_predict(self.model, obs, self.eval_env,
                                         deterministic=self.deterministic)
                obs, r, term, trunc, _ = self.eval_env.step(action)
                done = np.logical_or(term, trunc)
                ep_ret += r[0]; ep_len += 1
            returns.append(ep_ret); lengths.append(ep_len)

        self.last_mean_reward = float(np.mean(returns))
        self.last_mean_length = int(np.mean(lengths))
        self._on_event()

# ────────────────────────────────────────────────────────────
# Env factories
# ────────────────────────────────────────────────────────────

# --------------------------------------------------------------------------- #
# helper: illegal-action mask
# --------------------------------------------------------------------------- #
def mask_fn(env: HexEnv):
    """Return Boolean vector - True where the move is legal."""
    mask = env.action_masks()
    assert mask.any(), "mask is all-False - something is wrong in HexEnv"
    return mask

# --------------------------------------------------------------------------- #
# helper: training-env factory
# --------------------------------------------------------------------------- #
def make_train_env(board_size: int, opponent_fn, render_mode: str | None = None):
    """Return a callable that builds ONE env with the chosen opponent."""
    def _factory():
        env = HexEnv(board_size)
        env = OpponentWrapper(env, opponent_fn)
        env = ActionMasker(env, mask_fn)
        return env
    return _factory


def make_eval_env(board_size: int, video_folder: str | None, video_length: int = 300):
    """1-worker DummyVecEnv so no pygame Surface is pickled."""
    def _factory():
        env = HexEnv(board_size, render_mode="rgb_array")
        env = OpponentWrapper(env, rule_based_v4_agent)        # evaluate vs v4
        env = ActionMasker(env, mask_fn)
        return Monitor(env)
    venv = DummyVecEnv([_factory])
    if video_folder:
        venv = VecVideoRecorder(
            venv,
            video_folder=video_folder,
            record_video_trigger=lambda step: True,   # every episode
            video_length=video_length,                        # ≈ 10 s at 30 fps
            name_prefix="eval",
        )
    return venv


# ────────────────────────────────────────────────────────────
# Training entry - called from train_alg_hex.py
# ────────────────────────────────────────────────────────────
def train(args):

    # ------- read split ratios (defaults if CLI didn’t add them) -------
    v3_frac = getattr(args, "v3_frac", 0.25)
    v4_frac = getattr(args, "v4_frac", 0.25)
    assert 0.0 <= v3_frac <= 1.0 and 0.0 <= v4_frac <= 1.0 and v3_frac + v4_frac <= 1.0, \
        "Fractions must sum to ≤ 1."

    n_envs   = args.num_envs
    n_v3     = int(n_envs * v3_frac)
    n_v4     = int(n_envs * v4_frac)
    n_random = n_envs - n_v3 - n_v4   # remainder → random

    opponent_list = (
        [rule_based_v3_agent] * n_v3 +
        [rule_based_v4_agent] * n_v4 +
        [random_agent]        * n_random
    )
    random.shuffle(opponent_list)     # mix order for better batching

    # 1. Vectorised training envs (all head-less)
    vec_env = SubprocVecEnv(
        [make_train_env(args.board_size, opp_fn) for opp_fn in opponent_list],
        start_method="spawn",  # required for CUDA on Windows
    )

    # 2. Evaluation env, with optional video
    video_dir = f"{args.save_dir}/{args.run_name}/video" if getattr(args, "video_eval", False) else None
    eval_env  = make_eval_env(args.board_size, video_dir, video_length=args.video_len)

    # 3. Policy & model
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
        save_freq=100_000 // n_envs,
        save_path=log_dir,
        name_prefix="model",
    )
    # eval_cb = EvalCallback(
    #     eval_env,
    #     n_eval_episodes=30,
    #     eval_freq=50_000 // n_envs,
    #     deterministic=True,
    #     render=False,
    #     log_path=log_dir,
    # )
    eval_cb = FastEval(
        eval_env,
        n_eval_episodes = 10,          # 30 → 10
        eval_freq       = 25_000 // n_envs,   # evaluate half as often
        warmup_steps    = 10_000,      # wait a bit
        deterministic   = True,
    )
    strategy_cb = StrategyTBCallback(log_dir, flush_freq=10_000 // n_envs)

    # 5. Learn!
    model.learn(
        total_timesteps=args.timesteps,
        callback=[ckpt_cb, eval_cb, strategy_cb],
    )
    
    model.save(f"{args.save_dir}/{args.run_name}/final")

    vec_env.close()
    eval_env.close()
