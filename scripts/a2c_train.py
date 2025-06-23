from __future__ import annotations
import multiprocessing as mp
if mp.get_start_method(allow_none=True) != "spawn":
    mp.set_start_method("spawn", force=True)

import pathlib, torch as th
from stable_baselines3 import A2C
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from reil_hex_game.hex_engine.hex_env import HexEnv, OpponentWrapper
from reil_hex_game.agents.hex_cnn import HexCNN
from reil_hex_game.agents.rule_based_v4_agent import rule_based_v4_agent

def make_env(size: int, opponent_fn=None):
    """Factory so each worker gets its own environment instance."""
    def _f():
        env = HexEnv(size=size)
        if opponent_fn:                       # optional opponent swap
            env = OpponentWrapper(env, opponent_fn)
        return env
    return _f

def train(args):
    vec_env = SubprocVecEnv([
        make_env(args.board_size, rule_based_v4_agent if i % 2 == 0 else None)
        for i in range(args.num_envs)
    ])

    policy_kwargs = dict(
        features_extractor_class=HexCNN,
        features_extractor_kwargs=dict(features_dim=128),
        share_features_extractor=True,
    )

    model = A2C(
        "CnnPolicy",
        vec_env,
        policy_kwargs=policy_kwargs,
        n_steps=128,                 # 128 × 16 envs = 2 048 batch
        gae_lambda=0.95,
        gamma=0.99,
        ent_coef=0.01,
        vf_coef=0.5,
        learning_rate=7e-4,
        tensorboard_log=f"{args.save_dir}/{args.run_name}",
        device=args.device,
        verbose=1,
    )

    ckpt_cb = CheckpointCallback(
        save_freq=100_000 // args.num_envs,
        save_path=f"{args.save_dir}/{args.run_name}",
        name_prefix="model"
    )

    # Optional: evaluate on the rule-based agent every X steps
    eval_env = HexEnv(size=args.board_size)
    eval_cb = EvalCallback(
        eval_env,
        n_eval_episodes=30,
        eval_freq=50_000 // args.num_envs,
        deterministic=True,
        render=False,
    )

    model.learn(total_timesteps=args.timesteps,
                callback=[ckpt_cb, eval_cb])
    model.save(f"{args.save_dir}/{args.run_name}/final")