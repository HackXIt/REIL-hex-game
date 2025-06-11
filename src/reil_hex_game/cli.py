"""reil_hex_game.cli
====================
CLI for starting a Hex game shipped with *reil_hex_game*.

Key improvements in this revision
---------------------------------
* **Built-in agents by name** - pass, e.g. ``--agent rule_based`` to use the
  built-in rule-based bot located at
  ``reil_hex_game.agents.rule_based_agent:rule_based_agent``.
* The *interactive wizard* suggests **rule_based** as the default agent when a
  bot is required.
* Accepts fully-qualified *module:attr* paths for custom agents.
"""
from __future__ import annotations

import argparse
import importlib
import sys
from types import ModuleType
from typing import Callable, Optional, Sequence, Union, List

from .hex_engine import hexPosition

AgentCallable = Callable[["hexPosition"], int]  # expected signature

# ---------------------------------------------------------------------------
# Agent resolution helpers
# ---------------------------------------------------------------------------

def _import_module(module_path: str) -> ModuleType:
    try:
        return importlib.import_module(module_path)
    except ModuleNotFoundError as err:
        raise ValueError(f"Cannot import module {module_path!r}.") from err


def _load_attr(module: ModuleType, *candidate_names: str) -> AgentCallable:
    """Return the first attribute in *candidate_names* that is callable."""
    for name in candidate_names:
        attr = getattr(module, name, None)
        if callable(attr):
            return attr  # type: ignore[return-value]
    raise ValueError(
        f"None of the symbols {candidate_names!r} are callables in {module.__name__}."
    )


def _resolve_agent(agent: str) -> AgentCallable:
    """Return a callable for *agent*.

    * If *agent* contains a colon (``pkg.mod:attr``) **or** a dot outside a
      trailing ".py", it is treated as a fully-qualified path.
    * Otherwise, it is interpreted as the *name* of a built-in agent residing
      below :pymod:`reil_hex_game.agents`.
    """
    # ------------------------------------------------------------------
    # Special-case built-in random agent
    # ------------------------------------------------------------------
    if agent.lower() in {"random", "rnd"}:
        from random import choice
        return lambda _board, action_set: choice(action_set)
    # ------------------------------------------------------------------

    # Path-like: explicit attr specification
    if ":" in agent or "." in agent:
        try:
            module_path, attr_name = agent.split(":", 1)
        except ValueError as err:
            raise ValueError(
                "Agent path must be of the form 'module.sub:attr'."
            ) from err
        module = _import_module(module_path)
        return _load_attr(module, attr_name)

    # Built-in short name
    variations: Sequence[str] = (
        f"{agent}",
        f"{agent}_agent",
    )
    last_error: Exception | None = None
    for var in variations:
        try:
            module = _import_module(f"reil_hex_game.agents.{var}")
            # prioritise a generic symbol 'agent', else the module's basename
            func = _load_attr(module, "agent", var, f"{var}_agent")
            return func
        except Exception as err: # noqa: BLE001 - catch all for import errors
            last_error = err
            continue
    # If we get here all attempts failed
    raise ValueError(
        f"Could not resolve built-in agent {agent!r}: {last_error}")

# ---------------------------------------------------------------------------
# Interactive helpers
# ---------------------------------------------------------------------------

def _prompt_choice(prompt: str, choices: list[str], default: Optional[str] = None) -> str:
    choice_str = "/".join(choices)
    while True:
        inp = input(f"{prompt} [{choice_str}] ").strip().lower()
        if not inp and default is not None:
            return default
        if inp in choices:
            return inp
        print(f"Please type one of: {choice_str}\n")


def _prompt_yes_no(prompt: str, default: bool = False) -> bool:
    default_str = "Y/n" if default else "y/N"
    while True:
        inp = input(f"{prompt} [{default_str}] ").strip().lower()
        if not inp:
            return default
        if inp in ("y", "yes"):
            return True
        if inp in ("n", "no"):
            return False
        print("Please answer with 'y' or 'n'.\n")


def _interactive_wizard(args: argparse.Namespace) -> argparse.Namespace:
    """Fill unspecified *args* fields by prompting the user."""
    if args.mode is None:
        args.mode = _prompt_choice(
            "Choose game mode",
            ["hvh", "hvm", "mvm"],
            default="hvh",
        )

    if args.board_size is None:
        while True:
            try:
                size_str = input("Board size (2-26) [7] ").strip()
                args.board_size = int(size_str) if size_str else 7
                if 2 <= args.board_size <= 26:
                    break
            except ValueError:
                pass
            print("Please enter an integer between 2 and 26.\n")

    if args.use_pygame is None:
        args.use_pygame = _prompt_yes_no("Enable pygame GUI?", default=False)

    if args.mode == "hvm":
        if args.human_player is None:
            ans = _prompt_choice("Human plays as", ["1", "2"], default="1")
            args.human_player = int(ans)
    else:
        args.human_player = None

    if args.mode == "mvm":
        ans = _prompt_yes_no(
            "Enable auto-play mode (machine vs machine)?", default=True)
        args.auto = ans
        if ans:
            rate_str = input("Auto-play speed in moves/second [3.0] ").strip()
            args.rate = float(rate_str) if rate_str else 3.0
        else:
            args.rate = None

    if args.mode in ("hvm", "mvm") and args.agent is None:
        ans = input(
            "Built-in agent name or 'module:attr' [rule_based] "
        ).strip()
        args.agent = ans or "rule_based"

    print()  # spacing before game starts
    return args

# ---------------------------------------------------------------------------
# Helpers for multiple‑agent handling
# ---------------------------------------------------------------------------

def _normalise_agent_list(agent_opt: Union[None, str, List[str]]) -> List[str]:
    """Return a list with exactly two entries (duplicates if necessary)."""
    if agent_opt is None:
        return ["rule_based", "rule_based"]

    if isinstance(agent_opt, str):
        return [agent_opt, agent_opt]

    if isinstance(agent_opt, list):
        if len(agent_opt) == 0:
            return ["rule_based", "rule_based"]
        if len(agent_opt) == 1:
            return [agent_opt[0], agent_opt[0]]
        if len(agent_opt) == 2:
            return agent_opt  # type: ignore[return-value]
        raise ValueError("--agent accepts at most two values.")

    raise TypeError("Invalid --agent argument type.")

# ---------------------------------------------------------------------------
# Game runner
# ---------------------------------------------------------------------------

def _run_game(args: argparse.Namespace) -> None:
    game = hexPosition(
        size=args.board_size,
        use_pygame=args.use_pygame,
    )

    # Normalise agent option into a two‑element list
    agent_names = _normalise_agent_list(args.agent)

    if args.mode == "hvh":
        try:
            game.human_vs_human()  # type: ignore[attr‑defined]
        except AttributeError:
            raise SystemExit(
                "hexPosition currently lacks 'human_vs_human' support.")

    elif args.mode == "hvm":
        # Use only the *first* agent when two are supplied
        agent_callable = _resolve_agent(agent_names[0])
        human_player_flag = 1 if (args.human_player or 1) == 1 else -1
        try:
            game.human_vs_machine(human_player=human_player_flag,
                                   machine=agent_callable)
        except KeyboardInterrupt:
            print("\nGame interrupted by user.")
        finally:
            game.close()

    elif args.mode == "mvm":
        agent1_callable = _resolve_agent(agent_names[0])
        agent2_callable = _resolve_agent(agent_names[1])
        try:
            game.machine_vs_machine(machine1=agent1_callable,
                                     machine2=agent2_callable,
                                     auto=args.auto,
                                     rate=args.rate)
        except KeyboardInterrupt:
            print("\nGame interrupted by user.")
        finally:
            game.close()
    else:
        raise ValueError(f"Unsupported mode {args.mode!r}.")

# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def main(argv: Optional[list[str]] = None) -> None:  # noqa: D401 - simple name
    argv = sys.argv[1:] if argv is None else argv

    parser = argparse.ArgumentParser(
        prog="reil-hex-game",
        description="Play the Hex board game from the command line.",
    )

    parser.add_argument("--mode", choices=["hvh", "hvm", "mvm"],
                        help="Game mode: hvh (human vs human), hvm (human vs machine), mvm (machine vs machine)")
    parser.add_argument(
        "--auto", action="store_true",
        help="Start machine-vs-machine in continuous auto-play."
    )
    parser.add_argument(
        "--rate", type=float, default=3.0,
        help="Auto-play speed in moves / second (default 3)."
    )
    parser.add_argument("--board-size", type=int,
                        help="Board side length (2-26, default 7)")
    parser.add_argument("--use-pygame", action="store_true",
                        help="Enable pygame GUI")
    parser.add_argument(
        "--agent", nargs="+",
        help=(
            "Built-in agent name or module:attr path. "
            "Accepts one or two values - when two are given they are used as "
            "agent 1 and agent 2. In hvm mode the second value is ignored."
        )
    )
    parser.add_argument("--human-player", type=int, choices=[1, 2],
                        help="For hvm mode: 1=white, 2=black (default 1)")
    parser.add_argument("-i", "--interactive", action="store_true",
                        help="Prompt for options interactively (default when no flags are given)")

    args = parser.parse_args(argv)

    # Normalise unspecified flags to None so the wizard can override
    if not args.use_pygame:
        args.use_pygame = None

    need_interactive = args.interactive or len(argv) == 0

    if need_interactive:
        args = _interactive_wizard(args)
    else:
        if args.board_size is None:
            args.board_size = 7
        if args.mode is None:
            parser.error("--mode is required when not using interactive mode")
        if args.mode in ("hvm", "mvm") and args.agent is None:
            args.agent = "rule_based"
        if args.mode == "hvm" and args.human_player is None:
            args.human_player = 1
    _run_game(args)

# Allow "python -m reil_hex_game.cli" direct execution
if __name__ == "__main__":  # pragma: no cover
    main()
