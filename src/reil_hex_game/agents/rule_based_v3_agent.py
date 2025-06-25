import random
import heapq
from ..hex_engine.hex_engine import hexPosition
from copy import deepcopy
from .rule_based_helper import (
    STRATEGY_FUNCTIONS as BASE_STRATEGIES,  # ← just an alias
    HEX_NEIGHBORS, is_winning_move, is_forcing_win,
    infer_player, fallback_random,
)

# Work on a *copy* so other agents are unaffected
STRATEGY_FUNCTIONS_ADAPTED = BASE_STRATEGIES.copy()
# NOTE EXAMPLE of adapting a singular strategy for a specific agent at the end of this file.

# def take_center(board, action_set, player):
#     """Rule-4: only grab the centre in the very first two plies."""
#     if sum(cell != 0 for row in board for cell in row) < 2:
#         size   = len(board)
#         centre = (size // 2, size // 2)
#         return centre if centre in action_set else None
#     return None

# # **override just this one entry**
# STRATEGY_FUNCTIONS_ADAPTED["take_center"] = take_center

# Global dictionary to count how often each strategy was chosen
STRATEGY_USE_COUNT = {name: 0 for name in STRATEGY_FUNCTIONS_ADAPTED}

EVAL_CACHE = {}
LAST_PLAYER = None
LAST_MOVE_IS_BRIDGE = False


# NOTE THE NAME of the agent function MUST match the filename!
# Otherwise the agent resolver will not find it.
# When using an agent like 'uv run reil-hex-game --agent rule_based_v4' the 'agent' at the can be omitted
def rule_based_v3_agent(board, action_set, print_debug=False):
    # global is necessary to ACTUALLY mutate the global variables
    global LAST_PLAYER, STRATEGY_USE_COUNT, LAST_MOVE_IS_BRIDGE

    size          = len(board)
    player        = infer_player(board)
    opponent      = -player
    LAST_PLAYER   = player
    

    # ------------------------------------------------------------------ #
    # 0️⃣  Early tactical checks (immediate win / block / forcing win)
    # ------------------------------------------------------------------ #
    for move in action_set:
        if is_winning_move(board, move, player,  EVAL_CACHE):
            print(f"Immediate winning move found: {move}") if print_debug else None
            return move
        if is_winning_move(board, move, opponent, EVAL_CACHE):
            print(f"🛡️  Blocking opponent's immediate win: {move}") if print_debug else None
            return move

    for move in action_set:
        if is_forcing_win(board, move, player, EVAL_CACHE):
            print(f"Forcing 2-step win move found: {move}") if print_debug else None
            return move

    # ------------------------------------------------------------------ #
    # 1️⃣  Stage-dependent weight tweaks
    # ------------------------------------------------------------------ #
    stones       = sum(1 for row in board for cell in row if cell != 0)
    total_cells  = size * size
    first_move   = (sum(1 for row in board for cell in row if cell != 0) <=1)
    early_game   = stones < 0.10 * total_cells
    late_game    = stones > 0.50 * total_cells

    STRATEGY_WEIGHTS = {
        "take_center"                : 1,
        "extend_own_chain"           : 3,
        "break_opponent_bridge"      : 3,
        "protect_own_chain_from_cut" : 3,
        "create_double_threat"       : 4,
        "shortest_connection"        : 4,
        "make_own_bridge"            : 3,
        "mild_block_threat"          : 2,
        "advance_toward_goal"        : 2,
        "block_aligned_opponent_path": 3,
    }
    if first_move:
        STRATEGY_WEIGHTS["take_center"] += 10
    elif early_game:
        STRATEGY_WEIGHTS["make_own_bridge"]           += 3
        STRATEGY_WEIGHTS["advance_toward_goal"]       += 1
    elif late_game:
        STRATEGY_WEIGHTS["shortest_connection"]       += 4
        STRATEGY_WEIGHTS["advance_toward_goal"]       += 5
        STRATEGY_WEIGHTS["block_aligned_opponent_path"]+= 1
    else:
        STRATEGY_WEIGHTS["break_opponent_bridge"]      += 3
        STRATEGY_WEIGHTS["protect_own_chain_from_cut"] += 3
        STRATEGY_WEIGHTS["block_aligned_opponent_path"]+= 4
        STRATEGY_WEIGHTS["mild_block_threat"]          += 10

    if LAST_MOVE_IS_BRIDGE:
        STRATEGY_WEIGHTS["extend_own_chain"] += 5
        LAST_MOVE_IS_BRIDGE = False   

    # ------------------------------------------------------------------ #
    # 2️⃣  Ask **each** strategy once for its favourite move
    # ------------------------------------------------------------------ #
    suggestions = {
        name: func(board, action_set, player)
        for name, func in STRATEGY_FUNCTIONS_ADAPTED.items()
    }

    # ------------------------------------------------------------------ #
    # 3️⃣  Aggregate weighted votes -> total_move_scores
    # ------------------------------------------------------------------ #
    total_move_scores = {}
    for name, move in suggestions.items():
        if move is None:
            continue
        weight = STRATEGY_WEIGHTS[name]
        total_move_scores[move] = total_move_scores.get(move, 0) + weight

    # Debug print — show scores that are >0
    print("\n(Suggested moves) : total score for the move collected over all strategies:") if print_debug else None
    for mv, sc in sorted(total_move_scores.items(), key=lambda x: -x[1]):
        if sc > 0:
            print(f"  {mv}: {sc}") if print_debug else None

    # ------------------------------------------------------------------ #
    # 4️⃣  Choose a move (fallback to random if no strategy suggested a move)
    # ------------------------------------------------------------------ #
    if not total_move_scores:                      # every strategy passed
        chosen_move = fallback_random(action_set)
        max_score   = 0
    else:
        max_score = max(total_move_scores.values())
        best_moves = [m for m, s in total_move_scores.items() if s == max_score]
        chosen_move = random.choice(best_moves)

    # ------------------------------------------------------------------ #
    # 5️⃣  Update per-strategy usage count, list contributors
    # ------------------------------------------------------------------ #
    print(f"\nChosen move: {chosen_move} with score {max_score}") if print_debug else None
    print("Strategies that suggested this move:") if print_debug else None
    for name, move in suggestions.items():
        if move == chosen_move:
            STRATEGY_USE_COUNT[name] += 1
            print(f"  {name}: +{STRATEGY_WEIGHTS[name]}") if print_debug else None
            if name == "make_own_bridge":
                LAST_MOVE_IS_BRIDGE = True
    # ------------------------------------------------------------------ #
    return chosen_move
