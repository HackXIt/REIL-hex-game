import random
import heapq
from ..hex_engine.hex_engine import hexPosition
from copy import deepcopy
from .rule_based_helper import STRATEGY_FUNCTIONS, is_winning_move, is_forcing_win, infer_player

STRATEGY_FUNCTIONS_ADAPTED = STRATEGY_FUNCTIONS.copy()

# Neighboring directions on a hex grid (pointy-top orientation)
HEX_NEIGHBORS = [(-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0)]

# Global dictionary to count how often each strategy was chosen
STRATEGY_USE_COUNT = {
    "take_center": 0,
    "extend_own_chain": 0,
    "break_opponent_bridge": 0,
    "protect_own_chain_from_cut": 0,
    "create_double_threat": 0,
    "shortest_connection_path": 0,
    "favor_bridges": 0,
    "mild_block_threat": 0,
    "advance_toward_goal": 0,
    "block_aligned_opponent_path": 0
}

EVAL_CACHE = {}
LAST_PLAYER = None

def rule_based_agent_4(board, action_set):
    size = len(board)
    player = infer_player(board)
    opponent = -player
    LAST_PLAYER = player

    stones = sum(1 for row in board for cell in row if cell != 0)
    total_cells = size * size
    early_game = stones < 0.3 * total_cells
    late_game = stones > 0.7 * total_cells


    # Check for immediate winning move
    for move in action_set:
        if is_winning_move(board, move, player, EVAL_CACHE=EVAL_CACHE):
            print(f"Immediate winning move found: {move}")
            return move
    
    for move in action_set:
            if is_winning_move(board, move, opponent, EVAL_CACHE=EVAL_CACHE):
                print(f"🛡️ Blocking opponent's immediate win: {move}")
                return move

    # Check for 2-step forcing win
    for move in action_set:
        if is_forcing_win(board, move, player, EVAL_CACHE=EVAL_CACHE):
            print(f"Forcing 2-step win move found: {move}")
            return move

    strategy_functions = STRATEGY_FUNCTIONS_ADAPTED

    move_scores = {
        "take_center": {},
        "extend_own_chain": {},
        "break_opponent_bridge": {},
        "protect_own_chain_from_cut": {},
        "create_double_threat": {},
        "shortest_connection_path": {},
        "favor_bridges": {},
        "mild_block_threat": {},
        "advance_toward_goal": {},
        "block_aligned_opponent_path":{}
    }
    STRATEGY_WEIGHTS = {
        "take_center": 5,
        "extend_own_chain": 8,
        "break_opponent_bridge": 7,
        "protect_own_chain_from_cut": 6,
        "create_double_threat": 10,
        "shortest_connection_path": 12,
        "favor_bridges": 2,
        "mild_block_threat": 2,
        "advance_toward_goal": 4,
        "block_aligned_opponent_path": 2
    }

    if early_game:
        STRATEGY_WEIGHTS["take_center"] += 3
        STRATEGY_WEIGHTS["extend_own_chain"] += 2
        STRATEGY_WEIGHTS["shortest_connection_path"] += 2
    if late_game:
        STRATEGY_WEIGHTS["create_double_threat"] += 3
        STRATEGY_WEIGHTS["protect_own_chain_from_cut"] += 2
        STRATEGY_WEIGHTS["shortest_connection_path"] += 3

    for move in action_set:
        for tactic in strategy_functions:
            move_scores[tactic][move] = 0
            suggested = STRATEGY_FUNCTIONS_ADAPTED[tactic](board, [move], player)
            if suggested == move:
                move_scores[tactic][move] += STRATEGY_WEIGHTS[tactic]

    # Aggregate total scores for each move across all strategies
    total_move_scores = {}
    for tactic_scores in move_scores.values():
        for move, score in tactic_scores.items():
            total_move_scores[move] = total_move_scores.get(move, 0) + score

    print("\n(Suggested moves) : total score for the move collected over all strategies:")
    for move, score in sorted(total_move_scores.items(), key=lambda x: -x[1]):
        if score > 0:
            print(f"  {move}: {score}")

    if move_scores:
        max_score = max(total_move_scores.values())
        best_candidates = [m for m, score in total_move_scores.items() if score == max_score]
        chosen_move = random.choice(best_candidates)

        # Log which strategies recommended the chosen move
        print(f"\nChosen move: {chosen_move} with score {max_score}")
        print("Strategies that suggested this move:")
        for tactic in STRATEGY_USE_COUNT:
            if move_scores[tactic].get(chosen_move, 0) > 0:
                STRATEGY_USE_COUNT[tactic] += 1
                print(f"  {tactic}: {move_scores[tactic].get(chosen_move, 0)}")

        return chosen_move
    return fallback_random(action_set)

def fallback_random(action_set):
    return random.choice(action_set)

# def take_center(board, action_set, player):
#     size = len(board)
#     center = (size // 2, size // 2)
#     return center if center in action_set else None

# STRATEGY_FUNCTIONS_ADAPTED["take_center"] = take_center