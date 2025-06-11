# TODO Please change and adapt this code structure according to rule_based_agent_4.py

import random

# Neighboring directions on a hex grid (pointy-top orientation)
HEX_NEIGHBORS = [(-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0)]

# Global dictionary to count how often each strategy was chosen
STRATEGY_USE_COUNT = {
    "take_center": 0,
    "extend_own_chain": 0,
    "break_opponent_bridge": 0,
    "protect_own_chain_from_cut": 0,
    "create_double_threat": 0
}

def rule_based_v2_agent(board, action_set):
    size = len(board)
    player = infer_player(board)

    strategy_functions = [
        take_center,
        extend_own_chain,
        break_opponent_bridge,
        protect_own_chain_from_cut,
        create_double_threat
    ]

    move_scores = {}
    STRATEGY_WEIGHTS = {
        take_center: 5,
        extend_own_chain: 8,
        break_opponent_bridge: 7,
        protect_own_chain_from_cut: 6,
        create_double_threat: 10
    }

    stones = sum(1 for row in board for cell in row if cell != 0)
    total_cells = size * size
    early_game = stones < 0.3 * total_cells
    late_game = stones > 0.7 * total_cells

    if early_game:
        STRATEGY_WEIGHTS[take_center] += 3
        STRATEGY_WEIGHTS[extend_own_chain] += 2
    if late_game:
        STRATEGY_WEIGHTS[create_double_threat] += 3
        STRATEGY_WEIGHTS[protect_own_chain_from_cut] += 2

    for tactic in strategy_functions:
        move = tactic(board, action_set, player)
        if move:
            move_scores[move] = move_scores.get(move, 0) + STRATEGY_WEIGHTS[tactic]

    print("\n (Suggested moves) : total score for the move collected over all strategies ")
    for move, score in sorted(move_scores.items(), key=lambda x: -x[1]):
        print(f"  {move}: {score}")

    if move_scores:
        max_score = max(move_scores.values())
        best_candidates = [m for m, score in move_scores.items() if score == max_score]
        chosen_move = random.choice(best_candidates)

        # Log which strategy made the final chosen move
        for tactic in strategy_functions:
            if tactic(board, action_set, player) == chosen_move:
                STRATEGY_USE_COUNT[tactic.__name__] += 1
                break

        return chosen_move

    return fallback_random(action_set)



def infer_player(board):
    flat = [c for row in board for c in row]
    return 1 if flat.count(1) <= flat.count(-1) else -1

def take_center(board, action_set, player):
    size = len(board)
    center = (size // 2, size // 2)
    return center if center in action_set else None

def get_neighbors(i, j, size):
    return [
        (i+di, j+dj)
        for di, dj in HEX_NEIGHBORS
        if 0 <= i+di < size and 0 <= j+dj < size
    ]

def extend_own_chain(board, action_set, player):
    size = len(board)
    my_positions = [(i, j) for i in range(size) for j in range(size) if board[i][j] == player]
    random.shuffle(my_positions)
    for i, j in my_positions:
        for ni, nj in get_neighbors(i, j, size):
            if (ni, nj) in action_set:
                return (ni, nj)
    return None

def break_opponent_bridge(board, action_set, player):
    size = len(board)
    enemy = -player
    for i in range(size):
        for j in range(size):
            if board[i][j] != enemy:
                continue
            for dx, dy in [(-1, 1), (1, -1), (-1, -1), (1, 1)]:
                ni, nj = i + dx, j + dy
                mi, mj = i + dx//2, j + dy//2
                if (0 <= ni < size and 0 <= nj < size and
                    board[ni][nj] == enemy and
                    (mi, mj) in action_set and board[mi][mj] == 0):
                    return (mi, mj)
    return None

def protect_own_chain_from_cut(board, action_set, player):
    size = len(board)
    for i, j in action_set:
        board[i][j] = player
        if is_cut_point(board, player):
            board[i][j] = 0
            continue
        board[i][j] = 0
        return (i, j)
    return None

def is_cut_point(board, player):
    size = len(board)
    visited = set()
    starts = [(i, j) for i in range(size) for j in range(size) if board[i][j] == player]
    if not starts:
        return False
    def dfs(i, j):
        stack = [(i, j)]
        while stack:
            ci, cj = stack.pop()
            visited.add((ci, cj))
            for ni, nj in get_neighbors(ci, cj, size):
                if board[ni][nj] == player and (ni, nj) not in visited:
                    stack.append((ni, nj))
    dfs(starts[0][0], starts[0][1])
    return len(visited) < len(starts)

def create_double_threat(board, action_set, player):
    size = len(board)
    clusters = []
    visited = set()
    for i in range(size):
        for j in range(size):
            if board[i][j] == player and (i, j) not in visited:
                cluster = []
                stack = [(i, j)]
                while stack:
                    ci, cj = stack.pop()
                    if (ci, cj) in visited:
                        continue
                    visited.add((ci, cj))
                    cluster.append((ci, cj))
                    for ni, nj in get_neighbors(ci, cj, size):
                        if board[ni][nj] == player:
                            stack.append((ni, nj))
                clusters.append(cluster)
    for i, j in action_set:
        count = 0
        for cluster in clusters:
            if any((ni, nj) in cluster for ni, nj in get_neighbors(i, j, size)):
                count += 1
        if count >= 2:
            return (i, j)
    return None

def fallback_random(action_set):
    return random.choice(action_set)

def print_strategy_summary():
    print("\n Strategy usage summary:")
    for strategy, count in STRATEGY_USE_COUNT.items():
        print(f"  {strategy}: {count} times")
