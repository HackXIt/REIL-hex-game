import random
import heapq
from ..hex_engine.hex_engine import hexPosition
from copy import deepcopy



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
        if is_winning_move(board, move, player):
            print(f"Immediate winning move found: {move}")
            return move
    
    for move in action_set:
            if is_winning_move(board, move, opponent):
                print(f"🛡️ Blocking opponent's immediate win: {move}")
                return move

    # Check for 2-step forcing win
    for move in action_set:
        if is_forcing_win(board, move, player):
            print(f"Forcing 2-step win move found: {move}")
            return move

   
    

    strategy_functions = [
        take_center,
        extend_own_chain,
        break_opponent_bridge,
        protect_own_chain_from_cut,
        create_double_threat,
        shortest_connection_path,
        favor_bridges,
        mild_block_threat,
        advance_toward_goal,
        block_aligned_opponent_path
    ]

    move_scores = {}
    STRATEGY_WEIGHTS = {
        take_center: 5,
        extend_own_chain: 8,
        break_opponent_bridge: 7,
        protect_own_chain_from_cut: 6,
        create_double_threat: 10,
        shortest_connection_path: 12,
        favor_bridges: 2,
        mild_block_threat: 2,
        advance_toward_goal: 4,
        block_aligned_opponent_path: 2
    }



    if early_game:
        STRATEGY_WEIGHTS[take_center] += 3
        STRATEGY_WEIGHTS[extend_own_chain] += 2
        STRATEGY_WEIGHTS[shortest_connection_path] += 2
    if late_game:
        STRATEGY_WEIGHTS[create_double_threat] += 3
        STRATEGY_WEIGHTS[protect_own_chain_from_cut] += 2
        STRATEGY_WEIGHTS[shortest_connection_path] += 3

    for move in action_set:
        move_scores[move] = 0
        for tactic in strategy_functions:
            suggested = tactic(board, [move], player)
            if suggested == move:
                move_scores[move] += STRATEGY_WEIGHTS[tactic]

    print("\n(Suggested moves) : total score for the move collected over all strategies:")
    for move, score in sorted(move_scores.items(), key=lambda x: -x[1]):
        if score > 0:
            print(f"  {move}: {score}")

    if move_scores:
        max_score = max(move_scores.values())
        best_candidates = [m for m, score in move_scores.items() if score == max_score]
        chosen_move = random.choice(best_candidates)

        # Log which strategies recommended the chosen move
        for tactic in strategy_functions:
            suggested = tactic(board, [chosen_move], player)
            if suggested == chosen_move:
                STRATEGY_USE_COUNT[tactic.__name__] += 1

        return chosen_move

    return fallback_random(action_set)


def infer_player(board):
    flat = [c for row in board for c in row]
    return 1 if flat.count(1) <= flat.count(-1) else -1

def cache_key(board):
    return tuple(tuple(row) for row in board)

def is_winning_move(board, move, player):
    new_board = deepcopy(board)
    new_board[move[0]][move[1]] = player
    key = cache_key(new_board)
    if key in EVAL_CACHE:
        return EVAL_CACHE[key] == player
    sim = hexPosition(size=len(board))
    sim.board = new_board
    if player == 1:
        result = sim._evaluate_white(False)
    else:
        result = sim._evaluate_black(False)
    EVAL_CACHE[key] = player if result else 0
    return result

def is_forcing_win(board, move, player):
    new_board = deepcopy(board)
    new_board[move[0]][move[1]] = player
    key = cache_key(new_board)
    sim = hexPosition(size=len(board))
    sim.board = new_board
    if player == 1 and sim._evaluate_white(False):
        EVAL_CACHE[key] = 1
        return True
    if player == -1 and sim._evaluate_black(False):
        EVAL_CACHE[key] = -1
        return True
    for follow_move in [(i, j) for i in range(len(board)) for j in range(len(board)) if new_board[i][j] == 0]:
        test_board = deepcopy(new_board)
        test_board[follow_move[0]][follow_move[1]] = player
        test_key = cache_key(test_board)
        sim.board = test_board
        if test_key in EVAL_CACHE:
            if EVAL_CACHE[test_key] == player:
                return True
        elif (player == 1 and sim._evaluate_white(False)) or (player == -1 and sim._evaluate_black(False)):
            EVAL_CACHE[test_key] = player
            return True
    return False

def mild_block_threat(board, moves, player):
    opponent = -player
    size = len(board)
    for move in moves:
        new_board = deepcopy(board)
        new_board[move[0]][move[1]] = player
        opponent_threats = [
            (i, j) for i in range(size) for j in range(size)
            if new_board[i][j] == 0 and any(
                0 <= i + dx < size and 0 <= j + dy < size and new_board[i + dx][j + dy] == opponent
                for dx, dy in HEX_NEIGHBORS
            )
        ]
        if len(opponent_threats) > 10:
            return move
    return None

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

def shortest_connection_path(board, action_set, player):
    size = len(board)

    def neighbors(cell):
        i, j = cell
        return [(ni, nj) for ni, nj in get_neighbors(i, j, size) if board[ni][nj] in (0, player)]

    def dijkstra(start_edges, target_edges):
        visited = set()
        heap = [(0, pos) for pos in start_edges if board[pos[0]][pos[1]] in (0, player)]
        heapq.heapify(heap)

        while heap:
            dist, (i, j) = heapq.heappop(heap)
            if (i, j) in visited:
                continue
            visited.add((i, j))
            if (i, j) in target_edges:
                return (i, j)
            for ni, nj in neighbors((i, j)):
                if (ni, nj) not in visited:
                    cost = 0 if board[ni][nj] == player else 1
                    heapq.heappush(heap, (dist + cost, (ni, nj)))
        return None

    if player == -1:  # Black plays left <-> right
        starts = [(i, 0) for i in range(size)]
        goals = [(i, size - 1) for i in range(size)]
    else:  # White plays top <-> bottom
        starts = [(0, j) for j in range(size)]
        goals = [(size - 1, j) for j in range(size)]

    best_move = dijkstra(starts, goals)
    return best_move if best_move in action_set else None

    def dijkstra(start_edges, target_edges):
        visited = set()
        heap = [(0, pos) for pos in start_edges if board[pos[0]][pos[1]] in (0, player)]
        heapq.heapify(heap)

        while heap:
            dist, (i, j) = heapq.heappop(heap)
            if (i, j) in visited:
                continue
            visited.add((i, j))
            if (i, j) in target_edges:
                return (i, j)
            for ni, nj in neighbors((i, j)):
                if (ni, nj) not in visited:
                    cost = 0 if board[ni][nj] == player else 1
                    heapq.heappush(heap, (dist + cost, (ni, nj)))
        return None

    if player == 1:
        starts = [(i, 0) for i in range(size)]
        goals = [(i, size - 1) for i in range(size)]
    else:
        starts = [(0, j) for j in range(size)]
        goals = [(size - 1, j) for j in range(size)]

    best_move = dijkstra(starts, goals)
    return best_move if best_move in action_set else None



def favor_bridges(board, moves, player):
    size = len(board)
    own_stones = [(i, j) for i in range(size) for j in range(size) if board[i][j] == player]
    best_move = None

    for move in moves:
        mx, my = move
        for (sx, sy) in own_stones:
            dx = mx - sx
            dy = my - sy
            distance = abs(dx) + abs(dy)
            if distance in (2, 3):  # potential bridge
                # Respect connection direction: vertical for White (player 1), horizontal for Black (player -1)
                if (player == 1 and abs(dx) > abs(dy)) or (player == -1 and abs(dy) > abs(dx)):
                    return move
                if best_move is None:
                    best_move = move
    return best_move

    for move in moves:
        mx, my = move
        for (sx, sy) in own_stones:
            dx = mx - sx
            dy = my - sy
            distance = abs(dx) + abs(dy)
            if distance in (2, 3):  # bridge-like structure
                if player == 1 and dx > 0:  # white moves downward
                    return move
                if player == -1 and dy > 0:  # black moves right
                    return move
                if best_move is None:  # fallback if no forward bridge found
                    best_move = move

    return best_move

def advance_toward_goal(board, moves, player):
    size = len(board)
    own_stones = [(i, j) for i in range(size) for j in range(size) if board[i][j] == player]
    if not own_stones:
        return None

    if player == 1:
        # White goes top to bottom -> favor increasing row (i)
        avg = sum(i for i, _ in own_stones) / len(own_stones)
        return max(moves, key=lambda m: m[0] - avg)
    else:
        # Black goes left to right -> favor increasing column (j)
        avg = sum(j for _, j in own_stones) / len(own_stones)
        return max(moves, key=lambda m: m[1] - avg)

    if player == 1:
        avg = sum(i for i, _ in own_stones) / len(own_stones)
        return max(moves, key=lambda m: m[0] - avg)
    else:
        avg = sum(j for _, j in own_stones) / len(own_stones)
        return max(moves, key=lambda m: m[1] - avg)

def block_aligned_opponent_path(board, moves, player):
    size = len(board)
    opponent = -player
    opponent_stones = [(i, j) for i in range(size) for j in range(size) if board[i][j] == opponent]

    if not opponent_stones:
        return None

    # Track alignment frequency along the opponent's win direction
    axis_key = (lambda x, y: y) if opponent == -1 else (lambda x, y: x)  # Black: horizontal, White: vertical

    aligned = {}
    for x, y in opponent_stones:
        key = axis_key(x, y)
        aligned[key] = aligned.get(key, 0) + 1

    if not aligned:
        return None

    target = max(aligned, key=aligned.get)
    for move in moves:
        if (opponent == -1 and move[1] == target) or (opponent == 1 and move[0] == target):
            return move
    return None

    if not opponent_stones:
        return None

    aligned = {}
    for x, y in opponent_stones:
        key = y if direction_axis == 1 else x
        aligned[key] = aligned.get(key, 0) + 1

    target = max(aligned, key=aligned.get)
    for move in moves:
        if (direction_axis == 1 and move[1] == target) or (direction_axis == 0 and move[0] == target):
            return move
    return None

def fallback_random(action_set):
    return random.choice(action_set)

# Optional: function to display strategy usage summary at the end of the game
def print_strategy_summary():
    print("\n Strategy usage summary:")
    for strategy, count in STRATEGY_USE_COUNT.items():
        print(f"  {strategy}: {count} times")

def announce_agent_color(board):
    player = infer_player(board)
    color = "White (○)" if player == 1 else "Black (●)"
    print(f"\n Your agent is playing as: {color}")

