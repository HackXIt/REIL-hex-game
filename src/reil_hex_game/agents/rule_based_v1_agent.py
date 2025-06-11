import random

# Neighboring directions on a hex grid (pointy-top orientation)
HEX_NEIGHBORS = [(-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0)]

def rule_based_v1_agent(board, action_set):
    size = len(board)
    player = infer_player(board)

    for tactic in [
        take_center,
        extend_own_chain,
        break_opponent_bridge,
        protect_own_chain_from_cut,
        create_double_threat
    ]:
        move = tactic(board, action_set, player)
        if move:
            return move

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
    # Naive version: checks if all stones are connected (can be improved)
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
    # Tries moves that connect to two of own clusters
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
