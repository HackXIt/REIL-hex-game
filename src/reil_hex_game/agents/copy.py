import random, heapq, atexit, inspect, sys
import math
from ..hex_engine.hex_engine import hexPosition
from copy import deepcopy
from typing import List, Tuple, Dict, Callable, Set, Optional
from collections import deque
import heapq

# -------------------------------------------------------------------------------
# CONSTANTS
# -------------------------------------------------------------------------------
Coordinate = Tuple[int, int]
# Neighboring directions on a hex grid (pointy-top orientation)
HEX_NEIGHBORS = [(-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0)]
ADJACENCY_MAPS: Dict[int, Dict[Coordinate, List[Coordinate]]] = {}

# -------------------------------------------------------------------------------
# GENERAL HELPER FUNCTIONS FOR RULE BASED AGENT
# -------------------------------------------------------------------------------
def get_adjacency_map(size: int) -> Dict[Coordinate, List[Coordinate]]:
    if size in ADJACENCY_MAPS:
        return ADJACENCY_MAPS[size]

    adjacency = {}
    for x in range(size):
        for y in range(size):
            candidates = [
                (x-1, y),     # up
                (x+1, y),     # down
                (x, y-1),     # left
                (x, y+1),     # right
                (x-1, y+1),   # up-right
                (x+1, y-1)    # down-left
            ]
            adjacency[(x, y)] = [
                (i, j) for i, j in candidates
                if 0 <= i < size and 0 <= j < size
            ]
    ADJACENCY_MAPS[size] = adjacency
    return adjacency


def cache_key(board):
    """
    Generates a hashable cache key from a 2D board representation.

    Args:
        board (list[list[Any]]): The game board represented as a list of lists.

    Returns:
        tuple: A tuple of tuples representing the board, suitable for use as a dictionary key or for caching.

    Example:
        >>> cache_key([[0, 1], [1, 0]])
        ((0, 1), (1, 0))
    """
    return tuple(tuple(row) for row in board)


def infer_player(board):
    """
    Infers the current player's turn based on the state of the board.

    The function assumes that the board is represented as a 2D list, where each cell contains:
        1  - indicating player 1's move,
       -1  - indicating player -1's move,
        0  - indicating an empty cell.

    Returns:
        int: 1 if it is player 1's turn, -1 if it is player -1's turn.

    Logic:
        - Counts the number of moves made by each player.
        - If player 1 has made fewer or equal moves compared to player -1, it is player 1's turn.
        - Otherwise, it is player -1's turn.
    """
    flat = [c for row in board for c in row]
    return 1 if flat.count(1) <= flat.count(-1) else -1

def announce_agent_color(board):
    """
    Announces the color (White or Black) that the agent is currently playing as, 
    based on the current board state.

    Args:
        board: The current state of the game board, used to infer which player 
               the agent is.

    Side Effects:
        Prints a message to the console indicating the agent's color.
    """
    """Announce the color of the agent based on the current board state."""
    player = infer_player(board)
    color = "White (○)" if player == 1 else "Black (●)"
    print(f"\n Your agent is playing as: {color}")

# -------------------------------------------------------------------------------
# STRATEGY HELPER FUNCTIONS
# -------------------------------------------------------------------------------

def is_winning_move(board, move, player, EVAL_CACHE={}):
    """
    Determines if a given move results in a win for the specified player on the Hex board.

    This function simulates placing the player's piece at the specified move location,
    then evaluates whether this move leads to a win for the player. It uses a cache to
    avoid redundant board evaluations for previously seen positions.

    Args:
        board (list[list[int]]): The current Hex board state as a 2D list.
        move (tuple[int, int]): The (row, column) coordinates of the move to evaluate.
        player (int): The player making the move (typically 1 or 2).
        EVAL_CACHE (dict, optional): A cache dictionary for storing evaluated board positions.
            Defaults to an empty dict.

    Returns:
        bool: True if the move results in a win for the player, False otherwise.
    """
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

def is_forcing_win(board, move, player, EVAL_CACHE={}):
    """
    Determines if a given move is a forcing win for the specified player on the current Hex board.

    A forcing win is defined as a move that either immediately results in a win for the player,
    or leads to a board state where the player can guarantee a win regardless of the opponent's responses.

    Args:
        board (list[list[int]]): The current Hex board represented as a 2D list, where 0 indicates an empty cell,
            1 indicates a cell occupied by player 1, and -1 indicates a cell occupied by player -1.
        move (tuple[int, int]): The (row, column) coordinates of the move to evaluate.
        player (int): The player making the move (1 for white, -1 for black).
        EVAL_CACHE (dict, optional): A cache dictionary for storing previously evaluated board positions to avoid redundant computation.
            Defaults to an empty dictionary.

    Returns:
        bool: True if the move is a forcing win for the player, False otherwise.
    """
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

def find_most_direct_chain(board: List[List[int]], player: int, adjacency_map: Dict[Coordinate, List[Coordinate]]) -> Set[Coordinate]:
    size = len(board)
    visited = [[False]*size for _ in range(size)]

    def bfs(start: Coordinate) -> Set[Coordinate]:
        chain = set()
        queue = deque([start])
        while queue:
            current = queue.popleft()
            if current in chain:
                continue
            chain.add(current)
            for neighbor in adjacency_map[current]:
                if board[neighbor[0]][neighbor[1]] == player and neighbor not in chain:
                    queue.append(neighbor)
        return chain

    best_chain = set()
    best_score = -1

    for i in range(size):
        for j in range(size):
            if board[i][j] == player and not visited[i][j]:
                chain = bfs((i, j))
                for x, y in chain:
                    visited[x][y] = True

                if player == 1:
                    extent = max(y for x, y in chain) - min(y for x, y in chain)
                    progress = sum(1 for x, y in chain if y == 0 or y == size - 1)
                else:
                    extent = max(x for x, y in chain) - min(x for x, y in chain)
                    progress = sum(1 for x, y in chain if x == 0 or x == size - 1)

                score = progress * size + extent

                if score > best_score:
                    best_score = score
                    best_chain = chain

    return best_chain

def get_neighbors(i, j, size):
    """
    Returns a list of valid neighboring cell coordinates for a given cell (i, j) on a hexagonal grid.

    Args:
        i (int): The row index of the current cell.
        j (int): The column index of the current cell.
        size (int): The size of the grid (number of rows and columns; assumes a square grid).

    Returns:
        list of tuple: A list of (row, column) tuples representing the coordinates of neighboring cells
        that are within the bounds of the grid.

    Note:
        The function relies on the global variable HEX_NEIGHBORS, which should be a list of (di, dj)
        tuples representing the relative positions of neighboring cells in a hex grid.
    """
    return [
        (i+di, j+dj)
        for di, dj in HEX_NEIGHBORS
        if 0 <= i+di < size and 0 <= j+dj < size
    ]

def neighbors(board, player, cell):
    """
    Returns the neighboring cells of a given cell on the board that are either empty or occupied by the specified player.

    Args:
        board (list[list[int]]): The game board represented as a 2D list of integers.
        player (int): The player identifier (e.g., 1 or 2).
        cell (tuple[int, int]): The (row, column) coordinates of the cell whose neighbors are to be found.

    Returns:
        list[tuple[int, int]]: A list of (row, column) tuples representing the neighboring cells that are either empty (0) or occupied by the specified player.
    """
    i, j = cell
    size = len(board)
    return [(ni, nj) for ni, nj in get_neighbors(i, j, size) if board[ni][nj] in (0, player)]


def chain_cut_along_axis(board, player, edge_coords):
    """
    Checks if there's still a continuous path of 'player' stones connecting
    both sides of edge_coords (i.e. both "winning" edges).
    Returns True if *cut* (no full connection), False if still connected.
    """
    size = len(board)
    # Find any of the player's stones on one edge
    start = next(
        (pos for pos in edge_coords if board[pos[0]][pos[1]] == player),
        None
    )
    if not start:
        return True  # Without a stone on your edge, you're effectively disconnected

    visited = set([start])
    stack = [start]

    while stack:
        ci, cj = stack.pop()
        for ni, nj in get_neighbors(ci, cj, size):
            if board[ni][nj] == player and (ni, nj) not in visited:
                visited.add((ni, nj))
                stack.append((ni, nj))

    # Check if we reached both sides of the axis
    edge_cols = [(r, 0) for r in range(size)] if player == 1 else [(0, c) for c in range(size)]
    edge_cols += [(r, size - 1) for r in range(size)] if player == 1 else [(size - 1, c) for c in range(size)]

    reached = {pos for pos in visited if pos in edge_coords}

    # Player must reach at least one cell on *each* edge to avoid a cut
    if player == 1:
        return not (any(x == 0 for _, x in reached) and any(x == size - 1 for _, x in reached))
    else:
        return not (any(y == 0 for y, _ in reached) and any(y == size - 1 for y, _ in reached))



def dijkstra_shortest_paths(
    start_nodes: List[Coordinate],
    neighbor_fn: Callable[[Coordinate], List[Coordinate]],
    cost_fn: Callable[[Coordinate], int],
    goal_set: Set[Coordinate]
) -> Tuple[Dict[Coordinate, int], Dict[Coordinate, Optional[Coordinate]]]:
    dist: Dict[Coordinate, int] = {s: 0 for s in start_nodes}
    prev: Dict[Coordinate, Optional[Coordinate]] = {s: None for s in start_nodes}
    pq = [(0, s) for s in start_nodes]
    heapq.heapify(pq)
    visited: Set[Coordinate] = set()
    while pq:
        cur_d, node = heapq.heappop(pq)
        if node in visited:
            continue
        visited.add(node)
        if node in goal_set:
            # Optionally break early if desired
            pass
        for nbr in neighbor_fn(node):
            new_d = cur_d + cost_fn(nbr)
            if nbr not in dist or new_d < dist[nbr]:
                dist[nbr] = new_d
                prev[nbr] = node
                heapq.heappush(pq, (new_d, nbr))
    return dist, prev

# 2. Compute the full shortest path
def shortest_connection_path(board: List[List[int]], player: int) -> Optional[List[Coordinate]]:
    size = len(board)
    adjacency = get_adjacency_map(size)

    if player == 1:  # White: left-to-right
        starts = [(r, 0) for r in range(size) if board[r][0] != -1]
        goals = {(r, size - 1) for r in range(size)}
    else:  # Black: top-to-bottom
        starts = [(0, c) for c in range(size) if board[0][c] != 1]
        goals = {(size - 1, c) for c in range(size)}

    if not starts:
        return None

    def neighbor_fn(coord: Coordinate) -> List[Coordinate]:
        return [n for n in adjacency[coord] if board[n[0]][n[1]] != -player]

    def cost_fn(coord: Coordinate) -> int:
        return 0 if board[coord[0]][coord[1]] == player else 1

    dist, prev = dijkstra_shortest_paths(starts, neighbor_fn, cost_fn, goals)

    reachable = [g for g in goals if g in dist]
    if not reachable:
        return None

    goal = min(reachable, key=lambda g: dist[g])
    path: List[Coordinate] = []
    cur = goal
    while cur is not None:
        path.append(cur)
        cur = prev[cur]
    return list(reversed(path))




def is_cut_point(board, player):
    """
    Determines if the given player's stones on the board form a 'cut point'.

    A cut point is defined as a situation where not all of the player's stones are connected
    (i.e., there are at least two disconnected groups of the player's stones).

    Args:
        board (list[list[int]]): The game board represented as a 2D list, where each cell contains
            the player number or an empty value.
        player (int): The player number to check for cut points.

    Returns:
        bool: True if the player's stones are not all connected (i.e., a cut point exists),
        False otherwise.
    """
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

def fallback_random(action_set):
    """
    Selects and returns a random action from the provided set of actions.

    Args:
        action_set (Iterable): A collection (such as a list or set) of possible actions to choose from.

    Returns:
        Any: A randomly selected action from the action_set.

    Raises:
        IndexError: If action_set is empty.

    Note:
        The function relies on the global 'random' module being imported.
    """
    return random.choice(action_set)

# -------------------------------------------------------------------------------
# BASE STRATEGY FUNCTIONS
# -------------------------------------------------------------------------------

def take_center(board, action_set, player):
    """
    Selects the center position of the board if it is available in the given action set.

    Args:
        board (list[list[Any]]): The current state of the board as a 2D list.
        action_set (set[tuple[int, int]]): A set of available actions represented as (row, col) tuples.
        player (Any): The identifier for the current player (unused in this function).
            1 for white (left–right), -1 for black (top–bottom) 

    Returns:
        tuple[int, int] or None: The coordinates of the center position if available in action_set, otherwise None.
    """
    if not action_set:
        return None
    size = len(board)
    center = (size // 2, size // 2)

    def distance(a: Coordinate, b: Coordinate) -> float:
        return math.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)

    return min(action_set, key=lambda move: distance(move, center))

def extend_own_chain(board, action_set, player):
    """
    
    """
    size = len(board)
    adjacency_map = get_adjacency_map(size)

    main_chain = find_most_direct_chain(board, player, adjacency_map)

    candidate_moves = []
    if main_chain:
        axis_coords = [pos[1] if player == 1 else pos[0] for pos in main_chain]
        min_axis = min(axis_coords)
        max_axis = max(axis_coords)

        for move in action_set:
            for neighbor in adjacency_map[move]:
                if neighbor in main_chain:
                    move_axis = move[1] if player == 1 else move[0]
                    if move_axis < min_axis or move_axis > max_axis:
                        candidate_moves.append(move)

    if candidate_moves:
        return min(candidate_moves, key=lambda m: m[0] if player == -1 else m[1])

    axis_moves = []
    for move in action_set:
        dx_vals = [abs(move[0] - n[0]) for n in adjacency_map[move]]
        dy_vals = [abs(move[1] - n[1]) for n in adjacency_map[move]]
        if player == 1 and any(dy != 0 for dy in dy_vals):
            axis_moves.append(move)
        elif player == -1 and any(dx != 0 for dx in dx_vals):
            axis_moves.append(move)

    if axis_moves:
        return min(axis_moves, key=lambda m: m[0] if player == -1 else m[1])

    return None



def shortest_connection(board: List[List[int]], action_set: List[Coordinate], player: int) -> Optional[Coordinate]:
    best_move = None
    best_cost = float('inf')
    for move in action_set:
        # simulate board
        new_board = [row[:] for row in board]
        new_board[move[0]][move[1]] = player

        # get new shortest path and its cost
        path = shortest_connection_path(new_board, player)
        if path:
            cost = sum(1 for cell in path if new_board[cell[0]][cell[1]] != player)
        else:
            cost = float('inf')

        # select the move that gives minimal path cost
        if cost < best_cost:
            best_cost = cost
            best_move = move

    return best_move


def protect_own_chain_from_cut(board, action_set, player):
    """
    Attempts to select an action from the given action set that protects the player's own chain from being cut.

    This function iterates through each possible action (i, j) in the action_set, temporarily applies the action to the board,
    and checks if the player's chain would be cut using the is_cut_point function. If the action does not result in a cut,
    it is returned as the selected move. If all actions result in a cut, returns None.

    Args:
        board (list[list[int]]): The current game board represented as a 2D list.
        action_set (iterable[tuple[int, int]]): A set or list of possible (row, column) actions to consider.
        player (int): The identifier for the current player.

    Returns:
        tuple[int, int] or None: The (row, column) action that protects the player's chain, or None if no such action exists.
    """
    for i, j in action_set:
        board[i][j] = player
        cut = is_cut_point(board, player)
        board[i][j] = 0
        if not cut:
            return (i, j)
    return None






def make_own_bridge(board: List[List[int]], action_set: List[Coordinate], player: int) -> Coordinate | None:
    size = len(board)
    center = size // 2

    # Offsets, ordered by preference: first directional, then basic
    if player == 1:  # White (horizontal)
        offsets = [(1, -2), (-1, 2), (-1, -1), (1, 1)]
        axis = 1
    else:  # Black (vertical)
        offsets = [(2, -1), (-2, 1), (-1, -1), (1, 1)]
        axis = 0

    candidates = []

    for x in range(size):
        for y in range(size):
            if board[x][y] != player:
                continue
            for dx, dy in offsets:
                bx, by = x + dx, y + dy
                gx, gy = x + dx // 2, y + dy // 2  # the one gap

                if (
                    (bx, by) in action_set
                    and board[bx][by] == 0
                    and 0 <= gx < size and 0 <= gy < size
                    and board[gx][gy] == 0
                ):
                    candidates.append(((bx, by), dx, dy))

    if not candidates:
        return None

    # Separate directional vs basic bridges
    directional = [c for c in candidates if abs(c[1]) + abs(c[2]) > 2]
    basic = [c for c in candidates if abs(c[1]) + abs(c[2]) <= 2]

    def pick_best(group):
        # pick closest to center line along winning axis
        return min(group, key=lambda c: abs(c[0][axis] - center))[0]

    if directional:
        return pick_best(directional)
    else:
        return pick_best(basic)




# ══════════════════════════════════════════════════════════════════════════════
# Strategy registry – build AFTER all defs            ⬇️⬇️⬇️
# ══════════════════════════════════════════════════════════════════════════════
STRATEGY_FUNCTIONS: Dict[str, Callable] = {
    "take_center"                : take_center,
    "extend_own_chain"           : extend_own_chain,
    "break_opponent_bridge"      : break_opponent_bridge,
    "protect_own_chain_from_cut" : protect_own_chain_from_cut,
    "create_double_threat"       : create_double_threat,
    "shortest_connection"        : shortest_connection,
    "make_own_bridge"            : make_own_bridge,
    "mild_block_threat"          : mild_block_threat,
    "advance_toward_goal"        : advance_toward_goal,
    "block_aligned_opponent_path": block_aligned_opponent_path,
}
STRATEGIES = tuple(STRATEGY_FUNCTIONS)

# ---------------------------------------------------------------------------
# 🖨️  Cross-platform auto-print of STRATEGY_USE_COUNT at program exit
# (silent on miss)
# ---------------------------------------------------------------------------
def print_strategy_summary(counts: Dict[str, int]):
    """
    Prints a summary of how many times each strategy was used during the game.

    Args:
        counts (dict): A dictionary mapping strategy names (str) to the number of times (int) each strategy was used.

    Returns:
        None
    """
    print("\n Strategy usage summary:")
    for k in STRATEGIES:
        print(f"  {k}: {counts.get(k,0)} times")

def _locate_strategy_use_count():
    for frame in inspect.stack():
        if "STRATEGY_USE_COUNT" in frame.frame.f_globals:
            return frame.frame.f_globals["STRATEGY_USE_COUNT"]
    for mod in sys.modules.values():
        if hasattr(mod, "STRATEGY_USE_COUNT"):
            return getattr(mod, "STRATEGY_USE_COUNT")
    return None

def _summary_atexit():
    counts = _locate_strategy_use_count()
    if counts: print_strategy_summary(counts)

atexit.register(_summary_atexit)

__all__ = [
    "HEX_NEIGHBORS", "STRATEGY_FUNCTIONS", "STRATEGIES",
    "is_winning_move", "is_forcing_win", "infer_player", "fallback_random",
]