import random, heapq, atexit, inspect, sys
from ..hex_engine.hex_engine import hexPosition
from copy import deepcopy
from typing import List, Tuple, Dict, Callable

# -------------------------------------------------------------------------------
# CONSTANTS
# -------------------------------------------------------------------------------
Coordinate = Tuple[int, int]
# Neighboring directions on a hex grid (pointy-top orientation)
HEX_NEIGHBORS = [(-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0)]

# -------------------------------------------------------------------------------
# GENERAL HELPER FUNCTIONS FOR RULE BASED AGENT
# -------------------------------------------------------------------------------

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

def dijkstra(board, player, start_edges, target_edges):
    """
    Finds the shortest path from any of the given start edges to any of the target edges on a Hex board using Dijkstra's algorithm.
    Parameters:
        board (list[list[int]]): The Hex board represented as a 2D list, where each cell contains 0 (empty), 1, or 2 (player markers).
        player (int): The player number (1 or 2) for whom the path is being calculated.
        start_edges (Iterable[Tuple[int, int]]): Iterable of (row, col) tuples representing the starting edge positions.
        target_edges (Set[Tuple[int, int]]): Set of (row, col) tuples representing the target edge positions.
    Returns:
        Tuple[int, int] or None: The coordinates (row, col) of the first target edge reached via the shortest path, or None if no path exists.
    """
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
        for ni, nj in get_neighbors(i, j, len(board)):
            if (ni, nj) not in visited:
                cost = 0 if board[ni][nj] == player else 1
                heapq.heappush(heap, (dist + cost, (ni, nj)))
    return None

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

    Returns:
        tuple[int, int] or None: The coordinates of the center position if available in action_set, otherwise None.
    """
    size = len(board)
    center = (size // 2, size // 2)
    return center if center in action_set else None

def extend_own_chain(board, action_set, player):
    """
    Attempts to extend the current player's chain by selecting a neighboring empty position.

    This function searches for the player's existing positions on the board, shuffles them to introduce randomness,
    and then checks each neighbor of those positions. If a neighbor is available in the given action set (i.e., is a valid move),
    the function returns that position as the next action to extend the player's chain.

    Args:
        board (list[list[int]]): The current state of the game board as a 2D list, where each cell indicates the occupying player or is empty.
        action_set (set[tuple[int, int]]): A set of available actions represented as (row, column) tuples.
        player (int): The identifier for the current player.

    Returns:
        tuple[int, int] or None: The selected action (row, column) to extend the player's chain, or None if no such move is found.
    """
    size = len(board)
    stones = [(i, j) for i in range(size) for j in range(size) if board[i][j] == player]
    random.shuffle(stones)
    for i, j in stones:
        for ni, nj in get_neighbors(i, j, size):
            if (ni, nj) in action_set:
                return (ni, nj)
    return None

def break_opponent_bridge(board, action_set, player):
    """
    Attempts to find and return a move that breaks an opponent's bridge formation.

    A "bridge" in Hex is a two-stone diagonal connection with an empty cell between them,
    which can be used to create a strong link. This function scans the board for such
    enemy bridges and, if possible, returns the coordinates of the empty cell between
    two enemy stones that can be played to break the bridge.

    Args:
        board (list[list[int]]): The current Hex board as a 2D list, where 0 represents an empty cell,
                                 `player` represents the current player's stones, and `-player` the opponent's.
        action_set (set[tuple[int, int]]): Set of available actions (empty cells) as (row, col) tuples.
        player (int): The current player's identifier (typically 1 or -1).

    Returns:
        tuple[int, int] or None: The coordinates (row, col) of a move that breaks an opponent's bridge,
                                 or None if no such move is found.
    """
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

def create_double_threat(board, action_set, player):
    """
    Identifies and returns an action from the given action_set that creates a "double threat" for the specified player.

    A "double threat" is defined as a move that connects two or more separate clusters of the player's pieces,
    potentially creating multiple simultaneous threats on the board.

    Args:
        board (list[list[int]]): The current game board represented as a 2D list, where each cell contains a value indicating its state (e.g., empty, player 1, player 2).
        action_set (set[tuple[int, int]]): A set of available actions, where each action is a tuple (i, j) representing a board position.
        player (int): The identifier for the player (e.g., 1 or 2) for whom the double threat is being created.

    Returns:
        tuple[int, int] or None: The coordinates (i, j) of an action that creates a double threat, or None if no such action exists.

    Note:
        This function assumes the existence of a helper function `get_neighbors(i, j, size)` that returns the neighboring positions
        of a given cell (i, j) on the board of the specified size.
    """
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
    """
    Finds the shortest connection path for a given player on a Hex board using Dijkstra's algorithm.
    This function determines the optimal move for the specified player by searching for the shortest path
    between the player's starting and goal edges. The search is restricted to the set of available actions.
    Args:
        board (list[list[int]]): The current Hex board represented as a 2D list, where each cell indicates
            the state (e.g., empty, black, white).
        action_set (set[tuple[int, int]]): A set of available moves, each represented as a (row, col) tuple.
        player (int): The player for whom to compute the path. Use -1 for Black (left-to-right), and 1 for White (top-to-bottom).
    Returns:
        tuple[int, int] or None: The move (row, col) that is part of the shortest connection path for the player,
            if such a move exists in the action_set; otherwise, None.
    """
    size = len(board)

    if player == -1:  # Black plays left <-> right
        starts = [(i, 0) for i in range(size)]
        goals = [(i, size - 1) for i in range(size)]
    else:  # White plays top <-> bottom
        starts = [(0, j) for j in range(size)]
        goals = [(size - 1, j) for j in range(size)]

    best_move = dijkstra(board, player, starts, set(goals))
    return best_move if best_move in action_set else None

def favor_bridges(board, action_set, player):
    """
    Selects a move that favors forming "bridges" between the player's existing stones on a Hex board.
    A "bridge" is a potential connection between two stones that are two or three steps apart, 
    which can help the player build a path across the board. The function prioritizes moves that 
    create such connections, taking into account the preferred connection direction for each player:
    vertical for White (player 1) and horizontal for Black (player -1).
    Args:
        board (list[list[int]]): The current Hex board as a 2D list, where each cell is 0 (empty), 1 (White), or -1 (Black).
        action_set (list[tuple[int, int]]): A list of available moves, each as a (row, col) tuple.
        player (int): The current player (1 for White, -1 for Black).
    Returns:
        tuple[int, int] or None: The selected move that favors bridge formation, or None if no such move is found.
    """
    size = len(board)
    stones = [(i, j) for i in range(size) for j in range(size) if board[i][j] == player]

    for move in action_set:
        mx, my = move
        for (sx, sy) in stones:
            dx = mx - sx
            dy = my - sy
            distance = abs(dx) + abs(dy)
            if distance in (2, 3):  # potential bridge
                # Respect connection direction: vertical for White (player 1), horizontal for Black (player -1)
                if (player == 1 and abs(dx) > abs(dy)) or (player == -1 and abs(dy) > abs(dx)):
                    return move
    return None

def mild_block_threat(board, action_set, player):
    """
    Attempts to block mild threats from the opponent by simulating each possible move and checking for potential opponent threats.

    Args:
        board (list[list[int]]): The current game board represented as a 2D list, where 0 indicates an empty cell, and positive/negative values represent players.
        action_set (list[tuple[int, int]]): A list of possible moves, each represented as a tuple of (row, column) indices.
        player (int): The current player's identifier (typically 1 or -1).

    Returns:
        tuple[int, int] or None: Returns the move (row, column) that blocks a mild threat if found (i.e., a move after which the opponent has more than 10 potential threats), otherwise returns None.

    Note:
        This function assumes the existence of the global variable HEX_NEIGHBORS, which should define the relative neighbor positions for the hex grid.
    """
    opponent = -player
    size = len(board)
    for move in action_set:
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

def advance_toward_goal(board, action_set, player):
    """
    Selects the move that advances the player's position toward their goal edge in a Hex game.
    For player 1 (White), who aims to connect top to bottom, the function favors moves that increase the average row index.
    For player 2 (Black), who aims to connect left to right, it favors moves that increase the average column index.
    Args:
        board (list[list[int]]): The current Hex board as a 2D list, where each cell is 0 (empty), 1 (White), or 2 (Black).
        action_set (list[tuple[int, int]]): A list of available moves, each as a (row, column) tuple.
        player (int): The player number (1 for White, 2 for Black).
    Returns:
        tuple[int, int] or None: The move (row, column) that best advances toward the goal, or None if the player has no stones on the board.
    """
    size = len(board)
    own_stones = [(i, j) for i in range(size) for j in range(size) if board[i][j] == player]
    if not own_stones:
        return None

    if player == 1:
        # White goes top to bottom -> favor increasing row (i)
        avg = sum(i for i, _ in own_stones) / len(own_stones)
        return max(action_set, key=lambda m: m[0] - avg)
    else:
        # Black goes left to right -> favor increasing column (j)
        avg = sum(j for _, j in own_stones) / len(own_stones)
        return max(action_set, key=lambda m: m[1] - avg)

def block_aligned_opponent_path(board, action_set, player):
    """
    Selects a move that blocks the opponent's most aligned path towards victory in Hex.
    This function analyzes the current board state to identify the axis (row or column)
    where the opponent has the highest concentration of stones aligned in their winning direction.
    It then selects a move from the available moves that would block the opponent's progress
    along that axis, if possible.
    Args:
        board (list[list[int]]): The current Hex board as a 2D list, where each cell is 0 (empty), 1 (player 1), or -1 (player -1).
        action_set (list[tuple[int, int]]): A list of available moves, each represented as a tuple (row, col).
        player (int): The current player (1 or -1).
    Returns:
        tuple[int, int] or None: A move (row, col) that blocks the opponent's most aligned path,
        or None if no such move is found or the opponent has no stones on the board.
    """
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
    for move in action_set:
        if (opponent == -1 and move[1] == target) or (opponent == 1 and move[0] == target):
            return move
    return None

# ══════════════════════════════════════════════════════════════════════════════
# Strategy registry – build AFTER all defs            ⬇️⬇️⬇️
# ══════════════════════════════════════════════════════════════════════════════
STRATEGY_FUNCTIONS: Dict[str, Callable] = {
    "take_center"                : take_center,
    "extend_own_chain"           : extend_own_chain,
    "break_opponent_bridge"      : break_opponent_bridge,
    "protect_own_chain_from_cut" : protect_own_chain_from_cut,
    "create_double_threat"       : create_double_threat,
    "shortest_connection_path"   : shortest_connection_path,
    "favor_bridges"              : favor_bridges,
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