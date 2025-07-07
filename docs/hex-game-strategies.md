# Hex Game Strategies

This document provides a detailed explanation of the various logic-based strategies implemented in the `rule_based_helper.py` file. Each strategy is designed to identify and suggest moves based on different tactical and strategic considerations in the game of Hex.

-----

## ⚔️ Implemented Strategies

### `take_center`

This strategy suggests playing in the center of the board, which is a common opening move in Hex. It is most effective in the early stages of the game.

**Logic:**
The strategy identifies the center of the board and suggests this move if it's available.

**Pseudo-code:**

```
FUNCTION take_center(board, action_set):
  center = (board_size / 2, board_size / 2)
  IF center is in action_set:
    RETURN center
  ELSE:
    RETURN None
```

[Code Reference](https://github.com/HackXIt/REIL-hex-game/blob/main/src/reil_hex_game/agents/rule_based_helper.py#L549)

-----

### `extend_own_chain`

This strategy focuses on extending existing chains of the player's stones.

**Logic:**
It identifies the player's most promising chain and suggests a move that extends it.

**Pseudo-code:**

```
FUNCTION extend_own_chain(board, action_set, player):
  my_positions = GET_ALL_POSITIONS(player)
  SHUFFLE(my_positions)
  FOR each position in my_positions:
    FOR each neighbor of position:
      IF neighbor is in action_set:
        RETURN neighbor
  RETURN None
```

[Code Reference](https://github.com/HackXIt/REIL-hex-game/blob/main/src/reil_hex_game/agents/rule_based_helper.py#L570)

-----

### `shortest_connection`

This strategy aims to find the shortest path to victory and suggests a move that contributes to it.

**Logic:**
It uses Dijkstra's algorithm to find the shortest path for the player to connect their sides of the board.

**Pseudo-code:**

```
FUNCTION shortest_connection(board, action_set, player):
  best_move = None
  best_cost = infinity
  FOR each move in action_set:
    new_board = SIMULATE_MOVE(board, move, player)
    path = DIJKSTRA(new_board, player)
    cost = CALCULATE_PATH_COST(path)
    IF cost < best_cost:
      best_cost = cost
      best_move = move
  RETURN best_move
```

[Code Reference](https://github.com/HackXIt/REIL-hex-game/blob/main/src/reil_hex_game/agents/rule_based_helper.py#L611)

-----

### `break_opponent_bridge`

This strategy is designed to thwart the opponent's plans by breaking their bridges.

**Logic:**
It identifies two opponent stones that are separated by one empty cell and suggests placing a stone in that empty cell.

**Pseudo-code:**

```
FUNCTION break_opponent_bridge(board, action_set, player):
  enemy = -player
  FOR each cell on the board:
    IF cell is occupied by enemy:
      FOR each bridge pattern:
        IF bridge is valid and the middle cell is in action_set:
          RETURN middle_cell
  RETURN None
```

[Code Reference](https://github.com/HackXIt/REIL-hex-game/blob/main/src/reil_hex_game/agents/rule_based_helper.py#L635)

-----

### `protect_own_chain_from_cut`

This defensive strategy aims to prevent the opponent from cutting the player's chains.

**Logic:**
It simulates each possible move and checks if it would lead to a cut in the player's chains.

**Pseudo-code:**

```
FUNCTION protect_own_chain_from_cut(board, action_set, player):
  FOR each move in action_set:
    new_board = SIMULATE_MOVE(board, move, player)
    IF new_board is not a cut point:
      RETURN move
  RETURN None
```

[Code Reference](https://github.com/HackXIt/REIL-hex-game/blob/main/src/reil_hex_game/agents/rule_based_helper.py#L671)

-----

### `create_double_threat`

This offensive strategy tries to create multiple threats at once.

**Logic:**
It looks for a move that connects two or more of the player's existing stone clusters.

**Pseudo-code:**

```
FUNCTION create_double_threat(board, action_set, player):
  clusters = FIND_CLUSTERS(board, player)
  FOR each move in action_set:
    connected_clusters = 0
    FOR each cluster in clusters:
      IF move is adjacent to cluster:
        connected_clusters += 1
    IF connected_clusters >= 2:
      RETURN move
  RETURN None
```

[Code Reference](https://github.com/HackXIt/REIL-hex-game/blob/main/src/reil_hex_game/agents/rule_based_helper.py#L697)

-----

### `make_own_bridge`

This strategy proactively builds bridges to strengthen the player's position.

**Logic:**
It identifies two of the player's stones that can form a bridge and suggests placing a stone to complete it.

**Pseudo-code:**

```
FUNCTION make_own_bridge(board, action_set, player):
  FOR each cell on the board:
    IF cell is occupied by player:
      FOR each bridge pattern:
        IF bridge is valid and the connecting cell is in action_set:
          RETURN connecting_cell
  RETURN None
```

[Code Reference](https://github.com/HackXIt/REIL-hex-game/blob/main/src/reil_hex_game/agents/rule_based_helper.py#L745)

-----

### `mild_block_threat`

This strategy focuses on blocking the opponent's progress in a less direct way.

**Logic:**
It simulates each possible move and checks if it weakens the opponent's shortest path to victory.

**Pseudo-code:**

```
FUNCTION mild_block_threat(board, action_set, player):
  opponent = -player
  base_path = DIJKSTRA(board, opponent)
  base_threat = CALCULATE_THREAT(base_path)
  best_move = None
  max_reduction = 0
  FOR each move in action_set:
    new_board = SIMULATE_MOVE(board, move, player)
    new_path = DIJKSTRA(new_board, opponent)
    new_threat = CALCULATE_THREAT(new_path)
    reduction = base_threat - new_threat
    IF reduction > max_reduction:
      max_reduction = reduction
      best_move = move
  RETURN best_move
```

[Code Reference](https://github.com/HackXIt/REIL-hex-game/blob/main/src/reil_hex_game/agents/rule_based_helper.py#L793)

-----

### `advance_toward_goal`

This strategy encourages moves that advance towards the player's goal.

**Logic:**
It prioritizes moves that are closer to the opposite side of the board that the player needs to connect.

**Pseudo-code:**

```
FUNCTION advance_toward_goal(board, action_set, player):
  IF player is white:
    RETURN move in action_set with the highest column index
  ELSE:
    RETURN move in action_set with the highest row index
```

[Code Reference](https://github.com/HackXIt/REIL-hex-game/blob/main/src/reil_hex_game/agents/rule_based_helper.py#L836)

-----

### `block_aligned_opponent_path`

This strategy blocks the opponent's most direct path.

**Logic:**
It identifies the axis along which the opponent has the most stones and suggests a move to block that axis.

**Pseudo-code:**

```
FUNCTION block_aligned_opponent_path(board, action_set, player):
  opponent = -player
  axis_counts = COUNT_STONES_ALONG_AXIS(board, opponent)
  best_axis = FIND_BEST_AXIS(axis_counts)
  candidates = FIND_MOVES_ON_AXIS(action_set, best_axis)
  IF candidates:
    RETURN move in candidates closest to the center
  ELSE:
    RETURN None
```

[Code Reference](https://github.com/HackXIt/REIL-hex-game/blob/main/src/reil_hex_game/agents/rule_based_helper.py#L857)