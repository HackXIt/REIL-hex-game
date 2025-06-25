# reil_hex_game.hex_engine - *enhanced for pygame integration*
# ==============================================================
# Copyright (c) 2025.
# ------------------------------------------------------------------
# This version augments the classical text-only engine with:
#   • Optional real-time pygame rendering (use_pygame=True)
#   • Console output reduced to **move list only** when pygame is on
#   • Machine-vs-machine flow controlled by the *Enter* key **inside** the
#     pygame window (no more stdin blocking)
#   • Automatic hook-up so the pygame window displays a status message
#     prompting the user in MvM mode.
# ------------------------------------------------------------------

from __future__ import annotations

import threading
from typing import Callable, List, Tuple

Coordinate = Tuple[int, int]
Agent = Callable[[List[List[int]], List[Coordinate]], Coordinate]

import threading  # ← added, required only when use_pygame=True
import atexit
import time

# ==============================================================
# 🏗️  Game object
# ==============================================================

class hexPosition (object):
    """
    Objects of this class correspond to a game of Hex.

    Parameters
    ----------
    size : int, optional
        Board side length (min 2, max 26, default 7)
    use_pygame : bool, keyword-only, optional
        Launch the *hex_pygame* visual front-end. When *True* the board is **not**
        printed to the console; instead, each move is echoed as a single line
        (e.g. ``White -> (3, 4)``).
    
    Attributes
    ----------
    size : int 
        The size of the board. The board is 'size*size'.
    board : list[list[int]]
        An array representing the hex board. '0' means empty. '1' means 'white'. '-1' means 'black'.
    player : int
        The player who is currently required to make a move. '1' means 'white'. '-1' means 'black'.
    winner : int
        Whether the game is won and by whom. '0' means 'no winner'. '1' means 'white' has won. '-1' means 'black' has won.
    history : list[list[list[int]]]
        A list of board-state arrays. Stores the history of play.
    """
    _closing: bool = False          # becomes True when close() is called
    # ------------------------------------------------------------------
    # Construction / initialisation
    # ------------------------------------------------------------------
    def __init__(self, size: int = 7, **kwargs):
        """Create a new :class:`hexPosition`.

        Parameters
        ----------
        size :
            Board side length (min 2, max 26, default 7).
        use_pygame :
            Launch graphical frontend (keyword-only, default ``False``).
        """
        # Extract and consume the *use_pygame* kwarg early so that the rest of
        # the code base remains completely oblivious to it.
        self._use_pygame: bool = bool(kwargs.pop('use_pygame', False))
        # If any other unkown kwargs were provided, raise an explicit error.
        if kwargs:
            raise TypeError(
                f"Unexpected keyword argument(s): {', '.join(kwargs.keys())}"
            )

        # ------------------------------------------------------------------
        # Original state initialisation (unchanged)
        # ------------------------------------------------------------------
        size = max(2, min(size, 26))       # clamp board size
        self.size = size
        board = [[0 for _ in range(size)] for _ in range(size)]
        self.board: List[List[int]] = board
        self.player: int = 1                    # 1 → white, -1 → black
        self.winner: int = 0
        self.history: List[List[List[int]]] = [board]

        # ------------------------------------------------------------------
        # Optional pygame visual backend
        # ------------------------------------------------------------------
        if self._use_pygame:
            self._human_move_queue: list[Coordinate] = []
            self._human_move_event = threading.Event()
            self._init_pygame_backend(size)
        atexit.register(self.close)      # run at interpreter exit

    # ==============================================================
    # Helpers for training environment
    # ==============================================================
    def legal_moves(self):
        """
        Return a list of empty coordinates (row, col) that the current
        player could legally occupy. Coordinates are 0-based.
        """
        return [
            (r, c)
            for r in range(self.size)
            for c in range(self.size)
            if self.board[r][c] == 0
        ]
    
    # ==============================================================
    # 🔌  PYGAME INTEGRATION
    # ==============================================================
    def _init_pygame_backend(self, size: int) -> None:
        """Starts the pygame render loop and prepares helper plumbing."""
        from importlib import import_module
        from .hex_pygame import GameState, game_loop   # local relative import

        # background thread that houses the GUI event loop
        self._pygame_game_state: GameState = GameState()
        gs = self._pygame_game_state
        gs.board_width_tiles = size
        gs.board_height_tiles = size
        gs.generate_board()

        gs.engine = self
        gs.step_event = None  # set later for MvM
        self._shutdown_event = threading.Event()
        gs.shutdown_event = self._shutdown_event

        # Start pygame in a daemon thread so the interpreter can exit cleanly
        self._pygame_thread = threading.Thread(
            target=game_loop, args=(gs,), daemon=True, name='hex_pygame_loop'
        )
        self._pygame_thread.start()

    # --------------------------------------------------------------
    # helpers to keep GUI ⟷  engine in sync
    # --------------------------------------------------------------
    def _sync_move_to_pygame(self, coord: Coordinate, player: int) -> None:
        if not self._use_pygame:
            return
        gs = self._pygame_game_state
        idx = 0 if player == 1 else 1
        tile = gs.hex_grid.tiles.get(coord)
        if tile is None:
            return  # board mismatch - should not happen
        tile.colour = gs.player_colour[idx]
        if tile not in gs.moves:
            gs.moves.append(tile)
        gs.solution = gs.find_solution()
        gs.current_player = 1 - idx  # toggle turn indicator

    def _reset_pygame_board(self) -> None:
        if not self._use_pygame:
            return
        gs = self._pygame_game_state
        gs.generate_board()
        gs.moves.clear()
        gs.solution = None
        gs.current_player = 0
    
    # ──────────────────────────────────────────────────────────────────
    #  Public clean-up – safe to call multiple times
    # ──────────────────────────────────────────────────────────────────
    def close(self) -> None:
        """Gracefully stop the GUI thread **and** unblock every waiter."""
        self._closing = True

        # ----- unblock any waiting game-logic loops ---------------------
        for ev_name in ('_click_event', '_human_move_event'):
            ev = getattr(self, ev_name, None)
            if ev:                          # Event object exists?
                ev.set()                    # ...then wake its waiters

        # step_event lives inside the GameState object
        gs = getattr(self, '_pygame_game_state', None)
        if gs:
            if getattr(gs, 'step_event', None):
                gs.step_event.set()
            if getattr(gs, 'click_event', None):
                gs.click_event.set()

        # ----- tell pygame loop to leave its while-loop -----------------
        if getattr(self, '_shutdown_event', None):
            self._shutdown_event.set()

        # ----- join the GUI thread (unless we *are* that thread) -------
        t = getattr(self, '_pygame_thread', None)
        if t and t.is_alive() and threading.current_thread() is not t:
            t.join(timeout=3)

    # ==============================================================
    # 🎮  Core gameplay primitives
    # ==============================================================
    def reset(self) -> None:
        self.board = [[0] * self.size for _ in range(self.size)]
        self.player = 1
        self.winner = 0
        self.history = []
        self._reset_pygame_board()

    def move(self, coordinates: Coordinate) -> None:
        assert self.winner == 0, 'The game is already won.'
        assert self.board[coordinates[0]][coordinates[1]] == 0, 'Field occupied.'

        from copy import deepcopy
        current_player = self.player
        self.board[coordinates[0]][coordinates[1]] = current_player

        # Optional GUI mirroring
        self._sync_move_to_pygame(coordinates, current_player)

        if hasattr(self, '_click_event') and self._click_event:
            # If the GUI is active, signal the click event to continue
            if current_player == getattr(self, "_human_player", current_player):
                self._click_event.set()

        # *Console output*: only moves when GUI is active
        if self._use_pygame:
            who = 'Red' if current_player == 1 else 'Blue'
            print(f'{who} -> {coordinates}')

        # Continue with normal game logic
        self.player *= -1
        self.evaluate()
        self.history.append(deepcopy(self.board))

    # ------------------------------------------------------------------
    # Everything below is identical to the original implementation
    # ------------------------------------------------------------------
    def print(self, invert_colors: bool = True) -> None:
        if self._use_pygame:
            return  # suppressed when GUI is running

        """Print a Unicode representation of the board to stdout."""
        names = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        indent = 0
        headings = " "*5 + (" "*3).join(names[:self.size])
        print(headings)
        tops = " "*5 + (" "*3).join("_"*self.size)
        print(tops)
        roof = " "*4 + "/ \\" + "_/ \\" * (self.size - 1)
        print(roof)
        if invert_colors:
            color_mapping = (
                lambda i: ' ' if i == 0 else ('\u25CB' if i == -1 else '\u25CF')
            )
        else:
            color_mapping = (
                lambda i: ' ' if i == 0 else ('\u25CF' if i == -1 else '\u25CB')
            )
        for r in range(self.size):
            row_mid = " "*indent + "   | " + " | ".join(
                map(color_mapping, self.board[r])
            ) + f" | {r+1} "
            print(row_mid)
            row_bottom = " "*indent + " "*3 + " \\/"*self.size
            if r < self.size - 1:
                row_bottom += " \\\\"
            print(row_bottom)
            indent += 2
        print(" "*(indent-2) + headings)
    
    def translator (string):
        #This function translates human terminal input into the proper array indices.
        number_translated = 27
        letter_translated = 27
        names = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        if len(string) > 0:
            letter = string[0]
        if len(string) > 1:
            number1 = string[1]
        if len(string) > 2:
            number2 = string[2]
        for i in range(26):
            if names[i] == letter:
                letter_translated = i
                break
        if len(string) > 2:
            for i in range(10,27):
                if number1 + number2 == "{}".format(i):
                    number_translated = i-1
        else:
            for i in range(1,10):
                if number1 == "{}".format(i):
                    number_translated = i-1
        return (number_translated, letter_translated)

    # ==============================================================
    # 🤖  Match wrappers
    # ==============================================================
    def machine_vs_machine(self, machine1: Agent | None = None, machine2: Agent | None = None, *, auto: bool = False, rate: float = 3.0):
        """Computer-controlled duel.

        In GUI mode the *Enter* key inside the pygame window advances the game
        turn-by-turn. A message below the board informs the user."""
        # prepare default random agents if missing
        from random import choice
        if machine1 is None:
            machine1 = lambda board, al: choice(al)
        if machine2 is None:
            machine2 = lambda board, al: choice(al)

        if not self._use_pygame:
            return self._machine_vs_machine_cli(machine1, machine2, auto=auto, rate=rate)
        else:
            return self._machine_vs_machine_gui(machine1, machine2, auto=auto, rate=rate)
    
    def human_vs_machine(self, human_player: int = 1, machine: Agent | None = None):
        """
        Play a game against an AI.

        • console run  → letter/number input (unchanged)  
        • pygame run   → human clicks on the board
        """
        if not self._use_pygame:
            return self._human_vs_machine_cli(human_player, machine)
        else:
            return self._human_vs_machine_gui(human_player, machine)
    
    def human_vs_human(self):
        """
        Play a game against another human player.

        • console run  → letter/number input (unchanged)  
        • pygame run   → human clicks on the board
        """
        if not self._use_pygame:
            return self._human_vs_human_cli()
        else:
            return self._human_vs_human_gui()
        
     # --------------------------------------------------------------
    # CLI implementation (unchanged from original behaviour)
    # --------------------------------------------------------------
    def _machine_vs_machine_cli(self, machine1: Agent, machine2: Agent, auto: bool, rate: float):
        self.reset()
        while self.winner == 0 and not self._closing:
            self.print()
            if auto:
                time.sleep(1/max(rate, 0.1))
            else:
                input('Press ENTER to continue.')
            chosen = machine1(self.board, self.get_action_space()) if self.player == 1 else machine2(self.board, self.get_action_space())
            self.move(chosen)
            if self.winner == 1:
                self.print(); self._evaluate_white(verbose=True)
            elif self.winner == -1:
                self.print(); self._evaluate_black(verbose=True)

    def _human_vs_machine_cli(self, human_player=1, machine=None):
        """
        Play a game against an AI. The variable machine must point to a function that maps a board state and an action set to an element of the action set.
        If machine is not specified random actions will be used.
        This method should not be used for training an algorithm.
        """
        #default to random player if 'machine' not given
        if machine == None:
            def machine (board, action_set):
                from random import choice
                return choice(action_set)
        #the match
        self.reset()
        while self.winner == 0 and not self._closing:
            self.print()
            possible_actions = self.get_action_space()
            if self.player == human_player:
                while True:
                    human_input = self.translator(input("Enter your move (e.g. 'A1'): "))
                    if human_input in possible_actions:
                        break
                self.move(human_input)
            else:
                chosen = machine(self.board, possible_actions)
                self.move(chosen)
            if self.winner == 1:
                self.print()
                self._evaluate_white(verbose=True)
            if self.winner == -1:
                self.print()
                self._evaluate_black(verbose=True)

    def _human_vs_human_cli(self):
        self.reset()
        while self.winner == 0 and not self._closing:
            self.print()
            possible_actions = self.get_action_space()
            while True:
                human_input = self.translator(input("Enter your move (e.g. 'A1'): "))
                if human_input in possible_actions:
                    break
            self.move(human_input)
            if self.winner == 1:
                self.print()
                self._evaluate_white(verbose=True)
            if self.winner == -1:
                self.print()
                self._evaluate_black(verbose=True)

    # --------------------------------------------------------------
    # GUI implementation with Enter-to-step
    # --------------------------------------------------------------
    def _machine_vs_machine_gui(self, machine1: Agent, machine2: Agent, auto: bool, rate: float):
        self.reset()
        # Step-event shared with pygame event loop
        step_event = threading.Event()
        self._pygame_game_state.step_event = step_event  # type: ignore[attr-defined]
        if not auto:
            self._pygame_game_state.status_message = 'Press SPACE to advance'  # type: ignore[attr-defined]
        else:
            self._pygame_game_state.status_message = 'Auto-play - press ENTER to pause'  # type: ignore[attr-defined]
        self._pygame_game_state.auto_mode = auto  # type: ignore[attr-defined]
        self._pygame_game_state.auto_delay = 1 / rate if rate else 0.33  # type: ignore[attr-defined]

        while self.winner == 0 and not self._closing:
            # Wait for Enter key from GUI; tiny timeout keeps us responsive
            if self._pygame_game_state.auto_mode: # continuous mode
                if self._closing:
                    break
                time.sleep(self._pygame_game_state.auto_delay)
            else:                                        # step-mode
                while not self._closing and not step_event.wait(0.1):
                    pass
                step_event.clear()

            chosen = machine1(self.board, self.get_action_space()) if self.player == 1 else machine2(self.board, self.get_action_space())
            self.move(chosen)
        if self._shutdown_event.is_set():
            self.close()
            return
        # Announce winner in console (GUI already highlights path)
        if self.winner == 1:
            print('Red wins!')
        else:
            print('Blue wins!')
    
    def _human_vs_machine_gui(self, human_player: int, machine: Agent):
        """GUI version - human clicks, bot replies immediately."""

        if machine is None:                # default random bot
            from random import choice
            machine = lambda b, a: choice(a)

        self._human_player = human_player # remember color of human player

        self.reset()
        gs = self._pygame_game_state
        gs.status_message = 'Your turn - click a hex'

        click_event = threading.Event()
        self._click_event = click_event
        gs.click_event = click_event

        while self.winner == 0 and not self._closing:
            if self.player == self._human_player:
                while not self._closing and not click_event.wait(0.05):
                    pass
                if self._closing:
                    break
            else:
                gs.status_message = 'Bot is thinking…'
                chosen = machine(self.board, self.get_action_space())
                self.move(chosen)
                gs.status_message = 'Your turn - click a hex'

        print('Red wins!' if self.winner == 1 else 'Blue wins!')

    def _human_vs_human_gui(self):
        self.reset()
        gs = self._pygame_game_state
        gs.status_message = 'Your turn - click a hex'

        click_event = threading.Event()
        self._click_event = click_event
        gs.click_event = click_event

        while self.winner == 0 and not self._closing:
            while not self._closing and not click_event.wait(0.05):
                pass
            click_event.clear()
            if self._closing:
                break
        print('Red wins!' if self.winner == 1 else 'Blue wins!')

    # --------------------------------------------------------------
    # (The rest of the original methods without modification)
    # _get_adjacent
    # get_action_space
    # _random_move
    # _random_match
    # _prolong_path
    # evaluate
    # _evaluate_white
    # _evaluate_black
    # (human_vs_machine and machine_vs_machine was replaced by the above)
    # recode_black_as_white
    # recode_coordinates
    # coordinate_to_scalar
    # scalar_to_coordinates
    # replay_history
    # save
    # --------------------------------------------------------------
    def _get_adjacent (self, coordinates):
        """
        Helper function to obtain adjacent cells in the board array.
        Used in position evaluation to construct paths through the board.
        """
        u = (coordinates[0]-1, coordinates[1])
        d = (coordinates[0]+1, coordinates[1])
        r = (coordinates[0], coordinates[1]-1)
        l = (coordinates[0], coordinates[1]+1)
        ur = (coordinates[0]-1, coordinates[1]+1)
        dl = (coordinates[0]+1, coordinates[1]-1)
        return [pair for pair in [u,d,r,l,ur,dl] if max(pair[0], pair[1]) <= self.size-1 and min(pair[0], pair[1]) >= 0]

    def get_action_space (self, recode_black_as_white=False):
        """
        This method returns a list of array positions which are empty (on which stones may be put).
        """
        actions = []
        for i in range(self.size):
            for j in range(self.size):
                if self.board[i][j] == 0:
                    actions.append((i,j))
        if recode_black_as_white:
            return [self.recode_coordinates(action) for action in actions]
        else:
            return(actions)

    def _random_move (self):
        """
        This method enacts a uniformly randomized valid move.
        """
        from random import choice
        chosen = choice(self.get_action_space())
        self.move(chosen)

    def _random_match (self):
        """
        This method randomizes an entire playthrough. Mostly useful to test code functionality.
        """
        while self.winner == 0:
            self._random_move()

    def _prolong_path (self, path):
        """
        A helper function used for board evaluation.
        """
        player = self.board[path[-1][0]][path[-1][1]]
        candidates = self._get_adjacent(path[-1])
        #preclude loops
        candidates = [cand for cand in candidates if cand not in path]
        candidates = [cand for cand in candidates if self.board[cand[0]][cand[1]] == player]
        return [path+[cand] for cand in candidates]

    def evaluate (self, verbose=False):
        """
        Evaluates the board position and adjusts the 'winner' attribute of the object accordingly.
        """
        self._evaluate_white(verbose=verbose)
        self._evaluate_black(verbose=verbose)

    def _evaluate_white (self, verbose):
        """
        Evaluate whether the board position is a win for player '1'. Uses breadth first search.
        If verbose=True a winning path will be printed to the standard output (if one exists).
        This method may be time-consuming for huge board sizes.
        """
        paths = []
        visited = []
        for i in range(self.size):
            if self.board[i][0] == 1:
                paths.append([(i,0)])
                visited.append([(i,0)])
        while True:
            if len(paths) == 0:
                return False
            for path in paths:
                prolongations = self._prolong_path(path)
                paths.remove(path)
                for new in prolongations:
                    if new[-1][1] == self.size-1:
                        if verbose:
                            print("A winning path for 'white' ('1'):\n",new)
                        self.winner = 1
                        return True
                    if new[-1] not in visited:
                        paths.append(new)
                        visited.append(new[-1])

    def _evaluate_black (self, verbose):
        """
        Evaluate whether the board position is a win for player '-1'. Uses breadth first search.
        If verbose=True a winning path will be printed to the standard output (if one exists).
        This method may be time-consuming for huge board sizes.
        """
        paths = []
        visited = []
        for i in range(self.size):
            if self.board[0][i] == -1:
                paths.append([(0,i)])
                visited.append([(0,i)])
        while True:
            if len(paths) == 0:
                return False
            for path in paths:
                prolongations = self._prolong_path(path)
                paths.remove(path)
                for new in prolongations:
                    if new[-1][0] == self.size-1:
                        if verbose:
                            print("A winning path for 'black' ('-1'):\n",new)
                        self.winner = -1
                        return True
                    if new[-1] not in visited:
                        paths.append(new)
                        visited.append(new[-1])

    def recode_black_as_white (self, print=False, invert_colors=True):
        """
        Returns a board where black is recoded as white and wants to connect horizontally.
        This corresponds to flipping the board along the south-west to north-east diagonal and swapping colors.
        This may be used to train AI players in a 'color-blind' way.
        """
        flipped_board = [[0 for i in range(self.size)] for j in range(self.size)]
        #flipping and color change
        for i in range(self.size):
            for j in range(self.size):
                if self.board[self.size-1-j][self.size-1-i] == 1:
                    flipped_board[i][j] = -1
                if self.board[self.size-1-j][self.size-1-i] == -1:
                    flipped_board[i][j] = 1
        return flipped_board

    def recode_coordinates (self, coordinates):
        """
        Transforms a coordinate tuple (with respect to the board) analogously to the method recode_black_as_white.
        """
        assert(0 <= coordinates[0] and self.size-1 >= coordinates[0]), "There is something wrong with the first coordinate."
        assert(0 <= coordinates[1] and self.size-1 >= coordinates[1]), "There is something wrong with the second coordinate."
        return (self.size-1-coordinates[1], self.size-1-coordinates[0])

    def coordinate_to_scalar (self, coordinates):
        """
        Helper function to convert coordinates to scalars.
        This may be used as alternative coding for the action space.
        """
        assert(0 <= coordinates[0] and self.size-1 >= coordinates[0]), "There is something wrong with the first coordinate."
        assert(0 <= coordinates[1] and self.size-1 >= coordinates[1]), "There is something wrong with the second coordinate."
        return coordinates[0]*self.size + coordinates[1]

    def scalar_to_coordinates (self, scalar):
        """
        Helper function to transform a scalar "back" to coordinates.
        Reverses the output of 'coordinate_to_scalar'.
        """
        coord1 = int(scalar/self.size)
        coord2 = scalar - coord1 * self.size
        assert(0 <= coord1 and self.size-1 >= coord1), "The scalar input is invalid."
        assert(0 <= coord2 and self.size-1 >= coord2), "The scalar input is invalid."
        return (coord1, coord2)

    def replay_history (self):
        """
        Print the game history to standard output.
        """
        for board in self.history:
            temp = hexPosition(size=self.size)
            temp.board = board
            temp.print()
            input("Press ENTER to continue.")

    def save (self, path):
        """
        Serializes the object as a bytestream.
        """
        import pickle
        file = open(path, 'ab')
        pickle.dump(self, file)
        file.close()
