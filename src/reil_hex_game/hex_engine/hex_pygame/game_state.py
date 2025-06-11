# Credits: https://github.com/parappayo/hex-py/blob/master/game_state.py
# License: (MIT License) https://github.com/parappayo/hex-py/blob/master/LICENSE
# Author: Parappayo
# Modified by: HackXIt

from . import hex_geometry 
from ..hex_engine import hexPosition

class GameState:

    def __init__(self):
        self.background_colour = 0, 0, 0 # rgb 256
        self.screen_size = 1024, 768 # pixels
        self.board_position = 100, 100 # pixels

        self.status_message: str = ""
        self.engine: hexPosition = None
        self.step_event = None
        self.click_event = None
        # graceful shutdown flag (gets filled by hex_engine)
        self.shutdown_event = None
        self.auto_mode      = False  # bool – set by engine / Enter key
        self.auto_delay     = 1/3    # seconds per move (default 3 Hz)

        self.hex_tile_size = 32
        self.board_width_tiles = 11
        self.board_height_tiles = 11

        self.empty_hex_colour = (32, 32, 32)
        self.cursor_colour = (32, 128, 32)
        self.player_colour = [(255, 0, 0), (0, 0, 255)]

        self.nearest_tile_to_mouse = None
        self.current_player = 0
        self.moves = []
        self.solution = None

        self.generate_board()


    def generate_board(self):
        points_up = True

        self.hex_grid = hex_geometry.HexGrid(
            self.board_width_tiles,
            self.board_height_tiles,
            self.hex_tile_size,
            points_up)

        for tile in self.hex_tiles():
            tile.colour = self.empty_hex_colour


    def hex_tiles(self):
        return self.hex_grid.tiles.values()


    def nearest_hex_tile(self, pos):
        result = None
        min_distance = None

        for tile in self.hex_tiles():
            tile_distance = tile.distance_squared(pos, self.board_position)
            if result == None:
                min_distance = tile_distance
                result = tile
            elif tile_distance < min_distance:
                min_distance = tile_distance
                result = tile

        return result


    def find_solution(self):
        for tile in self.hex_grid.top_row():
            path = self.hex_grid.find_path(
                tile,
                self.hex_grid.bottom_row(),
                lambda x: x.colour == self.player_colour[0])
            if path != None:
                return path

        for tile in self.hex_grid.left_column():
            path = self.hex_grid.find_path(
                tile,
                self.hex_grid.right_column(),
                lambda x: x.colour == self.player_colour[1])
            if path != None:
                return path

        return None


    def is_game_over(self):
        return self.solution != None


    def is_valid_move(self, tile=None):
        if self.is_game_over():
            return False
        if tile == None:
            tile = self.nearest_tile_to_mouse
        return not tile in self.moves


    def take_move(self, tile=None):
        if tile == None:
            tile = self.nearest_tile_to_mouse
        tile.colour = self.player_colour[self.current_player]
        self.moves.append(tile)
        self.toggle_player_turn()
        self.solution = self.find_solution()


    def toggle_player_turn(self):
        if self.current_player == 0:
            self.current_player = 1
        else:
            self.current_player = 0