# Credits: https://github.com/parappayo/hex-py/blob/master/game_draw.py
# License: (MIT License) https://github.com/parappayo/hex-py/blob/master/LICENSE
# Author: Parappayo
# Modified by: HackXIt

import pygame
from .game_state import GameState
from .hex_geometry import HexTile

def draw_hex_tile(surface, game: GameState, tile: HexTile):
    center_point = tile.center_point(game.board_position)
    corner_points = tile.corner_points(game.board_position)
    pygame.draw.polygon(surface, tile.colour, corner_points)
    pygame.draw.polygon(surface, (255, 255, 255), corner_points, 2) # border

    if tile == game.nearest_tile_to_mouse:
        if not game.is_game_over():
            pygame.draw.circle(surface, game.cursor_colour, center_point, 12)
            #draw_hex_neighbours(surface, game, tile, (255, 255, 255))


def draw_hex_neighbours(surface, game: GameState, tile: HexTile, colour):
    width = 4
    from_point = tile.center_point(game.board_position)
    for neighbour in tile.neighbours:
        to_point = neighbour.center_point(game.board_position)
        pygame.draw.line(surface, colour, from_point, to_point, width)


def draw_hex_path(surface, game: GameState, path, colour):
    width = 4
    for i in range(len(path)-1):
        from_point = path[i].center_point(game.board_position)
        to_point = path[i+1].center_point(game.board_position)
        pygame.draw.line(surface, colour, from_point, to_point, width)


def draw_hex_top_border(surface, game: GameState, tile: HexTile, colour):
    width = 4
    corner_points = tile.corner_points(game.board_position)
    pygame.draw.lines(surface, colour, False, corner_points[3:6], width)


def draw_hex_bottom_border(surface, game: GameState, tile: HexTile, colour):
    width = 4
    corner_points = tile.corner_points(game.board_position)
    pygame.draw.lines(surface, colour, False, corner_points[0:3], width)


def draw_hex_left_border(surface, game: GameState, tile: HexTile, colour):
    width = 4
    corner_points = tile.corner_points(game.board_position)
    pygame.draw.lines(surface, colour, False, corner_points[1:4], width)


def draw_hex_right_border(surface, game: GameState, tile: HexTile, colour):
    width = 4
    corner_points = tile.corner_points(game.board_position)
    points = (corner_points[4], corner_points[5], corner_points[0])
    pygame.draw.lines(surface, colour, False, points, width)


def draw_board(surface, game: GameState):
    for tile in game.hex_tiles():
        draw_hex_tile(surface, game, tile)
    if game.solution != None:
        draw_hex_path(surface, game, game.solution, (255, 255, 255))


def draw_end_zones(surface, game: GameState):
    player_one_colour = game.player_colour[0]
    player_two_colour = game.player_colour[1]

    for tile in game.hex_grid.top_row():
        draw_hex_left_border(surface, game, tile, player_one_colour)

    for tile in game.hex_grid.bottom_row():
        draw_hex_right_border(surface, game, tile, player_one_colour)
 
    for tile in game.hex_grid.left_column():
        draw_hex_top_border(surface, game, tile, player_two_colour)

    for tile in game.hex_grid.right_column():
        draw_hex_bottom_border(surface, game, tile, player_two_colour)


def draw_frame(surface, game: GameState, flip=True):
    surface.fill(game.background_colour)

    draw_board(surface, game)
    draw_end_zones(surface, game)

    # ─── optional status banner ─────────────────────────────────────────
    if getattr(game, "status_message", ""):
        font  = pygame.font.SysFont(None, 28)
        txt   = font.render(game.status_message, True, (255, 255, 255))
        rect  = txt.get_rect()
        rect.centerx = surface.get_width() // 2      # horizontally centred
        rect.top     = 20                            # 20 px below window edge
        surface.blit(txt, rect)

    if flip:
        pygame.display.flip()