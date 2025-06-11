# Credits: https://github.com/parappayo/hex-py/blob/master/game_input.py
# License: (MIT License) https://github.com/parappayo/hex-py/blob/master/LICENSE
# Author: Parappayo
# Modified by: HackXIt

import sys, pygame
from .game_state import GameState

def on_quit(event, game: GameState):
    game.engine.close() if getattr(game, "engine", None) else sys.exit()

def on_key_down(event, game):
    if event.key == pygame.K_ESCAPE:
        on_quit(event, game)
    if event.key == pygame.K_SPACE and getattr(game, "step_event", None) and not getattr(game, "auto_mode", False):
        game.step_event.set()
    if event.key == pygame.K_RETURN:
        if game.auto_mode:
            game.auto_mode = False
            game.status_message = "Press SPACE to advance"
        else:
            game.auto_mode = True
            game.status_message = "Auto-play - press ENTER to pause"
            # wake the engine so it doesn't wait for a SPACE that will never come
            if getattr(game, "step_event", None):
                game.step_event.set()


def on_mouse_down(event, game):
    if getattr(game, "step_event", None):
        return  # Do not handle mouse clicks if step_event is set
    if event.button == 1 and game.is_valid_move():
        #  When a hexPosition engine is attached, delegate the move to it
        #  so that the logical board and the GUI stay in perfect sync.
        if getattr(game, "engine", None):
            game.engine.move(game.nearest_tile_to_mouse.grid_position)
        else:
            game.take_move()
    return


def on_mouse_up(event, game):
    return


def on_mouse_move(event, game):
    if getattr(game, "step_event", None):
        return  # Do not handle mouse move if step_event is set
    game.nearest_tile_to_mouse = game.nearest_hex_tile(event.pos)
    return


event_handlers = {
    pygame.QUIT: on_quit,
    pygame.KEYDOWN: on_key_down,
    pygame.MOUSEBUTTONDOWN: on_mouse_down,
    pygame.MOUSEBUTTONUP: on_mouse_up,
    pygame.MOUSEMOTION: on_mouse_move
}


def handle_events(events, game):
    for event in events:
        if not event.type in event_handlers:
            continue
        event_handlers[event.type](event, game)