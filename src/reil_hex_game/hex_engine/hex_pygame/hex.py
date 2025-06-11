# Credits: https://github.com/parappayo/hex-py/blob/master/hex.py
# License: (MIT License) https://github.com/parappayo/hex-py/blob/master/LICENSE
# Author: Parappayo
# Modified by: HackXIt

import sys, time, pygame
from . import game_state, game_input, game_draw


def game_loop(game):
    pygame.init()
    screen = pygame.display.set_mode(game.screen_size)
    running = True
    while running:
        game_input.handle_events(pygame.event.get(), game)
        game_draw.draw_frame(screen, game)
        sys.stdout.flush()
        if getattr(game, "shutdown_event", None) and game.shutdown_event.is_set():
            running = False
        time.sleep(0.05) # cap at 20 fps

if __name__ == '__main__':
    game = game_state.GameState()
    game_loop(game)