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
            # Draw final state *once more* to make sure last move is shown
            game_draw.draw_frame(screen, game)
            running = False
            # ──────────────────────────────────────────────────────────────
            # Wait for user confirmation before quitting
            # Accept: any key press, any mouse click, or window “close”
            # ──────────────────────────────────────────────────────────────
            waiting_for_ack = True
            while waiting_for_ack:
                for ev in pygame.event.get():
                    if ev.type in (pygame.KEYDOWN,
                                   pygame.MOUSEBUTTONDOWN,
                                   pygame.QUIT):
                        waiting_for_ack = False
                        break
                time.sleep(0.05)

        time.sleep(0.05)

    pygame.quit()


if __name__ == '__main__':
    game = game_state.GameState()
    game_loop(game)