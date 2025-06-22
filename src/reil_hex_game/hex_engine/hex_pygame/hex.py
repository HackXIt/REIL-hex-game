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
            pygame.display.flip()  # ensure it's shown
            break  # exit main loop

        time.sleep(0.05)

    # Keep final screen visible until window closed or ESC pressed
    print("Game finished. Close the window or press ESC to exit.")
    waiting = True
    while waiting:
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (
                event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                waiting = False

    pygame.quit()



if __name__ == '__main__':
    game = game_state.GameState()
    game_loop(game)