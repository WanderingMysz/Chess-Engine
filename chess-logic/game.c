// Represents the state of a chessboard as 64 2-byte squares
#include <stdbool.h>
#include <stdio.h>
#include <locale.h>
#include <string.h>
#include "render.h"
#include "piece_info.h"
#include "manipulate_board.h"
#include "move_validation.h"

int main(void) {
    setlocale(LC_ALL, ""); // sets default encoding method, presumably UTF-8
    Chessboard board = initialize_chessboard();
    char board_state[BOARD_SIZE * (UNICODE_BYTES + 2)];// unicode bytes + buffer
    visualize_board_state(&board, &board_state[0], false);
    printf("%s", board_state);

    // Game loop
    bool game_running = true;
    while (game_running) {
        printf("Enter move:\n> ");
        char move[16];
        fgets(move, sizeof(move), stdin);
        move[strcspn(move, "\n")] = '\0'; // Remove newline character

        if (strcasecmp(move, "quit") == 0) {
            game_running = 0;
            break;
        }

        if (is_valid_move(move)) {
            printf("%s is a valid move.\n", move);
        } else {
            printf("%s is an invalid move.\n", move);
        }
    }
}