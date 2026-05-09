// Represents the state of a chessboard as 64 2-byte squares
#include <stdbool.h>
#include <stdio.h>
#include <locale.h>
#include <string.h>
#include <ctype.h>
#include "render.h"
#include "piece_info.h"
#include "manipulate_board.h"
#include "move_validation.h"
#include "record.h"

#define streq(a, b) (strcmp((a), (b)) == 0)

static bool EN_PASSANT_FLAG = false;

int main(void) {
    setlocale(LC_ALL, ""); // sets default encoding method, presumably UTF-8
    Chessboard board = initialize_chessboard();
    char board_state[BOARD_SIZE * (UNICODE_BYTES + 2)];// unicode bytes + buffer

    // Game loop
    bool game_running = true;
    while (game_running) {
        visualize_board_state(&board, &board_state[0], false);
        printf("%s", board_state);

        printf("Enter move:\n> ");
        char typed_move[16];
        fgets(typed_move, sizeof(typed_move), stdin);
        typed_move[strcspn(typed_move, "\n")] = '\0'; // Remove newline character

        char move[16];
        for (int i = 0; i < 16; i++) {
            move[i] = toupper(typed_move[i]);
        }

        if (streq(move, "QUIT") || streq(move, "EXIT")) {
            printf("Closing game. Thank you for playing!\n");
            game_running = false;
            break;
        }

        if (is_SAN(move)) {
            Move_Record move_record;
            if (get_move_info(&board, move, &move_record) == 0) {
                make_move(&board, &move_record);
                
                continue;
            }
        }
        printf("\n%s is an invalid move. Try again\n\n", typed_move);
    }
}