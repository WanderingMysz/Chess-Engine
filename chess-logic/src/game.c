// Represents the state of a chessboard as 64 2-byte squares
#include <stdbool.h>
#include <stdio.h>
#include <locale.h>
#include <string.h>
#include <ctype.h>
#include "render.h"
#include "manipulate_board.h"
#include "move_validation.h"
#include "move_record.h"
#include "move_record_ops.h"

#define streq(a, b) (strcmp((a), (b)) == 0)

// static bool EN_PASSANT_FLAG = false;

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
        char user_input[16];
        fgets(user_input, sizeof(user_input), stdin);
        user_input[strcspn(user_input, "\n")] = '\0'; //Remove newline character

        // char move[16];
        // for (int i = 0; i < 16; i++) {
        //     move[i] = toupper(typed_move[i]);
        // }

        if (streq(user_input, "quit") || streq(user_input, "exit")) {
            printf("Closing game. Thank you for playing!\n");
            game_running = false;
            break;
        }

        if (is_SAN(user_input)) {
            Move_Record move_record;
            if (generate_record(&board, user_input, &move_record) == 0) {
                // print_record(&move_record);
                if (make_move(&board, &move_record) == 0) continue;
                printf("King in check.\n");
            }
        }
        else {
            printf("Error processing SAN. ");
        }

        printf("\n%s is an invalid move. Try again\n\n", user_input);
    }
}