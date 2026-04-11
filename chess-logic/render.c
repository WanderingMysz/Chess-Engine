#include <string.h>
#include <strings.h>
#include "render.h"

const char* unicode_pieces[2][6] = {
    {"♚", "♛", "♜", "♝", "♞", "♟"}, // Black
    {"♔", "♕", "♖", "♗", "♘", "♙"}  // White
};

void visualize_board_state(Chessboard *board, char* output, bool flip_color) {
    output[0] = '\0'; // zeroes output string

    for (int row = 8; row >= 1; row--) {
        for (int col = 1; col <= 8; col++) {
            int sq_idx = idx_from_int(col, row);
            int8_t piece = board->squares[sq_idx];
            int8_t piece_type = piece & ~(COLOR_MASK | MOVEMENT_MASK);

            if (piece_type) {
                strcat(output, get_unicode(piece, flip_color));
            } else {
                // checkerboard pattern
                bool is_light_sq = (row + col) % 2;
                strcat(output, is_light_sq ? "□" : "▪");
            }

            strcat(output, " ");
        }
        strcat(output, "\n");
    }
}

const char* get_unicode(PieceType piece, bool flip_color) {
    int type_idx = ffs(piece) - 1;
    int color_idx = piece & COLOR_MASK ? 0 : 1;

    if (flip_color) color_idx = !color_idx;
    return unicode_pieces[color_idx][type_idx];
}