#include <string.h>
#include <strings.h>
#include "render.h"
#include "types.h"

const char* UNICODE_REPRESENTATION[2][6] = {
    {"♔", "♕", "♖", "♗", "♘", "♙"},  // White
    {"♚", "♛", "♜", "♝", "♞", "♟"} // Black
};

// "□" : "▪"

void visualize_board_state(Chessboard *board, char* output, bool flip_color) {
    output[0] = '\0'; // zeroes output string

    for (int rank = 8; rank >= 1; rank--) {
        for (int file = 1; file <= 8; file++) {
            int sq_idx = idx_from_int(file, rank);
            Piece piece = board->squares[sq_idx];

            if (!cmp_piece_type(piece, NONE)) {
                strcat(output, get_unicode(piece, flip_color));
            } else {
                // checkerboard pattern
                bool is_light_sq = (rank + file) % 2;
                strcat(output, is_light_sq ? " " : "▪");
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
    return UNICODE_REPRESENTATION[color_idx][type_idx];
}