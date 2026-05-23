#ifndef RENDER_H
#define RENDER_H

#include "types.h"
#include <stdbool.h>

/* Takes a board representation and outputs the Unicode representation */
void visualize_board_state(Chessboard *board, char* output, bool flip_color);

/* Flips board state to represent opposing player's view */
void flip_viewpoint(Chessboard *board, char* output);

/* Inverts piece color definitions, i.e. for dark mode coherency */
void invert_colors(Chessboard *board, char* output);

/* Returns Unicode representation of piece */
const char* get_unicode(PieceType piece, bool flip_color);

#endif /* RENDER_H */