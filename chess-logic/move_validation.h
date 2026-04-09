#ifndef MOVE_VALIDATION_H
#define MOVE_VALIDATION_H

// TODO: Define methods for move_validation
#include <stdbool.h>
#include "types.h"

/* Returns if the given player's king can legally castle */
bool can_castle(Chessboard *board, bool white, bool kingside);

/* Returns if the given player is in check */
bool is_in_check(Chessboard *board, bool white);

/* Returns if the given player is checkmated */
bool is_checkmate(Chessboard *board, bool white);

/* Returns if the move is in standard algebraic notation (SAN) */
bool is_SAN(char* move);

#endif /* MOVE_VALIDATION_H */