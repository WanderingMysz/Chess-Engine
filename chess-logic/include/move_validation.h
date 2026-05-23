#ifndef MOVE_VALIDATION_H
#define MOVE_VALIDATION_H

#include <stdbool.h>
#include "types.h"
#include "move_record.h"

/* ----------------------------- Input Handling ----------------------------- */

/* Returns if the move is in standard algebraic notation (SAN) */
bool is_SAN(char* move);

/* ------------------------------- Board State ------------------------------ */

/* Returns if the given player's king can legally castle in a given direction */
bool can_castle(Chessboard *board, PlayerColor color, Direction direction);

/* Returns if the given player is in check. */
bool is_check(Chessboard *board, PlayerColor color);

/* Returns if the given player is checkmated */
bool is_checkmate(Chessboard *board, PlayerColor color);

/* ----------------------------- Piece Locating ----------------------------- */

/* Returns if the piece exists at the specified square */
bool piece_exists(Chessboard* board, int idx, Piece piece);

/* Returns piece matching move record, with file and rank as optional 
   restrictors */
int locate_piece(Chessboard* board, Move_Record* move_record, 
                 int file, int rank);

#endif /* MOVE_VALIDATION_H */