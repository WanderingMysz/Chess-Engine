#ifndef MOVE_VALIDATION_H
#define MOVE_VALIDATION_H

// TODO: Define methods for move_validation
#include <stdbool.h>
#include "types.h"

/* Returns if the given player's king can legally castle */
bool can_castle(Chessboard *board, bool white, bool kingside);

/* Returns if the given player is in check. */
bool is_check(Chessboard *board, bool white);

/* Returns if the given player is checkmated */
bool is_checkmate(Chessboard *board, bool white);

/* Returns if the move is in standard algebraic notation (SAN) */
bool is_SAN(char* move);

/* Returns if the piece exists at the specified square */
bool piece_exists(Chessboard* board, int idx, uint8_t piece);

/* Returns nearest piece, with col and row as optional restrictors */
int locate_piece(Chessboard* board, uint8_t piece, int dest_idx, 
                int col, int row, bool capture);

bool cmp_piece_type(uint8_t piece, PieceType comp);

bool cmp_piece_color(uint8_t piece, bool color);

#endif /* MOVE_VALIDATION_H */