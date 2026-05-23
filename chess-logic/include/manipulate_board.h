#ifndef MANIPULATE_BOARD_H
#define MANIPULATE_BOARD_H

#include "types.h"
#include "move_record.h"
#include "move_record_ops.h"

/* Creates Chessboard with standard starting pieces */
Chessboard initialize_chessboard();

/* Creates empty Chessboard */
Chessboard initialize_empty_chessboard();

/* Updates Square information */
void set_square(Chessboard *board, int idx, Piece piece);

/* Removes pieces from every square of the board */
void clear_board(Chessboard *board);

/* Attempts to make the given move. Returns 0 if the move is legal. */
int make_move(Chessboard *board, Move_Record* move);

#endif /* MANIPULATE_BOARD_H */
