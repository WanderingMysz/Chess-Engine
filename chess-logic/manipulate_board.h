#ifndef MANIPULATE_BOARD_H
#define MANIPULATE_BOARD_H

#include "types.h"
#include "piece_info.h"
#include "record.h"

/* Creates Chessboard */
Chessboard initialize_chessboard();

/* Creates empty Chessboard */
Chessboard initialize_empty_chessboard();

/* Updates Square information */
void set_square(Chessboard *board, int idx, uint8_t piece);

/* Removes pieces from every square of the board */
void clear_board(Chessboard *board);

/* Converts SAN input to move record */
int get_move_info(Chessboard* board, char* SAN_input, Move_Record* move_record);

/* Attempts to make the given move. Returns 1 if the move is illegal */
int make_move(Chessboard *board, Move_Record* move);

#endif /* MANIPULATE_BOARD_H */
