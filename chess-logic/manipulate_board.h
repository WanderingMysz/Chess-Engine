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

int get_move_info(Chessboard* board, char* move_str, Move_Record* move_record);

void make_move(Chessboard *board, Move_Record* move);

#endif /* MANIPULATE_BOARD_H */
