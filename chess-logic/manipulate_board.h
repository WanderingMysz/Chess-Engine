#ifndef MANIPULATE_BOARD_H
#define MANIPULATE_BOARD_H

#include "types.h"
#include "piece_info.h"

/* Creates Chessboard */
Chessboard initialize_chessboard();

/* Updates Square information */
void set_square(Square *sq, uint8_t position, uint8_t piece);

/* Removes pieces from every square of the board */
void clear_board(Chessboard *board);

#endif /* MANIPULATE_BOARD_H */
