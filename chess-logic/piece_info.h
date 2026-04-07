#ifndef PIECE_INFO_H
#define PIECE_INFO_H

#include "types.h"
#include <stdbool.h>

static inline bool isWhite(uint8_t piece) {
    return !(piece & COLOR_MASK);
}

static inline uint8_t encode_position(uint8_t col_bits, uint8_t row_bits) {
    return col_bits | row_bits;
}

// ASCII 'a' = 97
static int idx_from_char(char col, int row) {return col-97 + (row-1)*8;}
static int idx_from_int(int col, int row) {return (col-1) + (row-1)*8;}

#endif /* PIECE_INFO_H */