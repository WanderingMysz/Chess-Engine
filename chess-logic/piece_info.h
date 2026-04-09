#ifndef PIECE_INFO_H
#define PIECE_INFO_H

#include "types.h"
#include <stdbool.h>

//SECTION - Check methods
static inline bool isWhite(uint8_t piece) {
    return !(piece & COLOR_MASK);
}

static inline bool hasMoved(uint8_t piece) {
    return (piece & MOVEMENT_MASK);
}

static inline bool isPiece(uint8_t piece) {
    return (piece & PIECE_MASK);
}
//!SECTION

//SECTION - Set methods
static inline bool setWhite(uint8_t* piece) {
    return (*piece & !(COLOR_MASK));
}

static inline bool setBlack(uint8_t* piece) {
    return (*piece | MOVEMENT_MASK); 
}

static inline bool setMoved(uint8_t* piece) {
    return (*piece | MOVEMENT_MASK);
}

static inline bool setUnmoved(uint8_t* piece) {
    return (*piece & !(MOVEMENT_MASK));
}

static inline bool setPiece(uint8_t* piece, uint8_t piece_type) {
    return (*piece & (((PIECE_MASK) & piece_type) | !(PIECE_MASK)));
}
//!SECTION

// Returns the Index (0-63) of a given board coordinate, e.g. (a,1)
static int idx_from_char(char col, int row) {return col-97 + (row-1)*8;}
// Returns the Index (0-63) of a given board coordinate, e.g. (1,1)
static int idx_from_int(int col, int row) {return (col-1) + (row-1)*8;}

#endif /* PIECE_INFO_H */