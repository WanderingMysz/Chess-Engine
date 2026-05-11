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

static inline uint8_t piecetype(uint8_t piece) {
    return (piece & PIECE_MASK);
}

static inline bool isNone(uint8_t piece) {
    return !(piece & PIECE_MASK);
}
//!SECTION

//SECTION - Set methods
static inline void setWhite(uint8_t* piece) {
    *piece = (*piece & ~(COLOR_MASK));
}

static inline void setBlack(uint8_t* piece) {
    *piece = (*piece | COLOR_MASK); 
}

static inline void setMoved(uint8_t* piece) {
    *piece = (*piece | MOVEMENT_MASK);
}

static inline void setUnmoved(uint8_t* piece) {
    *piece = (*piece & ~(MOVEMENT_MASK));
}

static inline void setPiece(uint8_t* piece, uint8_t piece_type) {
    *piece = ((*piece & ~(PIECE_MASK)) | piece_type);
}
//!SECTION

// SECTION - Index methods
// Returns the Index (0-63) of a given board coordinate, e.g. (a,1)
static int idx_from_char(char col, char row) {return col-'a' + (row-'1')*8;}
// Returns the Index (0-63) of a given board coordinate, e.g. (1,1)
static int idx_from_int(int col, int row) {return (col-1) + (row-1)*8;}

static int col_from_idx(int idx) {return idx % 8 + 1;}

static int row_from_idx(int idx) {return idx / 8 + 1;}
//!SECTION

#endif /* PIECE_INFO_H */