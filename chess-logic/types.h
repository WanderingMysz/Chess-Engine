#ifndef TYPES_H
#define TYPES_H

#include <stdint.h>

#define COL_BITS(col) (((col-1) & 0b00000111) << 3)
#define ROW_BITS(row) ((row-1) & 0b00000111)

#define COLOR_MASK    (1 << 7)
#define MOVEMENT_MASK (1 << 6)

#define BOARD_SIZE    64
#define UNICODE_BYTES 3

// One-Hot encoding for piece types
typedef enum {
    NONE   = 0,
    PAWN   = (1 << 5),
    KNIGHT = (1 << 4),
    BISHOP = (1 << 3),
    ROOK   = (1 << 2),
    QUEEN  = (1 << 1),
    KING   = 1,
} PieceType;

typedef enum {
    WHITE = 0,
    BLACK = COLOR_MASK
} PieceColor;

typedef enum {
    UNMOVED = 0,
    MOVED   = MOVEMENT_MASK
} PieceMoved;

typedef struct {
    // Two highest bits denote square visibility
    // Remaining six bits denote the position on the board
    uint8_t position;
    // Two highest bits denote color and whether the piece has moved
    // Remaining six bits denote the piece type
    uint8_t piece;
} Square;

typedef struct {
    Square squares[BOARD_SIZE];
} Chessboard;

extern const char* unicode_pieces[2][6];

#endif /* TYPES_H */
