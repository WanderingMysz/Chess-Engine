#ifndef TYPES_H
#define TYPES_H

#include <stdint.h>
#include <stdbool.h>

/* -------------------------------------------------------------------------- */
/*                              Type Definitions                              */
/* -------------------------------------------------------------------------- */

#define PLAYER_COUNT        2
#define UNIQUE_PIECE_TYPES  6

typedef uint16_t    TurnNumber;
typedef uint8_t     Piece;

/* ----------------------------- Piece Encoding ----------------------------- */

#define FILE_BITS(file) (((file-1) & 0b00000111) << 3)
#define RANK_BITS(rank) ((rank-1) & 0b00000111)

#define COLOR_MASK      (1 << 7)
#define MOVEMENT_MASK   (1 << 6)
#define PIECE_MASK      ~(COLOR_MASK | MOVEMENT_MASK)

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

// typedef enum {
//     PIECE_WHITE = 0,
//     PIECE_BLACK = COLOR_MASK
// } PieceColor;

typedef enum {
    UNMOVED = 0,
    MOVED   = MOVEMENT_MASK
} MovementFlag;

/* ----------------------------- Indexing Values ---------------------------- */

typedef enum {
    PAWN_ID,
    KNIGHT_ID,
    BISHOP_ID,
    ROOK_ID,
    QUEEN_ID,
    KING_ID,
    NONE_ID
} PieceID;

typedef enum {
    WHITE,
    BLACK
} PlayerColor;

/* ------------------------------- Chessboard ------------------------------- */

#define BOARD_SIZE      64
#define UNICODE_BYTES   3

// An array of 64 piece values
typedef struct {
    uint8_t squares[BOARD_SIZE];
} Chessboard;

typedef enum {
    QUEENSIDE,
    KINGSIDE
} Direction;

extern const char* UNICODE_REPRESENTATION[PLAYER_COUNT][UNIQUE_PIECE_TYPES];

/* -------------------------------------------------------------------------- */
/*                                 Error Codes                                */
/* -------------------------------------------------------------------------- */

typedef enum {
    ERR_NONE_FOUND      = -1,
    ERR_MULTIPLE_FOUND  = -2,
    ERR_NOTATION        = -3,
    ERR_OTHER           = -4
} ValidationError;

typedef enum {
    ERR_CHECK           = -1,
    ERR_ILLEGAL_MOVE    = -2
} MovementError;

/* -------------------------------------------------------------------------- */
/*                               Inline Methods                               */
/* -------------------------------------------------------------------------- */

/* ------------------------------ Check Methods ----------------------------- */
static inline bool is_white(Piece piece) {
    return !(piece & COLOR_MASK);
}

static inline bool is_black(Piece piece) {
    return (piece & COLOR_MASK);
}

static inline bool has_moved(Piece piece) {
    return (piece & MOVEMENT_MASK);
}

static inline bool is_none(Piece piece) {
    return !(piece & PIECE_MASK);
}

/* --------------------------- Comparison Methods --------------------------- */

static inline bool cmp_piece_type(Piece piece, PieceType cmp_piece) {
    return ((piece & PIECE_MASK) & cmp_piece);
}

static inline bool cmp_piece_color(Piece piece, PlayerColor cmp_color) {
    return ((piece & COLOR_MASK) 
            & ((cmp_color == WHITE) ? ~(COLOR_MASK) : COLOR_MASK));
}

static inline bool cmp_pieces(Piece piece1, Piece piece2) {
    return ((piece1 | MOVEMENT_MASK) & (piece2 | MOVEMENT_MASK));
}

/* ------------------------------- Get Methods ------------------------------ */

static inline PieceType get_piece_type(Piece piece) {
    return (piece & PIECE_MASK);
}

static inline PlayerColor opposite_color(PlayerColor color) {
    return ((color == WHITE) ? BLACK : WHITE);
}

/* ------------------------------- Set Methods ------------------------------ */
static inline void set_white(Piece* piece) {
    *piece = (*piece & ~(COLOR_MASK));
}

static inline void set_black(Piece* piece) {
    *piece = (*piece | COLOR_MASK); 
}

static inline void set_color(Piece* piece, PlayerColor color) {
    *piece = ((*piece & ~(COLOR_MASK)) 
              | ((color == WHITE) ? ~(COLOR_MASK) : COLOR_MASK));
}

static inline void set_moved(Piece* piece) {
    *piece = (*piece | MOVEMENT_MASK);
}

static inline void set_unmoved(Piece* piece) {
    *piece = (*piece & ~(MOVEMENT_MASK));
}

static inline void set_piece(Piece* piece, PieceType piece_type) {
    *piece = ((*piece & ~(PIECE_MASK)) | piece_type);
}

/* ------------------------------ Index Methods ----------------------------- */

// Returns the Index (0-63) of a given board coordinate, e.g. (a,1)
static inline int idx_from_char(char file, char rank) {
    return (file-'a' + (rank-'1')*8);
}
// Returns the Index (0-63) of a given board coordinate, e.g. (1,1)
static inline int idx_from_int(int file, int rank) {
    return ((file-1) + (rank-1)*8);
}

static inline int file_from_idx(int idx) {
    return (idx % 8 + 1);
}

static inline int rank_from_idx(int idx) {
    return (idx / 8 + 1);
}

static inline int file_from_char(char file) {
    return (file - 'a' + 1);
}
static inline int rank_from_char(char rank) {
    return (rank - '1');
}

static inline bool valid_index(int idx) {
    return (0 <= idx && idx < BOARD_SIZE);
}

#endif /* TYPES_H */
