// Represents the state of a chessboard as 64 2-byte squares
#include <stdint.h>
#include <stdbool.h>
#include <stdio.h>
#include <locale.h>
#include <string.h>

#define COL_BITS(col) (((col-1) & 0b00000111) << 3)
#define ROW_BITS(row) ((row-1) & 0b00000111)

#define COLOR_MASK (1 << 7)
#define MOVEMENT_MASK (1 << 6)

#define BOARD_SIZE 64

#define UNICODE_BYTES 3

// One-Hot encoding for piece types and colors
typedef enum {
    NONE        = 0,
    PAWN        = (1 << 5),
    KNIGHT      = (1 << 4),
    BISHOP      = (1 << 3),
    ROOK        = (1 << 2),
    QUEEN       = (1 << 1),
    KING        = 1,
} PieceType;

const char* unicode_pieces[2][6] = {
    {"♚", "♛", "♜", "♝", "♞", "♟"}, // Black
    {"♔", "♕", "♖", "♗", "♘", "♙"}  // White
};

// Converts piece one-hot to unicode character
const char* get_unicode(PieceType piece, bool flip_color) {
    int type_idx = ffs(piece) - 1;
    int color_idx = piece & COLOR_MASK ? 0 : 1;

    if (flip_color) color_idx = !color_idx;
    return unicode_pieces[color_idx][type_idx];
}

typedef enum {
    WHITE       = 0,
    BLACK       = COLOR_MASK
} PieceColor;

typedef enum {
    UNMOVED     = 0,
    MOVED       = MOVEMENT_MASK
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

// ASCII 'a' = 97
static int idx_from_char(char col, int row) {return col-97 + (row-1)*8;}
static int idx_from_int(int col, int row) {return (col-1) + (row-1)*8;}

static inline uint8_t encode_position(uint8_t col_bits, uint8_t row_bits) {
    return col_bits | row_bits;
}

// static uint8_t encode_position_from_idx(int idx) {
//     uint8_t col_bits = COL_BITS(idx % 8);
//     uint8_t row_bits = ROW_BITS(idx / 8);

//     return encode_position(col_bits, row_bits);
// }

void set_square(Square *sq, uint8_t position, uint8_t piece) {
    sq->position = position;
    sq->piece = piece;
}

static void set_piece_color(Square *sq, uint8_t color) {
    sq->piece &= ~(COLOR_MASK);
    sq->piece |= color;
}

static void _set_pawns(Chessboard *board, int row) {
    for (int col = 1; col <= 8; col++) {
        uint8_t pos_col = COL_BITS(col);
        uint8_t pos_row = ROW_BITS(row);
        uint8_t pos = encode_position(pos_col, pos_row);

        uint8_t piece_type = PAWN;

        set_square(&board->squares[idx_from_int(col, row)], pos, piece_type);
    }
}

static void set_pawns(Chessboard *board) {
    _set_pawns(board, 2);
    _set_pawns(board, 7);
}

static void _set_pieces(Chessboard *board, int row) {
    for (int col = 1; col <= 8; col++) {
        uint8_t pos_col = COL_BITS(col);
        uint8_t pos_row = ROW_BITS(row);
        uint8_t pos = encode_position(pos_col, pos_row);
        
        uint8_t piece_type;
        switch (col) {
            case 1:
            case 8:
                piece_type = ROOK;
                break;
            case 2:
            case 7:
                piece_type = KNIGHT;
                break;
            case 3:
            case 6:
                piece_type = BISHOP;
                break;
            case 4:
                piece_type = KING;
                break;
            case 5:
                piece_type = QUEEN;
                break;
            default:
                piece_type = NONE;
                break;
        }
        set_square(&board->squares[idx_from_int(col, row)], pos, piece_type);
    }
}

static void set_pieces(Chessboard *board) {
    _set_pieces(board, 1);
    _set_pieces(board, 8);
}

void set_colors(Chessboard *board) {
    for (int row = 1; row <= 8; row++) {
        uint8_t color = (row <= 4) ? WHITE : BLACK;

        for (int col = 1; col <= 8; col++) {
            int idx = idx_from_int(col, row);
            uint8_t pos = encode_position(COL_BITS(col), ROW_BITS(row));

            Square* curr_square = &board->squares[idx];
            curr_square->position |= pos;
            set_piece_color(curr_square, color);
        }
    }
}

void clear_board(Chessboard *board) {
    for (int i = 0; i < BOARD_SIZE; i++) {
        set_square(&board->squares[i], 0, NONE);
    }
}

Chessboard initialize_chessboard() {
    Chessboard board = {0};

    // Initialize the chessboard with default values
    set_pawns(&board);
    set_pieces(&board);
    set_colors(&board);

    return board;
}
static inline bool isWhite(uint8_t piece) {
    return !(piece & COLOR_MASK);
}

void visualize_board_state(Chessboard *board, char* output, bool flip_color) {
    output[0] = '\0'; // zeroes output string

    for (int row = 8; row >= 1; row--) {
        for (int col = 1; col <= 8; col++) {
            int sq_idx = idx_from_int(col, row);
            int8_t piece = board->squares[sq_idx].piece;
            int8_t piece_type = piece & ~(COLOR_MASK | MOVEMENT_MASK);

            if (piece_type) {
                strcat(output, get_unicode(piece, flip_color));
            } else {
                // checkerboard pattern
                bool is_light_sq = (row + col) % 2;
                strcat(output, is_light_sq ? "□" : "▪");
            }

            strcat(output, " ");
        }
        strcat(output, "\n");
    }
}

int main(void) {
    setlocale(LC_ALL, ""); // sets default encoding method, presumably UTF-8
    Chessboard board = initialize_chessboard();
    char board_state[BOARD_SIZE * (UNICODE_BYTES + 2)]; //unicode bytes + buffer
    visualize_board_state(&board, &board_state[0], false);
    printf("%s", board_state);
    return 0;
}