// Represents the state of a chessboard as 64 2-byte squares
#include <stdint.h>
#include <stdbool.h>
#include <stdio.h>

#define COL_BITS(col) (((col-1) & 0b00000111) << 3)
#define ROW_BITS(row) (row & 0b00000111)

#define BOARD_SIZE 64

// One-Hot encoding for piece types and colors
typedef enum {
    NONE        = 0b00000000,
    PAWN        = 0b00100000,
    KNIGHT      = 0b00010000,
    BISHOP      = 0b00001000,
    ROOK        = 0b00000100,
    QUEEN       = 0b00000010,
    KING        = 0b00000001,
} PieceType;

typedef enum {
    WHITE       = 0b00000000,
    BLACK       = 0b01000000
} PieceColor;

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

void set_square(Square *sq, uint8_t position, uint8_t piece) {
    sq->position = position;
    sq->piece = piece;
}

static void set_piece_color(Square *sq, uint8_t color) {

}

static void _set_pawns(Chessboard *board, int row) {
    for (int col = 1; col <= 8; col++) {
        uint8_t pos_col = COL_BITS(col);
        uint8_t pos_row = ROW_BITS(row);
        uint8_t pos = encode_position(pos_col, pos_row);

        uint8_t piece_type = PAWN;
        // printf("Setting pawn at col %d, row %d (IDX %d)\n", 
        //     col, row, idx_from_int(col, row));
        set_square(&board->squares[idx_from_int(col, row)], pos, piece_type);
    }
}

static void set_pawns(Chessboard *board) {
    _set_pawns(board, 2);
    _set_pawns(board, 7);
}

void set_colors(Chessboard *board) {
    for (int row = 1; row <= 8; row++) {
        if (2 < row && row < 7) continue;

        uint8_t color = (row <=2) ? WHITE : BLACK;
        for (int col = 1; col <= 8; col++) {
            // PLACEHOLDER
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

    return board;
}

void visualize_board_state(Chessboard *board, char* output) {
    for (int i = 0; i < BOARD_SIZE*2; i++) {
        // Insert newline after every row
        if (i % 16 == 15) {
            output[i] = '\n';
            continue;
        }
        // Populates pieces
        if (i % 2) {
            int sq_idx = i / 2;
            if (board->squares[sq_idx].piece & PAWN) output[i] = 'P';

            // Produces checkerboard # = # =
            else output[i] = ((sq_idx % 8 + sq_idx / 8) % 2) ? '#' : '=';

        } else {
            output[i] = ' ';
        }
    }
    output[BOARD_SIZE * 2] = '\0';
}

int main(void) {
    Chessboard board = initialize_chessboard();
    char board_state[BOARD_SIZE * 2]; // x2 so spaces can be added
    visualize_board_state(&board, &board_state[0]);
    printf("%s", board_state);
    return 0;
}