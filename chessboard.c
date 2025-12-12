// Represents the state of a chessboard as 64 2-byte squares
#include <stdint.h>
#include <stdbool.h>
#include <stdio.h>

// ASCII 'a' = 97
#define IDX_CHAR(col, row) (col-97 + (row-1)*8)
#define IDX_NUM(col, row) ((col-1) + (row-1)*8)

#define COL_BITS(col) (((col-1) & 0b00000111) << 3)
#define ROW_BITS(row) (row & 0b00000111)
#define POS_BYTE(col_bits, row_bits, vis) (col_bits | row_bits | vis)

const int BOARD_SIZE = 64;

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
    Square squares[64];
} Chessboard;

void set_square(Square *sq, uint8_t position, uint8_t piece) {
    sq->position = position;
    sq->piece = piece;
}

void set_pawns(Chessboard *board, char row) {
    for (int col = 1; col <= 8; col++) {
        uint8_t pos_col  = COL_BITS(col);
        uint8_t pos_row = ROW_BITS(row);
        uint8_t visibility = (row == 2) ? 0b10000000 : 0b01000000;
        uint8_t position = POS_BYTE(pos_col, pos_row, visibility);

        uint8_t piece_type = PAWN;
        piece_type |= (row == 2) ? WHITE : BLACK;
        printf("Setting pawn at col %d, row %d (IDX %d)\n", col, row, IDX_NUM(col, row));
        set_square(&board->squares[IDX_NUM(col, row)], position, piece_type);
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
    set_pawns(&board, 2); // White pawns
    set_pawns(&board, 7); // Black pawns

    return board;
}

void visualize_board_state(Chessboard *board, char* output) {
    int offset = 0; // Offset to account for newlines

    for (int i = 0; i < BOARD_SIZE + 8; i++) {
        // Insert newline after every row
        if (i % 9 == 8) {
            output[i] = '\n';
            offset++;
            continue;
        }
        output[i] = (board->squares[i - offset].piece & PAWN) ? 'P' : 
            (i % 2) ? '#' : ' '; // checkerboard pattern
    }
    output[BOARD_SIZE + 8] = '\0';
}

int main(void) {
    Chessboard board = initialize_chessboard();
    char board_state[BOARD_SIZE + 9];
    visualize_board_state(&board, &board_state[0]);
    printf("%s", board_state);
    return 0;
}