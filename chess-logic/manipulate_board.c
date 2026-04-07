#include "manipulate_board.h"
#include <stdio.h>

void set_square(Square *sq, uint8_t position, uint8_t piece) {
    sq->position = position;
    sq->piece = piece;
}
/* Sets the color of a given square's piece */
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

Chessboard initialize_chessboard() {
    Chessboard board = {0};

    // Initialize the chessboard with default values
    set_pawns(&board);
    set_pieces(&board);
    set_colors(&board);

    return board;
}

void clear_board(Chessboard *board) {
    for (int i = 0; i < BOARD_SIZE; i++) {
        set_square(&board->squares[i], 0, NONE);
    }
}