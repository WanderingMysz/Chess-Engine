#include "manipulate_board.h"
#include "move_validation.h"
#include "record.h"
#include <stdio.h>
#include <regex.h>
#include <string.h>
#include <errno.h>
#include <stdbool.h>

static bool wh_turn = true;

void update_turn() {
    wh_turn = !wh_turn;
}

void set_square(Chessboard* board, int idx, uint8_t piece) {
    board->squares[idx] = piece;
}

/* Sets the color of a given square's piece */
static void set_piece_color(uint8_t* piece, uint8_t color) {
    *piece &= ~(COLOR_MASK);
    *piece |= color;
}

static void _set_pawns(Chessboard *board, int row) {
    for (int col = 1; col <= 8; col++) {
        uint8_t piece_type = PAWN;
        set_square(board, idx_from_int(col, row), piece_type);
    }
}

static void set_pawns(Chessboard *board) {
    _set_pawns(board, 2);
    _set_pawns(board, 7);
}

static void _set_pieces(Chessboard *board, int row) {
    for (int col = 1; col <= 8; col++) {
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
        set_square(board, idx_from_int(col, row), piece_type);
    }
}

static void set_pieces(Chessboard *board) {
    _set_pieces(board, 1);
    _set_pieces(board, 8);
}

static void set_colors(Chessboard *board) {
    for (int row = 1; row <= 8; row++) {
        uint8_t color = (row <= 4) ? WHITE : BLACK;

        for (int col = 1; col <= 8; col++) {
            int idx = idx_from_int(col, row);
            set_piece_color(&(board->squares[idx]), color);
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

Chessboard initialize_empty_chessboard() {
    Chessboard board = {0};
    return board;
}

void clear_board(Chessboard *board) {
    for (int i = 0; i < BOARD_SIZE; i++) {
        set_square(board, i, NONE);
    }
}