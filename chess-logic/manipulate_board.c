#include "manipulate_board.h"
#include "move_validation.h"
#include "record.h"
#include "types.h"
#include "piece_info.h"
#include "errors.h"
#include <stdio.h>
#include <regex.h>
#include <string.h>
#include <errno.h>
#include <stdbool.h>
#include <stdlib.h>

// TODO: Move regex compilation outside function scope so it only occurs once
static bool wh_turn = true;
static Chessboard board_state;

void update_turn() {
    wh_turn = !wh_turn;
}

void set_square(Chessboard* board, int idx, uint8_t piece) {
    board->squares[idx] = piece;
}

// /* Sets the color of a given square's piece */
// static void set_piece_color(uint8_t* piece, uint8_t color) {
//     *piece &= ~(COLOR_MASK);
//     *piece |= color;
// }

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
                piece_type = QUEEN;
                break;
            case 5:
                piece_type = KING;
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
            (color == WHITE)? setWhite(&(board->squares[idx])) 
                            : setBlack(&(board->squares[idx]));
        }
    }
}

static uint8_t _get_piece_type(char letter) {
    switch (letter) {
        case 'N':
            return KNIGHT;
        case 'B':
            return BISHOP;
        case 'R':
            return ROOK;
        case 'Q':
            return QUEEN;
        case 'K':
            return KING;
        default:
            return PAWN;
    }
}

int get_move_info(Chessboard* board, char* SAN_input, Move_Record* move_record) {
    int src_idx = -1, dest_idx = -1;
    uint8_t piece;
    bool capture = false;

    int left_idx = 0;
    int right_idx = strlen(SAN_input) - 1;

    // last two values excluding promotions and checks is always destination

    // Checks for Promotion
    char* ptr = strchr("NBRQ", SAN_input[right_idx]);
    if (ptr) {
        move_record->promotion = _get_piece_type(*ptr);
        right_idx -= 2;
    }

    // Checks for Check(mate)
    switch (SAN_input[right_idx]) {
        // Check / Checkmate
        case '#':
            move_record->checkmate = true;
        case '+':
            move_record->check = true;
            right_idx--;
            break;
        // Standard move
        default:
            break;
    }
    dest_idx = idx_from_char(SAN_input[right_idx-1],SAN_input[right_idx]);

    // Cannot capture yourself
    uint8_t dest_piece = board->squares[dest_idx];
    if (!cmp_piece_type(dest_piece, NONE) && (isWhite(dest_piece) == wh_turn)) {
        printf("Cannot capture your own piece.\n");
        return 1;
    }

    move_record->dest_idx = dest_idx;
    right_idx-= 2;

    if (SAN_input[right_idx] == 'x') {
        right_idx--;
        capture = true;
    }
    move_record->capture = capture;

    // pieces default to white
    piece = _get_piece_type(SAN_input[0]);

    if (!wh_turn) {
        setBlack(&piece);
        move_record->color = BLACK;
    } else {
        move_record->color = WHITE;
    }

    move_record->piece_type = piece;

    /* NOTE: Piece's color could be set after the comparison, but this is
       structured per the natural logic */
       
    // Extracts information from bracketed area: N [g6] xe5 
    char src_info[3] = {'\0', '\0', '\0'};
    if (!cmp_piece_type(piece, PAWN)) left_idx++;
    for (int i = 0; left_idx <= right_idx; i++) {
        if (2 <= i) {
            printf("Too many values for source idx. Aborting.\n");
            return 1; // TODO: Make dict to enumerate error codes
        }
        src_info[i] = SAN_input[left_idx];
        left_idx++;
    }

    // If the entire source information is provided, validate then return
    switch (strlen(src_info)) {
        case 2:
            src_idx = locate_piece(board, piece, dest_idx, 
                                    src_info[0], src_info[1], capture);
            break;
        case 1:
            ;
            char* ptr;

            ptr = strchr("abcdefgh", src_info[0]);
            if (ptr) {
                int col = *ptr - 'a' + 1;
                src_idx = locate_piece(board, piece, dest_idx, col, 0, capture);
                break;
            }
            
            ptr = strchr("12345678", src_info[0]);
            if (ptr) {
                int row = *ptr - '0';
                src_idx = locate_piece(board, piece, dest_idx, 0, row, capture);
                break;
            }

            return 1;

        case 0:
            src_idx = locate_piece(board, piece, dest_idx, 0, 0, capture);
            break;
        
        default:
            return 1;
    }

    // Error codes all negative
    if (src_idx < 0) {
        printf("ERROR: %d\n", src_idx);
        return 1;
    }

    // TODO: Add descriptive error codes
    move_record->src_idx = src_idx;
    if (piece_exists(board, src_idx, piece)) {
        return 0;
    }
    printf("Piece %d @ %d does not exist.\n", (int)piece, src_idx);
    return 1;
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

int make_move(Chessboard *board, Move_Record* move) {
    // Store relevant information to rollback move if necessary
    memcpy(&board_state, board, sizeof(*board));

    // Make the move
    set_square(board, move->src_idx, NONE);
    int color = move->color;

    uint8_t piece = move->piece_type;
    (color == WHITE) ? setWhite(&piece) : setBlack(&piece);
    setMoved(&piece);

    set_square(board, move->dest_idx, piece);

    // Validate move legality
    if (!is_check(board, color==WHITE)) {
        update_turn();
        return 0;
    }

    printf("Rolling back...\n");

    // Rollback change if necessary
    memcpy(board, &board_state, sizeof(board_state));
    return 1;
}