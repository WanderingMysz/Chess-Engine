#include "manipulate_board.h"
#include "move_validation.h"
#include "move_record.h"
#include "move_record_ops.h"
#include "types.h"

#include <stdio.h>
#include <regex.h>
#include <string.h>
#include <errno.h>
#include <stdbool.h>
#include <stdlib.h>

// TODO: Move regex compilation outside function scope so it only occurs once

static PlayerColor CurrentPlayer = WHITE;
static Chessboard SaveState;

void update_turn() { CurrentPlayer = (CurrentPlayer == WHITE) ? BLACK : WHITE; }

inline void set_square(Chessboard* board, int idx, Piece piece) {
    board->squares[idx] = piece;
}

static void _set_pawns(Chessboard *board, int rank) {
    for (int file = 1; file <= 8; file++) {
        set_square(board, idx_from_int(file, rank), PAWN);
    }
}

static void set_pawns(Chessboard *board) {
    _set_pawns(board, 2);
    _set_pawns(board, 7);
}

static void _set_pieces(Chessboard *board, int rank) {
    for (int file = 1; file <= 8; file++) {
        PieceType type;
        switch (file) {
            case 1:
            case 8:
                type = ROOK;
                break;
            case 2:
            case 7:
                type = KNIGHT;
                break;
            case 3:
            case 6:
                type = BISHOP;
                break;
            case 4:
                type = QUEEN;
                break;
            case 5:
                type = KING;
                break;
            default:
                type = NONE;
                break;
        }
        set_square(board, idx_from_int(file, rank), type);
    }
}

static void set_pieces(Chessboard *board) {
    _set_pieces(board, 1);
    _set_pieces(board, 8);
}

static void set_colors(Chessboard *board) {
    for (int rank = 1; rank <= 8; rank++) {
        PlayerColor color = (rank <= 4) ? WHITE : BLACK;

        for (int file = 1; file <= 8; file++) {
            int idx = idx_from_int(file, rank);
            set_color(&(board->squares[idx]), color);
        }
    }
}

// NOTE: May have case '' = PAWN and default = NONE
static PieceType type_from_letter(char letter) {
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

static int process_SAN(Chessboard* board, char* SAN_input,
                       Move_Record* move_record, char* src_coord) {
    int dest_idx = -1;
    bool capture = false;

    int left_idx = 0;
    int right_idx = strlen(SAN_input) - 1;

    // last two values excluding promotions and checks is always destination

    // Checks for Promotion
    char* promotion_ptr = strchr("NBRQ", SAN_input[right_idx]);
    if (promotion_ptr) {
        move_record->promotion = type_from_letter(*promotion_ptr);
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
    Piece dest_piece = board->squares[dest_idx];
    if (!cmp_piece_type(dest_piece, NONE) 
        && cmp_piece_color(dest_piece, CurrentPlayer)) {

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

    Piece piece;

    set_color(&piece, CurrentPlayer);
    move_record->color = CurrentPlayer;

    PieceType piece_type = type_from_letter(SAN_input[0]);
    set_piece(&piece, piece_type);
    move_record->piece_type = piece_type;

    // Extracts information from bracketed area: N [g6] xe5 
    if (!cmp_piece_type(piece, PAWN)) left_idx++;
    for (int i = 0; left_idx <= right_idx; i++) {
        if (2 <= i) {
            printf("Too many values for source idx. Aborting.\n");
            return ERR_NOTATION;
        }
        src_coord[i] = SAN_input[left_idx];
        left_idx++;
    }

    return 0;
}

int generate_record(Chessboard* board, char* SAN_input, 
                    Move_Record* move_record) {

    /* NOTE: Piece's color could be set after the comparison, but this is
       structured per the natural logic */
       
    char src_coord[3] = {'\0', '\0', '\0'};
    if (process_SAN(board, SAN_input, move_record, src_coord) != 0) {
        printf("Processing SAN failed!\n");
        return 1;
    }
    // If the entire source information is provided, validate then return
    int file = 0, rank = 0;
    switch (strlen(src_coord)) {
        case 2:

            printf("Two source coordinate parts given\n");

            file = file_from_char(src_coord[0]);
            rank = rank_from_char(src_coord[1]);
            break;
        case 1:
            ;
            char* ptr;

            printf("One source coordinate part given\n");

            ptr = strchr("abcdefgh", src_coord[0]);
            if (ptr) {
                file = file_from_char(*ptr);
                break;
            }
            
            ptr = strchr("12345678", src_coord[0]);
            if (ptr) {
                rank = rank_from_char(*ptr);
                break;
            }

            return 1;

        case 0:
            printf("No source coordinate part given\n");
            break;
        
        default:
            printf("ERROR with coordinates\n");
            return 1;
    }

    int ret_idx = locate_piece(board, move_record, file, rank);
    printf("Ret Idx: %d\n", ret_idx);
    return ret_idx;
}

static void find_name(PieceType piece_type, char* piece_name) {
    switch(piece_type) {
        case KNIGHT:
            strcpy(piece_name, "KNIGHT");
            break;
        case BISHOP:
            strcpy(piece_name, "BISHOP");
            break;
        case ROOK:
            strcpy(piece_name, "  ROOK");
            break;
        case QUEEN:
            strcpy(piece_name, " QUEEN");
            break;
        case KING:
            strcpy(piece_name, "  KING");
            break;
        case PAWN:
            strcpy(piece_name, "  PAWN");
            break;
        default:
            strcpy(piece_name, "      ");
            break;
    }
}

void print_record(Move_Record* move_record) {
    char color = (move_record->color == WHITE) ? 'w' : 'b';

    char piece_name[7];
    find_name(move_record->piece_type, piece_name);

    char promotion[7];
    find_name(move_record->promotion, promotion);

    int src_idx = move_record->src_idx;
    char src_file = file_from_idx(src_idx) + 'a' - 1;
    char src_rank = rank_from_idx(src_idx) + '0';

    int dest_idx = move_record->dest_idx;
    char dest_file = file_from_idx(dest_idx) + 'a' - 1;
    char dest_rank = rank_from_idx(dest_idx) + '0';

    char capture = (move_record->capture) ? 'x' : ' ';

    char checkmate_status = ' ';
    if (move_record->check) checkmate_status = '+';
    if (move_record->checkmate) checkmate_status = '#';

    printf("%05d %c %s %c%c %c %c%c %c %s\n",
        move_record->turn_number,
        color,
        piece_name,
        src_file,
        src_rank,
        capture,
        dest_file,
        dest_rank,
        checkmate_status,
        promotion
    );
}

Chessboard initialize_chessboard() {
    Chessboard board = {NONE};

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
    memcpy(&SaveState, board, sizeof(*board));

    // Make the move
    set_square(board, move->src_idx, NONE);

    PlayerColor color = move->color;
    Piece piece = move->piece_type;
    set_color(&piece, color);

    set_moved(&piece);
    set_square(board, move->dest_idx, piece);

    // Validate move legality
    if (!is_check(board, color)) {
        update_turn();
        return 0;
    }

    // Rollback change if necessary
    printf("Rolling back...\n");
    memcpy(board, &SaveState, sizeof(SaveState));
    return 1;
}