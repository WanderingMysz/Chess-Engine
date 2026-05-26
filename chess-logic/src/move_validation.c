#include "move_validation.h"
#include "types.h"
#include "move_record.h"
#include <regex.h>
#include <stdio.h>
#include <stdlib.h>

// Starting positions for the two kings
static int WH_KING_IDX = 4;
static int BL_KING_IDX = 60;

static void print_color(PlayerColor color) {
    char* color_string = (color == WHITE) ? "WHITE" : "BLACK";
    printf("Color: %s\n", color_string);
}

static void set_king(PlayerColor color, int idx) {
    if (color == WHITE) WH_KING_IDX = idx;
    else BL_KING_IDX = idx;
}

static bool cmp_with_idx(Chessboard *board, Piece piece, int idx) {
    Piece piece_at_idx = board->squares[idx];
    return cmp_pieces(piece_at_idx, piece);
}

static int _locate_knight(Chessboard* board, PlayerColor color, int dest_idx, 
                          int file, int rank) {
    int offsets[8][2] = {
        {1,2},
        {2,1},
        {2,-1},
        {1,-2},
        {-1,-2},
        {-2,-1},
        {-2,1},
        {-1,2}
    };

    int dest_file = file_from_idx(dest_idx);
    int dest_rank = rank_from_idx(dest_idx);
    int ret_idx = ERR_NONE_FOUND;
    Piece piece = KNIGHT;
    set_color(&piece, color);

    int src_idx;
    for (int i = 0; i < 8; i++) {
        int src_file = offsets[i][0] + dest_file;
        int src_rank = offsets[i][1] + dest_rank;

        // Ignore potential sources outside boundaries
        if (!(1 <= src_file && src_file <= 8)) continue;
        if (!(1 <= src_rank && src_rank <= 8)) continue;

        // Ignore potential sources that don't line up with given information
        if (file != 0 && src_file != file) continue;
        if (rank != 0 && src_rank != rank) continue;

        src_idx = idx_from_int(src_file, src_rank);

        // First piece found with right type / color is used
        if (cmp_with_idx(board, piece, src_idx))  {
            if (ret_idx == ERR_NONE_FOUND) ret_idx = src_idx;
            else return ERR_MULTIPLE_FOUND;
        }
    }
    return ret_idx;
}

static int _locate_rook(Chessboard* board, PlayerColor color, int dest_idx, 
                        int file, int rank, bool locating_queen) {

    Piece piece = (locating_queen) ? QUEEN : ROOK;
    set_color(&piece, color);

    int dest_file = file_from_idx(dest_idx);
    int dest_rank = rank_from_idx(dest_idx);
    int ret_idx = ERR_NONE_FOUND;

    int src_idx;

    int src_file = dest_file;
    // Check up
    for (int src_rank = dest_rank + 1; src_rank <= 8; src_rank++) {
        if (rank != 0 && src_rank != rank) continue;

        src_idx = idx_from_int(src_file, src_rank);

        if (cmp_with_idx(board, piece, src_idx))  {
            if (ret_idx == ERR_NONE_FOUND) ret_idx = src_idx;
            else return ERR_MULTIPLE_FOUND;
        }
        if (!is_none(board->squares[src_idx])) break;
    }

    // Check down
    for (int src_rank = dest_rank - 1; 1 <= src_rank; src_rank--) {
        if (rank != 0 && src_rank != rank) continue;

        src_idx = idx_from_int(src_file, src_rank);

        if (cmp_with_idx(board, piece, src_idx))  {
            if (ret_idx == ERR_NONE_FOUND) ret_idx = src_idx;
            else return ERR_MULTIPLE_FOUND;
        }
        if (!is_none(board->squares[src_idx])) break;
    }

    int src_rank = dest_rank;
    // Check right
    for (int src_file = dest_file + 1; src_file <= 8; src_file++) {
        if (file != 0 && src_file != file) continue;

        src_idx = idx_from_int(src_file, src_rank);

        if (cmp_with_idx(board, piece, src_idx))  {
            if (ret_idx == ERR_NONE_FOUND) ret_idx = src_idx;
            else return ERR_MULTIPLE_FOUND;
        }
        if (!is_none(board->squares[src_idx])) break;
    }

    // Check left
    for (int src_file = dest_file - 1; 1 <= src_file; src_file--) {
        if (file != 0 && src_file != file) continue;

        src_idx = idx_from_int(src_file, src_rank);

        if (cmp_with_idx(board, piece, src_idx))  {
            if (ret_idx == ERR_NONE_FOUND) ret_idx = src_idx;
            else return ERR_MULTIPLE_FOUND;
        }
        if (!is_none(board->squares[src_idx])) break;
    }

    return ret_idx;
}

static int _locate_bishop(Chessboard* board, PlayerColor color, int dest_idx, 
                          int file, int rank, bool locating_queen) {

    PieceType piece_type = (locating_queen) ? QUEEN : BISHOP;

    Piece piece = piece_type;
    set_color(&piece, color);

    int dest_file = file_from_idx(dest_idx);
    int dest_rank = rank_from_idx(dest_idx);
    int ret_idx = ERR_NONE_FOUND;
    
    int src_idx, src_file, src_rank;

    // up_right
    src_file = dest_file + 1;
    src_rank = dest_rank + 1;
    while (src_file <= 8 && src_rank <= 8) {
        if (file != 0 && src_file != file) continue;
        if (rank != 0 && src_rank != rank) continue;

        src_idx = idx_from_int(src_file, src_rank);
        if (cmp_with_idx(board, piece, src_idx))  {
            if (ret_idx == ERR_NONE_FOUND) ret_idx = src_idx;
            else return ERR_MULTIPLE_FOUND;
        }
        if (!is_none(board->squares[src_idx])) break;

        src_file++;
        src_rank++;
    }

    // down_right
    src_file = dest_file + 1;
    src_rank = dest_rank - 1;
    while (src_file <= 8 && src_rank >= 1) {
        if (file != 0 && src_file != file) continue;
        if (rank != 0 && src_rank != rank) continue;

        src_idx = idx_from_int(src_file, src_rank);
        if (cmp_with_idx(board, piece, src_idx))  {
            if (ret_idx == ERR_NONE_FOUND) ret_idx = src_idx;
            else return ERR_MULTIPLE_FOUND;
        }
        if (!is_none(board->squares[src_idx])) break;

        src_file++;
        src_rank--;
    }

    // down_left
    src_file = dest_file - 1;
    src_rank = dest_rank - 1;
    while (src_file >= 1 && src_rank >= 1) {

        if (file != 0 && src_file != file) continue;
        if (rank != 0 && src_rank != rank) continue;

        src_idx = idx_from_int(src_file, src_rank);
        if (cmp_with_idx(board, piece, src_idx))  {
            if (ret_idx == ERR_NONE_FOUND) ret_idx = src_idx;
            else return ERR_MULTIPLE_FOUND;
        }
        if (!is_none(board->squares[src_idx])) break;

        src_file--;
        src_rank--;
    }
    

    // up_left
    src_file = dest_file - 1; 
    src_rank = dest_rank + 1;
    while (src_file >= 1 && src_rank <= 8) {
        if (file != 0 && src_file != file) continue;
        if (rank != 0 && src_rank != rank) continue;

        src_idx = idx_from_int(src_file, src_rank);
        if (cmp_with_idx(board, piece, src_idx))  {
            if (ret_idx == ERR_NONE_FOUND) ret_idx = src_idx;
            else return ERR_MULTIPLE_FOUND;
        }
        if (!is_none(board->squares[src_idx])) {
            break;}

        src_file--;
        src_rank++;
    }

    return ret_idx;
}

static int _locate_queen(Chessboard* board, PlayerColor color, int dest_idx, 
                         int file, int rank) {

    // Check rook-like squares
    int ret_idx_r = _locate_rook(board, color, dest_idx, file, rank, true);
    if (ret_idx_r == ERR_MULTIPLE_FOUND) return ERR_MULTIPLE_FOUND;

    // Check bishop-like squares
    int ret_idx_b = _locate_bishop(board, color, dest_idx, file, rank, true);
    if (ret_idx_b == ERR_MULTIPLE_FOUND) return ERR_MULTIPLE_FOUND;

    if (ret_idx_r >= 0 && ret_idx_b >= 0) return ERR_MULTIPLE_FOUND;

    if (ret_idx_r == ERR_NONE_FOUND) {
        return ret_idx_b;
    }
    return ret_idx_r;
}

static int _locate_pawn(Chessboard* board, PlayerColor color, int dest_idx, 
                        int file, bool capture){
    int dest_file = file_from_idx(dest_idx);
    int dest_rank = rank_from_idx(dest_idx);
    Piece piece = PAWN;
    set_color(&piece, color);
    int src_idx;

    if (!capture) {
        // Should not be notated for standard movement
        if (file == 0) file = dest_file;
        else return ERR_NOTATION;

        if (color == WHITE) {
            src_idx = idx_from_int(file, --dest_rank);
            if (cmp_with_idx(board, piece, src_idx)) return src_idx;
            if (!is_none(board->squares[src_idx])) return ERR_NONE_FOUND;

            // Repeat to check for initial two-square movement
            if (dest_rank == 3) {
                src_idx = idx_from_int(file, --dest_rank);
                if (cmp_with_idx(board, piece, src_idx)) return src_idx;
            }
        }
        else {
            src_idx = idx_from_int(file, ++dest_rank);
            if (cmp_with_idx(board, piece, src_idx)) return src_idx;
            if (!is_none(board->squares[src_idx])) return ERR_NONE_FOUND;

            // Repeat to check for initial two-square movement
            if (dest_rank == 6) {
                src_idx = idx_from_int(file, ++dest_rank);
                if (cmp_with_idx(board, piece, src_idx)) return src_idx;
            }
        }
        return ERR_NONE_FOUND;
    } 

    // Column always provided for captures
    if (file < 1 || 8 < file) return ERR_NOTATION;

    src_idx = idx_from_int(file, --dest_rank);
    return cmp_with_idx(board, piece, src_idx) ? src_idx : ERR_NONE_FOUND;
}

static int _locate_king(Chessboard* board, PlayerColor color, int dest_idx) {
    int dest_file = file_from_idx(dest_idx);
    int dest_rank = rank_from_idx(dest_idx);
    Piece piece = KING;
    set_color(&piece, color);
    
    int src_idx;
    for (int file_offset = -1; file_offset <= 1; file_offset++) {
        for (int rank_offset = -1; rank_offset <= 1; rank_offset++) {
            if (file_offset == 0 && rank_offset == 0) continue;

            src_idx = idx_from_int(dest_file + file_offset, 
                                   dest_rank + rank_offset);
            if (!valid_index(src_idx)) continue;

            if (cmp_with_idx(board, piece, src_idx)) return src_idx;
        }
    }
    return ERR_NONE_FOUND;
}

int locate_piece(Chessboard* board, Move_Record* move_record, 
                 int file, int rank) {

    PlayerColor color = move_record->color;
    PieceType piece_type = move_record->piece_type;
    Piece piece = piece_type;
    set_color(&piece, color);

    int dest_idx = move_record->dest_idx;
    bool capture = move_record->capture;

    if (!capture && !is_none(board->squares[dest_idx])) {
        printf("Cannot move to occupied square. Can capture if legal.\n");
        return ERR_NOTATION;
    }

    int src_idx = ERR_NONE_FOUND;
    switch (piece_type) {
        case KNIGHT:
            src_idx = _locate_knight(board, color, dest_idx, file, rank);
            break;
        case BISHOP:
            src_idx = _locate_bishop(board, color, dest_idx, file, rank, false);
            break;
        case ROOK:
            src_idx = _locate_rook(board, color, dest_idx, file, rank, false);
            break;
        case QUEEN:
            src_idx = _locate_queen(board, color, dest_idx, file, rank);
            break;
        case KING:
            src_idx = _locate_king(board, color, dest_idx);
            break;
        case PAWN:
            src_idx = _locate_pawn(board, color, dest_idx, file, capture);
            break;
        default:
            src_idx = ERR_NOTATION;
    }

    if (valid_index(src_idx)) {
        move_record->src_idx = src_idx;
        return 0;
    }
    return src_idx;
}

bool is_SAN(char* user_input) {
    // TODO Correct such that promotion only works with pawns
    const char* pattern = ( "^(O-O(-O)?|0-0(-0)?|" // Castling
                            "[NBRQK]?[a-h]?[1-8]?x?[a-h][1-8])" // Standard
                            "(=[NBRQ])?[+#]?$"); // Checks and Promotions

    regex_t re;
    if (regcomp(&re, pattern, REG_EXTENDED | REG_ICASE)){
        printf("Failed to compile regex\n");
        return false;
    }

    int status = regexec(&re, user_input, 0, NULL, 0);
    regfree(&re);

    return status == 0;
}

// Helper function to allow for additional squares (such as during castling)
bool _is_check(Chessboard* board, PlayerColor color, int king_idx) {
    Piece piece = KING;
    set_color(&piece, color);

    PlayerColor opp_color = opposite_color(color);

    // Find king. If not at original index, scan neighboring squares
    if (king_idx == -1) {
        king_idx = (color == WHITE) ? WH_KING_IDX : BL_KING_IDX; 
        if (!cmp_with_idx(board, piece, king_idx)) {
            king_idx = _locate_king(board, color, king_idx);
        }
    }

    int ret_val;
    int king_file = file_from_idx(king_idx);

    if (king_file > 1) {
        ret_val = _locate_pawn(board, opp_color, king_idx, king_file-1, true);
        if (ret_val != ERR_NONE_FOUND) return true;
    }
    if (king_file < 8) {
        ret_val = _locate_pawn(board, opp_color, king_idx, king_file+1, true);
        if (ret_val != ERR_NONE_FOUND) return true;
    }
    ret_val = _locate_knight(board, opp_color, king_idx, 0, 0);
    if (ret_val != ERR_NONE_FOUND) return true;
    ret_val = _locate_bishop(board, opp_color, king_idx, 0, 0, false);
    if (ret_val != ERR_NONE_FOUND) return true;
    ret_val = _locate_rook(board, opp_color, king_idx, 0, 0, false);
    if (ret_val != ERR_NONE_FOUND) return true;
    ret_val = _locate_queen(board, opp_color, king_idx, 0, 0);
    if (ret_val != ERR_NONE_FOUND) return true;
    ret_val = _locate_king(board, opp_color, king_idx);
    if (ret_val != ERR_NONE_FOUND) return true;

    set_king(color, king_idx);
    return false;
}

bool is_check(Chessboard* board, PlayerColor color) {
    return _is_check(board, color, -1);
}

bool can_castle(Chessboard *board, PlayerColor color, Direction direction) {
    int king_file = 5;
    int rank = (color == WHITE) ? 1 : 8;

    int king_idx = idx_from_int(king_file, rank);
    Piece king_piece = board->squares[king_idx];

    // If the piece is not a king or has moved, return false
    if (!cmp_piece_type(king_piece, KING) 
        || !cmp_piece_color(king_piece, color)
        || has_moved(king_piece)) {
        return false;
    }

    int rook_file = (direction == KINGSIDE) ? 8 : 1;
    int rook_idx = idx_from_int(rook_file, rank);
    Piece rook_piece = board->squares[rook_idx];

    // If the piece is not a rook or has moved, return false
    if (!cmp_piece_type(rook_piece, ROOK) 
        || !cmp_piece_color(rook_piece, color)
        || has_moved(rook_piece)) {
        return false;
    }

    // If there are intervening pieces, return false
    for (int file = (direction == KINGSIDE) ? --rook_file : ++rook_file; 
        file != king_file; 
        (direction == KINGSIDE) ? --file : ++file
    ) {
        int intervening_idx = idx_from_int(file, rank);
        Piece intervening_piece = board->squares[intervening_idx];
        if (!cmp_piece_type(intervening_piece, NONE)) return false;
    }

    // Cannot castle into check
    int idx;
    if (direction == QUEENSIDE) {
        idx = idx_from_int(king_file-1, rank);
        if (_is_check(board, color, idx)) return false;

        idx = idx_from_int(king_file-2, rank);
        if (_is_check(board, color, idx)) return false;
    }
    else {
        idx = idx_from_int(king_file+1, rank);
        if (_is_check(board, color, idx)) return false;

        idx = idx_from_int(king_file+2, rank);
        if (_is_check(board, color, idx)) return false;
    }

    return true;
}

