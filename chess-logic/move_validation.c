#include "move_validation.h"
#include "piece_info.h"
#include "errors.h"
#include <regex.h>
#include <stdio.h>
#include <stdlib.h>

static int WH_KING_IDX = 4;
static int BL_KING_IDX = 60;

void set_king(bool white, int idx) {
    if (white) WH_KING_IDX = idx;
    else BL_KING_IDX = idx;
}

// Returns if a piece matches the given type
bool cmp_piece_type(uint8_t piece, PieceType comp) {
    if (comp == NONE) {
        piece &= PIECE_MASK;

        if (piece) return false;
        return true;
    }
    return piece & comp;
}

bool cmp_piece_color(uint8_t piece, bool white) {
    return (white) ? isWhite(piece) : !isWhite(piece);
}

bool piece_exists(Chessboard* board, int idx, uint8_t piece) {
    uint8_t piece_there = board->squares[idx];

    // Movement_mask doesn't matter for id
    piece |= MOVEMENT_MASK;
    piece_there |= MOVEMENT_MASK;

    return piece == piece_there;
}

static bool check_square(Chessboard *board, uint8_t piece_type, bool white, 
                        int src_idx) {
    uint8_t piece_at_idx = board->squares[src_idx];

    // None doesn't compare about color
    if (piece_type == NONE) {
        return isNone(piece_at_idx);
    }

    // First piece found with right type / color is used
    if (cmp_piece_type(piece_at_idx, piece_type) 
        && isWhite(piece_at_idx) == white) {
        return true;
    }
    return false;
}

/* TODO: Refactor to have explicit error codes that are called, not nums */
/* -1 Not Found, -2 More than one found, -3 Notation error, -4 Other */
static int _locate_knight(Chessboard* board, bool white, int dest_idx, 
                    int col, int row) {
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

    int dest_col = col_from_idx(dest_idx);
    int dest_row = row_from_idx(dest_idx);
    int ret_idx = -1;

    for (int i = 0; i < 8; i++) {
        int src_col = offsets[i][0] + dest_col;
        int src_row = offsets[i][1] + dest_row;

        // Ignore potential sources outside boundaries
        if (!(1 <= src_col && src_col <= 8)) continue;
        if (!(1 <= src_row && src_row <= 8)) continue;

        // Ignore potential sources that don't line up with given information
        if (col != 0 && src_col != col) continue;
        if (row != 0 && src_row != row) continue;

        int src_idx = idx_from_int(src_col, src_row);

        // First piece found with right type / color is used
        if (check_square(board, KNIGHT, white, src_idx))  {
            if (ret_idx == -1) ret_idx = src_idx;
            else return ERR_MULTIPLE_FOUND;
        }
    }

    return ret_idx;
}

static int _locate_rook(Chessboard* board, bool white, int dest_idx, 
                    int col, int row, bool queen) {

    uint8_t piece_type = (queen) ? QUEEN : ROOK;

    int dest_col = col_from_idx(dest_idx);
    int dest_row = row_from_idx(dest_idx);
    int ret_idx = ERR_NONE_FOUND;

    int src_col = dest_col;
    // Check up
    for (int src_row = dest_row + 1; src_row <= 8; src_row++) {
        if (row != 0 && src_row != row) continue;

        int src_idx = idx_from_int(src_col, src_row);

        if (check_square(board, piece_type, white, src_idx))  {
            if (ret_idx == ERR_NONE_FOUND) ret_idx = src_idx;
            else return ERR_MULTIPLE_FOUND;
        }
        if (!check_square(board, NONE, white, src_idx)) {
            break;
        }
    }

    // Check down
    for (int src_row = dest_row - 1; 1 <= src_row; src_row--) {
        if (row != 0 && src_row != row) continue;

        int src_idx = idx_from_int(src_col, src_row);

        if (check_square(board, piece_type, white, src_idx))  {
            if (ret_idx == ERR_NONE_FOUND) ret_idx = src_idx;
            else return ERR_MULTIPLE_FOUND;
        }
        if (!check_square(board, NONE, white, src_idx)) {
            break;
        }
    }

    int src_row = dest_row;
    // Check right
    for (int src_col = dest_col + 1; src_col <= 8; src_col++) {
        if (col != 0 && src_col != col) continue;

        int src_idx = idx_from_int(src_col, src_row);

        if (check_square(board, piece_type, white, src_idx))  {
            if (ret_idx == ERR_NONE_FOUND) ret_idx = src_idx;
            else return ERR_MULTIPLE_FOUND;
        }
        if (!check_square(board, NONE, white, src_idx)) {
            break;
        }
    }

    // Check left
    for (int src_col = dest_col - 1; 1 <= src_col; src_col--) {
        if (col != 0 && src_col != col) continue;

        int src_idx = idx_from_int(src_col, src_row);

        if (check_square(board, piece_type, white, src_idx))  {
            if (ret_idx == ERR_NONE_FOUND) ret_idx = src_idx;
            else return ERR_MULTIPLE_FOUND;
        }
        if (!check_square(board, NONE, white, src_idx)) {
            break;
        }
    }

    return ret_idx;
}

static int _locate_bishop(Chessboard* board, bool white, int dest_idx, 
                    int col, int row, bool queen) {

    uint8_t piece_type = (queen) ? QUEEN : BISHOP;

    int dest_col = col_from_idx(dest_idx);
    int dest_row = row_from_idx(dest_idx);
    int ret_idx = ERR_NONE_FOUND;

    bool break_loop = false;
    
    // up_right
    int src_row = dest_row + 1;
    for (int src_col = dest_col + 1; src_col <= 8 && src_row <= 8; src_col++) {
        if (col != 0 && src_col != col) continue;
        if (row != 0 && src_row != row) continue;

        int src_idx = idx_from_int(src_col, src_row);

        if (check_square(board, piece_type, white, src_idx))  {
            if (ret_idx == ERR_NONE_FOUND) ret_idx = src_idx;
            else return ERR_MULTIPLE_FOUND;
        }

        if (!check_square(board, NONE, white, src_idx)) {
            break;
        }

        src_row++;
    }

    // down_right
    src_row = dest_row - 1;
    for (int src_col = dest_col + 1; src_col <= 8 && 1 <= src_row; src_col++) {
        if (col != 0 && src_col != col) continue;
        if (row != 0 && src_row != row) continue;

        int src_idx = idx_from_int(src_col, src_row);

        if (check_square(board, piece_type, white, src_idx))  {
            if (ret_idx == ERR_NONE_FOUND) ret_idx = src_idx;
            else return ERR_MULTIPLE_FOUND;
        }

        if (!check_square(board, NONE, white, src_idx)) {
            break;
        }

        src_row--;
    }

    // down_left
    src_row = dest_row - 1;
    for (int src_col = dest_col - 1; 1 <= src_col && 1 <= src_row; src_col--) {
        if (col != 0 && src_col != col) continue;
        if (row != 0 && src_row != row) continue;

        int src_idx = idx_from_int(src_col, src_row);

        if (check_square(board, piece_type, white, src_idx))  {
            if (ret_idx == ERR_NONE_FOUND) ret_idx = src_idx;
            else return ERR_MULTIPLE_FOUND;
        }
        if (!check_square(board, NONE, white, src_idx)) {
            break;
        }

        src_row--;
    }
    

    // up_left
    src_row = dest_row + 1;
    for (int src_col = dest_col - 1; 1 <= src_col && src_row <= 8; src_col--) {
        if (col != 0 && src_col != col) continue;
        if (row != 0 && src_row != row) continue;

        int src_idx = idx_from_int(src_col, src_row);

        if (check_square(board, piece_type, white, src_idx))  {
            if (ret_idx == ERR_NONE_FOUND) ret_idx = src_idx;
            else return ERR_MULTIPLE_FOUND;
        }
        if (!check_square(board, NONE, white, src_idx)) {
            break;
        }

        src_row++;
    }


    return ret_idx;
}

static int _locate_queen(Chessboard* board, bool white, int dest_idx, 
                    int col, int row) {

    // Check rook-like squares
    int ret_idx_r = _locate_rook(board, white, dest_idx, col, row, true);
    if (ret_idx_r == ERR_MULTIPLE_FOUND) return ERR_MULTIPLE_FOUND;

    // Check bishop-like squares
    int ret_idx_b = _locate_bishop(board, white, dest_idx, col, row, true);
    if (ret_idx_b == ERR_MULTIPLE_FOUND) return ERR_MULTIPLE_FOUND;

    if ((ret_idx_r >= 0 && ret_idx_b >= 0)
        || ret_idx_r == ERR_MULTIPLE_FOUND 
        || ret_idx_b == ERR_MULTIPLE_FOUND) 
        return ERR_MULTIPLE_FOUND;

    if (ret_idx_r == ERR_NONE_FOUND) {
        return ret_idx_b;
    }
    return ret_idx_r;
}

static int _locate_pawn(Chessboard* board, bool white, int dest_idx, 
                        int col, bool capture){
    int dest_col = col_from_idx(dest_idx);
    int dest_row = row_from_idx(dest_idx);
    int src_idx = ERR_NONE_FOUND;

    if (!capture) {
        // Should not be notated for standard movement
        if (col == 0) col = dest_col;
        else return ERR_NOTATION;

        if (white) {
            dest_row--;
            src_idx = idx_from_int(col, dest_row);
            if (check_square(board, PAWN, white, src_idx)) return src_idx;
            if (!check_square(board, NONE, white, src_idx)) {
                return ERR_NONE_FOUND;
            }

            // Repeat to check for initial two-square movement
            if (dest_row == 3) {
                dest_row--;
                src_idx = idx_from_int(col, dest_row);
                if (check_square(board, PAWN, white, src_idx)) return src_idx;
            }
        }
        else {
            dest_row++;
            src_idx = idx_from_int(col, dest_row);
            if (check_square(board, PAWN, white, src_idx)) return src_idx;
            if (!check_square(board, NONE, white, src_idx)) {
                return ERR_NONE_FOUND;
            }

            // Repeat to check for initial two-square movement
            if (dest_row == 6) {
                dest_row++;
                src_idx = idx_from_int(col, dest_row);
                if (check_square(board, PAWN, white, src_idx)) return src_idx;
            }
        }

        return ERR_NONE_FOUND;
    } 

    // Column always provided for captures
    if (col < 1 || 8 < col) return ERR_NOTATION;
    src_idx = idx_from_int(col, --dest_row);
    return check_square(board, PAWN, white, src_idx) ? src_idx : ERR_NONE_FOUND;
}

// static int _locate_king(bool white, int dest_idx) {
//     int dest_col = col_from_idx(dest_idx);
//     int dest_row = row_from_idx(dest_idx);
//     int king_col = col_from_idx((white) ? WH_KING_IDX : BL_KING_IDX);
//     int king_row = row_from_idx((white) ? WH_KING_IDX : BL_KING_IDX);

//     int ret_idx = ERR_NOTATION;

//     if (abs(dest_col - king_col) <= 1 && abs(dest_row - king_row) <= 1) {
//         if (white) {
//             ret_idx = WH_KING_IDX;
//             WH_KING_IDX = dest_idx;
//         } else {
//             ret_idx = BL_KING_IDX;
//             BL_KING_IDX = dest_idx;
//         }
//     }
//     return ret_idx;
// }

static int _locate_king(Chessboard* board, bool color, int dest_idx) {
    int src_idx = col_from_idx(dest_idx);
    int src_row = row_from_idx(dest_idx);

    for (int i = -1; i <= 1; i++) {
        if (i < 1 || 8 < i) continue;
        for (int j = -1; j <= 1; j++) {
            if (j < 1 || j < i) continue;

            if (i == 0 && j == 0) continue;

            int new_idx = idx_from_int(src_idx + i, src_row + j);
            uint8_t piece_found = board->squares[new_idx];

            if (cmp_piece_type(piece_found, KING) 
                && cmp_piece_color(piece_found, color)) {
                    return new_idx;
                }
        }
    }
    return -1;
}

int locate_piece(Chessboard* board, uint8_t piece, int dest_idx, 
                 int col, int row, bool capture) {
    bool white = isWhite(piece);
    uint8_t piece_type = piecetype(piece);

    if (!capture && !check_square(board, NONE, white, dest_idx)) {
        printf("Cannot move to occupied square. Can capture if legal.\n");
        return ERR_NOTATION;
    }

    int ret_idx = ERR_NONE_FOUND;
    switch (piece_type) {
        case KNIGHT:
            ret_idx = _locate_knight(board, white, dest_idx, col, row);
            break;
        case BISHOP:
            ret_idx = _locate_bishop(board, white, dest_idx, col, row, false);
            break;
        case ROOK:
            ret_idx = _locate_rook(board, white, dest_idx, col, row, false);
            break;
        case QUEEN:
            ret_idx = _locate_queen(board, white, dest_idx, col, row);
            break;
        case KING:
            ret_idx = _locate_king(board, white, dest_idx);
            break;
        default:
            ret_idx = _locate_pawn(board, white, dest_idx, col, capture);
            break;
    }

    if (ret_idx >= 64) {
        printf("ERROR: Piece found to be out of bounds at index %d.\n",ret_idx);
        return ERR_OTHER;
    }
    return ret_idx;
}

// FIXME cannot castle into danger
bool can_castle(Chessboard *board, bool white, bool kingside) {
    int king_col = 5;
    int row = white ? 1 : 8;

    int king_idx = idx_from_int(king_col, row);
    uint8_t king_piece = board->squares[king_idx];

    // If the piece is not a king or has moved, return false
    if (!cmp_piece_type(king_piece, KING) || hasMoved(king_piece)) {
        return false;
    }

    int rook_col = kingside ? 8 : 1;
    int rook_idx = idx_from_int(rook_col, row);
    uint8_t rook_piece = board->squares[rook_idx];

    // If the piece is not a rook or has moved, return false
    if (!cmp_piece_type(rook_piece, ROOK) || hasMoved(rook_piece)) {
        return false;
    }

    // If there are intervening pieces, return false
    for (int col = (kingside) ? --rook_col : ++rook_col; 
        col != king_col; 
        (kingside) ? --col : ++col
    ) {
        int intervening_idx = idx_from_int(col, row);
        uint8_t intervening_piece = board->squares[intervening_idx];

        if ((intervening_piece & ~(COLOR_MASK | MOVEMENT_MASK))) return false;
    }

    return true;
}

bool is_SAN(char* move) {
    // TODO Correct such that promotion only works with pawns
    const char* pattern = ( "^(O-O(-O)?|0-0(-0)?|" // Castling
                            "[NBRQK]?[a-h]?[1-8]?x?[a-h][1-8])" // Standard
                            "[+#]?(=[NBRQ])?$"); // Checks and Promotions

    regex_t re;
    if (regcomp(&re, pattern, REG_EXTENDED | REG_ICASE)){
        printf("Failed to compile regex\n");
        return false;
    }

    int status = regexec(&re, move, 0, NULL, 0);
    regfree(&re);

    return status == 0;
}

bool is_check(Chessboard* board, bool white) {
    // Find king. If not at original index, scan neighboring squares
    int king_idx = white ? WH_KING_IDX : BL_KING_IDX;
    uint8_t piece_at_idx = board->squares[king_idx];
    if (!cmp_piece_type(piece_at_idx, KING)) {
        king_idx = _locate_king(board, white, king_idx);
        // NOTE: Architecturally, there shouldn't be an error here
        // Should figure out a proper balance of checking for errors and
        // assuming code works as designed.
    }

    // printf("White's Turn?: %d\n", color);

    int king_col = col_from_idx(king_idx);

    int ret_val;
    if (king_col > 1) {
        ret_val = _locate_pawn(board, !white, king_idx, king_col-1, true);
        if (ret_val != -1) return true;
    }
    if (king_col < 8) {
        ret_val = _locate_pawn(board, !white, king_idx, king_col+1, true);
        if (ret_val != -1) return true;
    }
    ret_val = _locate_knight(board, !white, king_idx, 0, 0);
    if (ret_val != -1) return true;
    ret_val = _locate_bishop(board, !white, king_idx, 0, 0, false);
    if (ret_val != -1) return true;
    ret_val = _locate_rook(board, !white, king_idx, 0, 0, false);
    if (ret_val != -1) return true;
    ret_val = _locate_queen(board, !white, king_idx, 0, 0);
    if (ret_val != -1) return true;
    ret_val = _locate_king(board, !white, king_idx);
    if (ret_val != -1) return true;

    set_king(white, king_idx);
    return false;
}
