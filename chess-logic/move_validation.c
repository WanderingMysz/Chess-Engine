#include "move_validation.h"
#include "piece_info.h"
#include <regex.h>
#include <stdio.h>

static int WH_KING_IDX = -1;
static int BL_KING_IDX = -1;

// Returns if a piece matches the given type
bool cmp_piece_type(uint8_t piece, PieceType comp) {
    if (comp == NONE) {
        piece &= PIECE_MASK;

        if (piece) return false;
        return true;
    }
    return piece & comp;
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

/* -1 Not Found, -2 More than one found */
int _locate_knight(Chessboard* board, bool white, int dest_idx, 
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
            else return -2;
        }
    }

    return ret_idx;
}

int _locate_rook(Chessboard* board, bool white, int dest_idx, 
                    int col, int row, bool queen) {

    uint8_t piece_type = (queen) ? QUEEN : ROOK;

    int dest_col = col_from_idx(dest_idx);
    int dest_row = row_from_idx(dest_idx);
    int ret_idx = -1;

    // Check up
    for (int src_row = dest_row + 1; src_row <= 8; src_row++) {
        if (row != 0 && src_row != row) continue;

        int src_idx = idx_from_int(col, src_row);

        if (check_square(board, piece_type, white, src_idx))  {
            if (ret_idx == -1) ret_idx = src_idx;
            else return -2;
        }
        if (!check_square(board, NONE, white, src_idx)) {
            break;
        }
    }

    // Check down
    for (int src_row = dest_row - 1; 1 <= src_row; src_row--) {
        if (row != 0 && src_row != row) continue;

        int src_idx = idx_from_int(col, src_row);

        if (check_square(board, piece_type, white, src_idx))  {
            if (ret_idx == -1) ret_idx = src_idx;
            else return -2;
        }
        if (!check_square(board, NONE, white, src_idx)) {
            break;
        }
    }

    // Check right
    for (int src_col = dest_col + 1; src_col <= 8; src_col++) {
        if (col != 0 && src_col != col) continue;

        int src_idx = idx_from_int(src_col, row);

        if (check_square(board, piece_type, white, src_idx))  {
            if (ret_idx == -1) ret_idx = src_idx;
            else return -2;
        }
        if (!check_square(board, NONE, white, src_idx)) {
            break;
        }
    }

    // Check left
    for (int src_col = dest_col - 1; 1 <= src_col; src_col--) {
        if (col != 0 && src_col != col) continue;

        int src_idx = idx_from_int(src_col, row);

        if (check_square(board, piece_type, white, src_idx))  {
            if (ret_idx == -1) ret_idx = src_idx;
            else return -2;
        }
        if (!check_square(board, NONE, white, src_idx)) {
            break;
        }
    }

    return ret_idx;
}

int _locate_bishop(Chessboard* board, bool white, int dest_idx, 
                    int col, int row, bool queen) {

    uint8_t piece_type = (queen) ? QUEEN : ROOK;

    int dest_col = col_from_idx(dest_idx);
    int dest_row = row_from_idx(dest_idx);
    int ret_idx = -1;

    bool break_loop = false;
    
    // up_right
    int src_row = dest_row + 1;
    for (int src_col = dest_col + 1; src_col <= 8 && src_row <= 8; src_col++) {
        if (col != 0 && src_col != col) continue;
        if (row != 0 && src_row != row) continue;

        int src_idx = idx_from_int(src_col, src_row);

        if (check_square(board, piece_type, white, src_idx))  {
            if (ret_idx == -1) ret_idx = src_idx;
            else return -2;
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
            if (ret_idx == -1) ret_idx = src_idx;
            else return -2;
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
            if (ret_idx == -1) ret_idx = src_idx;
            else return -2;
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
            if (ret_idx == -1) ret_idx = src_idx;
            else return -2;
        }
        if (!check_square(board, NONE, white, src_idx)) {
            break;
        }

        src_row++;
    }


    return ret_idx;
}

int _locate_queen(Chessboard* board, bool white, int dest_idx, 
                    int col, int row) {

    // Check rook-like squares
    int ret_idx_r = _locate_rook(board, white, dest_idx, col, row, true);
    if (ret_idx_r == -2) return -2;

    // Check bishop-like squares
    int ret_idx_b = _locate_bishop(board, white, dest_idx, col, row, true);
    if (ret_idx_b == -2) return -2;

    if ((ret_idx_r >= 0 && ret_idx_b >= 0)
        || ret_idx_r == -2 
        || ret_idx_b == -2) 
        return -2;

    if (ret_idx_r == -1) {
        return ret_idx_b;
    }
    return ret_idx_r;
}

// Unlike pieces, pawns will ALWAYS have a col label and NEVER a row label
// int _locate_pawn(Chessboard* board, bool white, int dest_idx, int col);

int locate_piece(Chessboard* board, uint8_t piece, int dest_idx, int col, int row) {
    bool white = isWhite(piece);
    uint8_t piece_type = piecetype(piece);

    int ret_idx = -1;
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
            ret_idx = (white) ? WH_KING_IDX : BL_KING_IDX;
            break;
        default:
            ret_idx = _locate_pawn(board, white, dest_idx, col, row);
            break;
    }

    return ret_idx;
}

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
    const char* pattern = "^(O-O(-O)?|[NBRQK]?[A-H]?x?[A-H][1-8])[+#]?(=[NBRQ])?$";

    regex_t re;
    if (regcomp(&re, pattern, REG_EXTENDED | REG_ICASE)){
        printf("Failed to compile regex\n");
        return false;
    }

    int status = regexec(&re, move, 0, NULL, 0);
    regfree(&re);

    return status == 0;
}