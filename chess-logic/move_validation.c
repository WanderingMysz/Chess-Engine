#include "move_validation.h"
#include "piece_info.h"
#include <regex.h>
#include <stdio.h>

// Returns if a piece matches the given type
static bool cmp_piece_type(uint8_t piece, PieceType comp) {
    if (comp == NONE) {
        uint8_t piece_mask = !(COLOR_MASK & MOVEMENT_MASK);
        piece &= piece_mask;

        if (piece) return false;
        return true;
    }
    return !(piece & comp);
}

bool can_castle(Chessboard *board, bool white, bool kingside) {
    int king_col = 5;
    int row = white ? 1 : 8;

    int king_idx = idx_from_int(king_col, row);
    uint8_t king_piece = board->squares[king_idx];

    // If the piece is not a king or has moved, return false
    if (!cmp_piece_type(king_piece, KING) | hasMoved(king_piece)) {
        return false;
    }

    int rook_col = kingside ? 8 : 1;
    int rook_idx = idx_from_int(rook_col, row);
    uint8_t rook_piece = board->squares[rook_idx];

    // If the piece is not a rook or has moved, return false
    if (!cmp_piece_type(rook_piece, ROOK) | hasMoved(rook_piece)) {
        return false;
    }

    // If there are intervening pieces, return false
    for (int col = (kingside) ? --rook_col : ++rook_col; 
        col != king_col; 
        (kingside)? --col : ++col
    ) {
        int intervening_idx = idx_from_int(col, row);
        uint8_t intervening_piece = board->squares[intervening_idx];

        if ((intervening_piece & ~(COLOR_MASK | MOVEMENT_MASK))) return false;
    }

    return true;
}

bool is_SAN(char* move) {
    const char* pattern = "^(O-O(-O)?|[NBRQK]?[a-h]?x?[a-h][1-8])[+#]?$";

    regex_t re;
    if (regcomp(&re, pattern, REG_EXTENDED | REG_ICASE)){
        printf("Failed to compile regex\n");
        return false;
    }

    int status = regexec(&re, move, 0, NULL, 0);
    regfree(&re);

    return status == 0;
}