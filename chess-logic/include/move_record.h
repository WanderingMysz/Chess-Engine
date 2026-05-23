#ifndef MOVE_RECORD_H
#define MOVE_RECORD_H

#include "types.h"
#include <stdbool.h>

typedef struct {
    TurnNumber      turn_number;    // Turn Number
    PlayerColor     color;          // Current Player
    PieceType       piece_type;     // Piece Moving
    int             src_idx;        // Source Square
    int             dest_idx;       // Destination Square
    bool            capture;        // Capture?
    bool            check;          // Check?
    bool            checkmate;      // Checkmate?
    PieceType       promotion;      // Pawn Promotion?
} Move_Record;

#endif /* MOVE_RECORD_H */