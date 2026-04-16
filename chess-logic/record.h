#ifndef RECORD_H
#define RECORD_H

#include <stddef.h>
#include <stdint.h>
#include "types.h"

typedef struct {
    size_t turn_number; // Turn Number // TODO: Decide if needed
    int color; // Current Player
    uint8_t piece_type; // Piece Moving
    int src_idx; // Source Square
    int dest_idx; // Destination Square
    bool capture; // Capture?
    uint8_t promotion; // Pawn Promotion?
    bool check; // Check?
    bool checkmate; // Checkmate?
} Move_Record;

void clear_record(Move_Record* move_record);

#endif /* RECORD_H */