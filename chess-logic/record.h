#ifndef RECORD_H
#define RECORD_H

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdbool.h>

typedef struct {
    size_t turn_number; // Turn Number // TODO: Decide if needed
    char color; // Current Player
    char piece_type; // Piece Moving
    int src_idx; // Source Square
    int dest_idx; // Destination Square
    bool capture; // Capture?
    char promotion; // Pawn Promotion?
    bool check; // Check?
    bool checkmate; // Checkmate?
} Move_Record;

void clear_record(Move_Record* move_record);

void print_record(Move_Record* move_record);

void fprint_record(FILE* fp, Move_Record* move_record);

#endif /* RECORD_H */