#include "record.h"
#include "types.h"
#include <stdbool.h>
#include <stdio.h>

void clear_record(Move_Record* move_record) {
    move_record->turn_number = -1;
    move_record->color = 'x';
    move_record->piece_type = 'x';
    move_record->src_idx = -1;
    move_record->dest_idx = -1;
    move_record->capture = false;
    move_record->promotion = 'x';
    move_record->check = false;
    move_record->checkmate = false;
}

void print_record(Move_Record* move_record) {
    printf("%4.4d%c %c %d->%d %d%c%d%d", 
            move_record->turn_number,
            move_record->color,
            move_record->piece_type,
            move_record->src_idx,
            move_record->dest_idx,
            move_record->capture,
            move_record->promotion,
            move_record->check,
            move_record->checkmate);
}

void fprint_record(FILE* fp, Move_Record* move_record) {
    fprintf(fp, "%4.4d%c %c %d->%d %d%c%d%d", 
                move_record->turn_number,
                move_record->color,
                move_record->piece_type,
                move_record->src_idx,
                move_record->dest_idx,
                move_record->capture,
                move_record->promotion,
                move_record->check,
                move_record->checkmate);
}