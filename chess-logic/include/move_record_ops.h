#ifndef MOVE_RECORD_OPS_H
#define MOVE_RECORD_OPS_H

#include <stddef.h>
#include <stdint.h>
#include "types.h"
#include "move_record.h"

// Zeroes out a move record
void clear_record(Move_Record* move_record);

/* Converts SAN input to move record. Returns 0 on success */
int generate_record(Chessboard* board, char* SAN_input, 
                    Move_Record* move_record);

void print_record(Move_Record* move_record);

#endif /* MOVE_RECORD_OPS_H */