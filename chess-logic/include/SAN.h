#ifndef SAN_H
#define SAN_H

#include <string.h>
#include <stdbool.h>
#include "types.h"
#include "move_record.h"

/* ------------------------ Basic SAN Feature Parsing ----------------------- */

// Returns if a string is properly formatted SAN
bool is_SAN(char* string);

// Returns if the SAN is a capture
bool is_capture(char* SAN_string);

// Returns if the SAN includes a check/checkmate
bool is_check(char* SAN_string);

// Returns if the SAN is a checkmate
bool is_checkmate(char* SAN_string);

// Returns if the SAN is a valid promotion (Pawn to final rank)
bool valid_promotion(char* SAN_string);

/* -------------------- Coordinate (File/Rank) Validation ------------------- */

// Returns if a given file is valid, formatted using char
bool valid_file_char(char file);

// Returns if a given file is valid, formatted using int
bool valid_file_int(int file);

// Returns if a given rank is valid, formatted using char
bool valid_rank_char(char rank);

// Returns if a given rank is valid, formatted using int
bool valid_rank_int(int rank);

// Returns if a given coordinate is valid, formatted using char
bool valid_coordinate_char(char file, char rank);

// Returns if a given coordinate is valid, formatted using int
bool valid_coordinate_int(int file, int rank);

/* -------------------------- Piece Interpretation -------------------------- */

// Returns if the movement present is on the surface legal
bool valid_movement(char* SAN_string);

// Returns the type of piece represented by a given letter
PieceType interpret_letter(char letter);

/* ------------------------ Conversion to Move Record ----------------------- */

// Interprets string as SAN and converts to a Move_Record, as much as possible
Move_Record convert_to_record(char* SAN_string);

// Updates a given Move_Record based on provided SAN string
void update_record(char* SAN_string, Move_Record* move_record);

#endif /* SAN_H */
