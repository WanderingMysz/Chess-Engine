#include "manipulate_board.h"
#include "move_validation.h"
#include "types.h"
#include "unity.h"
#include <stdbool.h>

Chessboard board;
int wh_k_rook_idx;
int wh_q_rook_idx;
int wh_king_idx;

int bl_k_rook_idx;
int bl_q_rook_idx;
int bl_king_idx;

void setUp(void) {
    // Clean setup
    board = initialize_empty_chessboard();
    wh_k_rook_idx = idx_from_int(1,1);
    wh_q_rook_idx = idx_from_int(8,1);
    wh_king_idx   = idx_from_int(4,1);

    bl_k_rook_idx = idx_from_int(1,8);
    bl_q_rook_idx = idx_from_int(8,8);
    bl_king_idx   = idx_from_int(4,8);
}

void tearDown(void) {

}

void set_kings_and_rooks(void) {
    // Sets white pieces
    set_square(&board, wh_k_rook_idx, ROOK);
    set_square(&board, wh_q_rook_idx, ROOK);
    set_square(&board, wh_king_idx, KING);
    // Sets black pieces
    set_square(&board, bl_k_rook_idx, ROOK);
    set_square(&board, bl_q_rook_idx, ROOK);
    set_square(&board, bl_king_idx, KING);
}

// Trying to castle with empty squares
void test_can_castle_missing_pieces(void) {
    TEST_ASSERT_FALSE(can_castle(&board, true, true)); 
    TEST_ASSERT_FALSE(can_castle(&board, true, false)); 
    TEST_ASSERT_FALSE(can_castle(&board, false, true)); 
    TEST_ASSERT_FALSE(can_castle(&board, false, false));
    
    // Sets white kingside rook only
    set_square(&board, wh_k_rook_idx, ROOK);
    TEST_ASSERT_FALSE(can_castle(&board, true, true)); 

    // Sets white king, tries castling queenside
    set_square(&board, wh_king_idx, KING);
    TEST_ASSERT_FALSE(can_castle(&board, true, false)); 
}

void test_can_castle_unblocked_unmoved(void) {
    set_kings_and_rooks();

    TEST_ASSERT_TRUE(can_castle(&board, true, true)); 
    TEST_ASSERT_TRUE(can_castle(&board, true, false)); 
    TEST_ASSERT_TRUE(can_castle(&board, false, true)); 
    TEST_ASSERT_TRUE(can_castle(&board, false, false));
}

void test_can_castle_unblocked_moved(void) {
    set_kings_and_rooks();

    // White kingside rook moved
    setMoved(&(board.squares[wh_k_rook_idx]));
    TEST_ASSERT_FALSE(can_castle(&board, true, true));
    TEST_ASSERT_TRUE(can_castle(&board, true, false));

    // White king moved
    setMoved(&(board.squares[wh_king_idx]));
    TEST_ASSERT_FALSE(can_castle(&board, true, true));
    TEST_ASSERT_FALSE(can_castle(&board, true, false));
}

void test_can_castle_blocked_unmoved(void) {
    set_kings_and_rooks();

    // Sets knight between king and rook
    int wh_k_knight_idx = idx_from_int(2,1);
    set_square(&board, wh_k_knight_idx, KNIGHT);
    TEST_ASSERT_FALSE(can_castle(&board, true, true));

    // Sets knight to having moved (at some point) but it is still in the square
    setMoved(&(board.squares[wh_k_knight_idx]));
    TEST_ASSERT_FALSE(can_castle(&board, true, true));

    // Sets knight's square to having no piece to verify NONE
    set_square(&board, wh_k_knight_idx, NONE);
    TEST_ASSERT_TRUE(can_castle(&board, true, true));
}

int main(void) {
    UNITY_BEGIN();
    
    RUN_TEST(test_can_castle_missing_pieces);
    RUN_TEST(test_can_castle_unblocked_unmoved);
    RUN_TEST(test_can_castle_unblocked_moved);

    return UNITY_END();
}