#include "piece_info.h"
#include "unity.h"

void setUp(void) {

}

void tearDown(void) {

}

void test_idx_from_char(void) {
    TEST_ASSERT_TRUE(idx_from_char('A','8') == 56);
    TEST_ASSERT_TRUE(idx_from_char('H','8') == 63);
    TEST_ASSERT_TRUE(idx_from_char('A','1') == 0);
    TEST_ASSERT_TRUE(idx_from_char('H','1') == 7);
}

void test_idx_from_int(void) {
    TEST_ASSERT_TRUE(idx_from_int(1, 8) == 56);
    TEST_ASSERT_TRUE(idx_from_int(8, 8) == 63);
    TEST_ASSERT_TRUE(idx_from_int(1, 1) == 0);
    TEST_ASSERT_TRUE(idx_from_int(8, 1) == 7);
}

void test_col(void) {
    TEST_ASSERT_TRUE(col_from_idx(0) == 1);
    TEST_ASSERT_TRUE(col_from_idx(7) == 8);
    TEST_ASSERT_TRUE(col_from_idx(56) == 1);
    TEST_ASSERT_TRUE(col_from_idx(63) == 8);
}

void test_row(void) {
    TEST_ASSERT_TRUE(row_from_idx(0) == 1);
    TEST_ASSERT_TRUE(row_from_idx(7) == 1);
    TEST_ASSERT_TRUE(row_from_idx(56) == 8);
    TEST_ASSERT_TRUE(row_from_idx(63) == 8);
}

int main(void) {
    UNITY_BEGIN();
    
    RUN_TEST(test_idx_from_char);
    RUN_TEST(test_idx_from_int);
    RUN_TEST(test_col);
    RUN_TEST(test_row);

    return UNITY_END();
}