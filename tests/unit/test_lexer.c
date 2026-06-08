#include "test_helpers.h"
#include "parser.tab.h"

/* Flex interface */
typedef struct yy_buffer_state *YY_BUFFER_STATE;
extern int yylex(void);
extern YY_BUFFER_STATE yy_scan_string(const char *str);
extern void yy_delete_buffer(YY_BUFFER_STATE buf);
extern char *yytext;
extern YYSTYPE yylval;

TEST(test_dim_keyword) {
    YY_BUFFER_STATE buf = yy_scan_string("dim");
    ASSERT_INT_EQ(TOK_DIM, yylex());
    yy_delete_buffer(buf);
}

TEST(test_tensor_keyword) {
    YY_BUFFER_STATE buf = yy_scan_string("tensor");
    ASSERT_INT_EQ(TOK_TENSOR, yylex());
    yy_delete_buffer(buf);
}

TEST(test_identifier) {
    YY_BUFFER_STATE buf = yy_scan_string("myVar");
    ASSERT_INT_EQ(TOK_IDENT, yylex());
    ASSERT_STR_EQ("myVar", yylval.str_val);
    free(yylval.str_val);
    yy_delete_buffer(buf);
}

TEST(test_int_literal) {
    YY_BUFFER_STATE buf = yy_scan_string("42");
    ASSERT_INT_EQ(TOK_INT_LIT, yylex());
    ASSERT_INT_EQ(42, yylval.int_val);
    yy_delete_buffer(buf);
}

TEST(test_float_literal) {
    YY_BUFFER_STATE buf = yy_scan_string("3.14");
    ASSERT_INT_EQ(TOK_FLOAT_LIT, yylex());
    ASSERT_TRUE(yylval.float_val > 3.13 && yylval.float_val < 3.15);
    yy_delete_buffer(buf);
}

TEST(test_double_colon) {
    YY_BUFFER_STATE buf = yy_scan_string("::");
    ASSERT_INT_EQ(TOK_DOUBLE_COLON, yylex());
    yy_delete_buffer(buf);
}

TEST(test_gen_keyword) {
    YY_BUFFER_STATE buf = yy_scan_string("gen");
    ASSERT_INT_EQ(TOK_GEN, yylex());
    yy_delete_buffer(buf);
}

TEST(test_stop_keyword) {
    YY_BUFFER_STATE buf = yy_scan_string("stop");
    ASSERT_INT_EQ(TOK_STOP, yylex());
    yy_delete_buffer(buf);
}

TEST(test_stable_keyword) {
    YY_BUFFER_STATE buf = yy_scan_string("stable");
    ASSERT_INT_EQ(TOK_STABLE, yylex());
    yy_delete_buffer(buf);
}

TEST(test_plus_token) {
    YY_BUFFER_STATE buf = yy_scan_string("+");
    ASSERT_INT_EQ('+', yylex());
    yy_delete_buffer(buf);
}

TEST(test_full_token_sequence) {
    YY_BUFFER_STATE buf = yy_scan_string("dim M = 4;");
    ASSERT_INT_EQ(TOK_DIM, yylex());
    ASSERT_INT_EQ(TOK_IDENT, yylex());
    ASSERT_STR_EQ("M", yylval.str_val);
    free(yylval.str_val);
    ASSERT_INT_EQ('=', yylex());
    ASSERT_INT_EQ(TOK_INT_LIT, yylex());
    ASSERT_INT_EQ(4, yylval.int_val);
    ASSERT_INT_EQ(';', yylex());
    yy_delete_buffer(buf);
}

TEST(test_comment_skipped) {
    YY_BUFFER_STATE buf = yy_scan_string("// this is a comment\ndim");
    ASSERT_INT_EQ(TOK_DIM, yylex());
    yy_delete_buffer(buf);
}

TEST(test_map_reduce_keywords) {
    YY_BUFFER_STATE buf = yy_scan_string("map reduce");
    ASSERT_INT_EQ(TOK_MAP, yylex());
    ASSERT_INT_EQ(TOK_REDUCE, yylex());
    yy_delete_buffer(buf);
}

TEST(test_load_keyword) {
    YY_BUFFER_STATE buf = yy_scan_string("load");
    ASSERT_INT_EQ(TOK_LOAD, yylex());
    yy_delete_buffer(buf);
}

TEST(test_save_keyword) {
    YY_BUFFER_STATE buf = yy_scan_string("save");
    ASSERT_INT_EQ(TOK_SAVE, yylex());
    yy_delete_buffer(buf);
}

TEST(test_star_token) {
    YY_BUFFER_STATE buf = yy_scan_string("*");
    ASSERT_INT_EQ('*', yylex());
    yy_delete_buffer(buf);
}

TEST(test_arrow_token) {
    YY_BUFFER_STATE buf = yy_scan_string("->");
    ASSERT_INT_EQ(TOK_ARROW, yylex());
    yy_delete_buffer(buf);
}

TEST(test_string_literal) {
    YY_BUFFER_STATE buf = yy_scan_string("\"graph.mtx\"");
    ASSERT_INT_EQ(TOK_STRING_LIT, yylex());
    ASSERT_STR_EQ("graph.mtx", yylval.str_val);
    free(yylval.str_val);
    yy_delete_buffer(buf);
}

int main(void) {
    RUN_TEST(test_dim_keyword);
    RUN_TEST(test_tensor_keyword);
    RUN_TEST(test_identifier);
    RUN_TEST(test_int_literal);
    RUN_TEST(test_float_literal);
    RUN_TEST(test_double_colon);
    RUN_TEST(test_gen_keyword);
    RUN_TEST(test_stop_keyword);
    RUN_TEST(test_stable_keyword);
    RUN_TEST(test_plus_token);
    RUN_TEST(test_full_token_sequence);
    RUN_TEST(test_comment_skipped);
    RUN_TEST(test_map_reduce_keywords);
    RUN_TEST(test_load_keyword);
    RUN_TEST(test_save_keyword);
    RUN_TEST(test_star_token);
    RUN_TEST(test_arrow_token);
    RUN_TEST(test_string_literal);
    return test_report();
}
