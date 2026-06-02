#include "test_helpers.h"
#include "ast.h"

typedef struct yy_buffer_state *YY_BUFFER_STATE;
extern YY_BUFFER_STATE yy_scan_string(const char *str);
extern void yy_delete_buffer(YY_BUFFER_STATE buf);
extern int yyparse(void);
extern AstNode *parse_result;

TEST(test_parse_dim_decl) {
    YY_BUFFER_STATE buf = yy_scan_string("dim M = 4;");
    ASSERT_INT_EQ(0, yyparse());
    ASSERT_NOT_NULL(parse_result);
    ASSERT_INT_EQ(NODE_PROGRAM, parse_result->type);
    ASSERT_INT_EQ(1, parse_result->data.program.count);
    AstNode *dim = parse_result->data.program.statements[0];
    ASSERT_INT_EQ(NODE_DIM_DECL, dim->type);
    ASSERT_STR_EQ("M", dim->data.dim_decl.name);
    ASSERT_INT_EQ(4, dim->data.dim_decl.size);
    ast_free(parse_result);
    parse_result = NULL;
    yy_delete_buffer(buf);
}

TEST(test_parse_tensor_decl) {
    YY_BUFFER_STATE buf = yy_scan_string(
        "tensor Z[R=M, C=N] : float, empty = 0.0;");
    ASSERT_INT_EQ(0, yyparse());
    ASSERT_NOT_NULL(parse_result);
    AstNode *td = parse_result->data.program.statements[0];
    ASSERT_INT_EQ(NODE_TENSOR_DECL, td->type);
    ASSERT_STR_EQ("Z", td->data.tensor_decl.name);
    ASSERT_INT_EQ(2, td->data.tensor_decl.num_ranks);
    ASSERT_STR_EQ("R", td->data.tensor_decl.ranks[0].rank_name);
    ASSERT_STR_EQ("M", td->data.tensor_decl.ranks[0].dim_name);
    ASSERT_STR_EQ("C", td->data.tensor_decl.ranks[1].rank_name);
    ASSERT_STR_EQ("N", td->data.tensor_decl.ranks[1].dim_name);
    ASSERT_INT_EQ(TYPE_FLOAT, td->data.tensor_decl.datatype);
    ast_free(parse_result);
    parse_result = NULL;
    yy_delete_buffer(buf);
}

TEST(test_parse_init) {
    YY_BUFFER_STATE buf = yy_scan_string("init A = [[1, 2], [3, 4]];");
    ASSERT_INT_EQ(0, yyparse());
    ASSERT_NOT_NULL(parse_result);
    AstNode *init = parse_result->data.program.statements[0];
    ASSERT_INT_EQ(NODE_INIT, init->type);
    ASSERT_STR_EQ("A", init->data.init.tensor_name);
    ASSERT_TRUE(init->data.init.data->is_list);
    ASSERT_INT_EQ(2, init->data.init.data->data.list.count);
    ast_free(parse_result);
    parse_result = NULL;
    yy_delete_buffer(buf);
}

TEST(test_parse_einsum) {
    YY_BUFFER_STATE buf = yy_scan_string(
        "Z[m,n] = A[k,m] . B[k,n] :: map(mul) reduce(add);");
    ASSERT_INT_EQ(0, yyparse());
    ASSERT_NOT_NULL(parse_result);
    AstNode *es = parse_result->data.program.statements[0];
    ASSERT_INT_EQ(NODE_EINSUM, es->type);
    ASSERT_STR_EQ("Z", es->data.einsum.output.tensor_name);
    ASSERT_INT_EQ(2, es->data.einsum.output.num_vars);
    ASSERT_STR_EQ("m", es->data.einsum.output.rank_vars[0]);
    ASSERT_STR_EQ("n", es->data.einsum.output.rank_vars[1]);
    ASSERT_STR_EQ("A", es->data.einsum.input1.tensor_name);
    ASSERT_STR_EQ("B", es->data.einsum.input2.tensor_name);
    ASSERT_INT_EQ(OP_MUL, es->data.einsum.map_op);
    ASSERT_INT_EQ(OP_ADD, es->data.einsum.reduce_op);
    ast_free(parse_result);
    parse_result = NULL;
    yy_delete_buffer(buf);
}

TEST(test_parse_full_program) {
    const char *src =
        "dim M = 2;\n"
        "dim N = 2;\n"
        "dim K = 3;\n"
        "tensor Z[R=M, C=N] : float, empty = 0.0;\n"
        "tensor A[R=K, C=M] : float, empty = 0.0;\n"
        "tensor B[R=K, C=N] : float, empty = 0.0;\n"
        "init A = [[1, 2], [3, 4], [5, 6]];\n"
        "init B = [[7, 8], [9, 10], [11, 12]];\n"
        "Z[m,n] = A[k,m] . B[k,n] :: map(mul) reduce(add);\n";
    YY_BUFFER_STATE buf = yy_scan_string(src);
    ASSERT_INT_EQ(0, yyparse());
    ASSERT_NOT_NULL(parse_result);
    ASSERT_INT_EQ(9, parse_result->data.program.count);
    ast_free(parse_result);
    parse_result = NULL;
    yy_delete_buffer(buf);
}

TEST(test_parse_einsum_with_merge) {
    const char *src =
        "dim D = 3; dim S = 3;\n"
        "tensor F1[R=D] : float, empty = inf;\n"
        "tensor G[R=S, C=D] : float, empty = 0.0;\n"
        "tensor F[R=S]  : float, empty = inf;\n"
        "F1[d] = G[s,d] . F[s] :: map(add, intersect) reduce(min, union);\n";
    YY_BUFFER_STATE buf = yy_scan_string(src);
    ASSERT_INT_EQ(0, yyparse());
    yy_delete_buffer(buf);
    AstNode *prog = parse_result;
    ASSERT_NOT_NULL(prog);
    /* find the einsum statement (last child) */
    int n = prog->data.program.count;
    AstNode *ein = prog->data.program.statements[n - 1];
    ASSERT_INT_EQ(NODE_EINSUM, ein->type);
    ASSERT_INT_EQ(OP_ADD, ein->data.einsum.map_op);
    ASSERT_INT_EQ(MERGE_INTERSECT, ein->data.einsum.map_merge_op);
    ASSERT_INT_EQ(OP_MIN, ein->data.einsum.reduce_op);
    ASSERT_INT_EQ(MERGE_UNION, ein->data.einsum.reduce_merge_op);
    ast_free(prog);
    parse_result = NULL;
}

TEST(test_parse_einsum_default_merge) {
    const char *src =
        "dim M = 2; dim K = 2; dim N = 2;\n"
        "tensor Z[R=M, C=N] : float, empty = 0.0;\n"
        "tensor A[R=K, C=M] : float, empty = 0.0;\n"
        "tensor B[R=K, C=N] : float, empty = 0.0;\n"
        "Z[m,n] = A[k,m] . B[k,n] :: map(mul) reduce(add);\n";
    YY_BUFFER_STATE buf = yy_scan_string(src);
    ASSERT_INT_EQ(0, yyparse());
    yy_delete_buffer(buf);
    AstNode *prog = parse_result;
    ASSERT_NOT_NULL(prog);
    int n = prog->data.program.count;
    AstNode *ein = prog->data.program.statements[n - 1];
    ASSERT_INT_EQ(MERGE_PASSTHROUGH, ein->data.einsum.map_merge_op);
    ASSERT_INT_EQ(MERGE_PASSTHROUGH, ein->data.einsum.reduce_merge_op);
    ast_free(prog);
    parse_result = NULL;
}

TEST(test_parse_generational_rank) {
    YY_BUFFER_STATE buf = yy_scan_string(
        "tensor F[gen I, R=N] : float, empty = 0.0;");
    ASSERT_INT_EQ(0, yyparse());
    ASSERT_NOT_NULL(parse_result);
    AstNode *td = parse_result->data.program.statements[0];
    ASSERT_INT_EQ(NODE_TENSOR_DECL, td->type);
    ASSERT_INT_EQ(2, td->data.tensor_decl.num_ranks);
    ASSERT_INT_EQ(1, td->data.tensor_decl.ranks[0].is_generational);
    ASSERT_STR_EQ("I", td->data.tensor_decl.ranks[0].rank_name);
    ASSERT_INT_EQ(0, td->data.tensor_decl.ranks[1].is_generational);
    ASSERT_STR_EQ("R", td->data.tensor_decl.ranks[1].rank_name);
    ASSERT_STR_EQ("N", td->data.tensor_decl.ranks[1].dim_name);
    ast_free(parse_result);
    parse_result = NULL;
    yy_delete_buffer(buf);
}

TEST(test_parse_offset_coord) {
    YY_BUFFER_STATE buf = yy_scan_string(
        "F[i+1, d] = G[s,d] . F[i, s] :: map(mul, intersect) reduce(max, union);");
    ASSERT_INT_EQ(0, yyparse());
    ASSERT_NOT_NULL(parse_result);
    AstNode *e = parse_result->data.program.statements[0];
    ASSERT_INT_EQ(NODE_EINSUM, e->type);
    ASSERT_INT_EQ(0, e->data.einsum.output.gen_axis);
    ASSERT_INT_EQ(1, e->data.einsum.output.gen_offset);
    ASSERT_STR_EQ("i", e->data.einsum.output.rank_vars[0]);
    ASSERT_STR_EQ("d", e->data.einsum.output.rank_vars[1]);
    ASSERT_INT_EQ(-1, e->data.einsum.input2.gen_axis);
    ASSERT_INT_EQ(0, e->data.einsum.input2.gen_offset);
    ast_free(parse_result);
    parse_result = NULL;
    yy_delete_buffer(buf);
}

TEST(test_parse_stop_stmt) {
    YY_BUFFER_STATE buf = yy_scan_string("stop: stable(F);");
    ASSERT_INT_EQ(0, yyparse());
    ASSERT_NOT_NULL(parse_result);
    AstNode *s = parse_result->data.program.statements[0];
    ASSERT_INT_EQ(NODE_STOP, s->type);
    ASSERT_STR_EQ("F", s->data.stop.watch_tensor);
    ast_free(parse_result);
    parse_result = NULL;
    yy_delete_buffer(buf);
}

int main(void) {
    RUN_TEST(test_parse_dim_decl);
    RUN_TEST(test_parse_tensor_decl);
    RUN_TEST(test_parse_init);
    RUN_TEST(test_parse_einsum);
    RUN_TEST(test_parse_full_program);
    RUN_TEST(test_parse_einsum_with_merge);
    RUN_TEST(test_parse_einsum_default_merge);
    RUN_TEST(test_parse_generational_rank);
    RUN_TEST(test_parse_offset_coord);
    RUN_TEST(test_parse_stop_stmt);
    return test_report();
}
