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

TEST(test_parse_einsum_negated_operand) {
    YY_BUFFER_STATE buf = yy_scan_string(
        "Z[d] = A[d] . !B[d] :: map(take_left, intersect) reduce(or, union);");
    ASSERT_INT_EQ(0, yyparse());
    ASSERT_NOT_NULL(parse_result);
    AstNode *es = parse_result->data.program.statements[0];
    ASSERT_INT_EQ(NODE_EINSUM, es->type);
    ASSERT_INT_EQ(0, es->data.einsum.input1.negate);
    ASSERT_INT_EQ(1, es->data.einsum.input2.negate);
    ast_free(parse_result);
    parse_result = NULL;
    yy_delete_buffer(buf);
}

TEST(test_parse_einsum_negated_left_operand) {
    YY_BUFFER_STATE buf = yy_scan_string(
        "Z[d] = !A[d] . B[d] :: map(and, intersect) reduce(or, union);");
    ASSERT_INT_EQ(0, yyparse());
    ASSERT_NOT_NULL(parse_result);
    AstNode *es = parse_result->data.program.statements[0];
    ASSERT_INT_EQ(1, es->data.einsum.input1.negate);
    ASSERT_INT_EQ(0, es->data.einsum.input2.negate);
    ast_free(parse_result);
    parse_result = NULL;
    yy_delete_buffer(buf);
}

TEST(test_parse_single_reduce_over) {
    YY_BUFFER_STATE buf = yy_scan_string(
        "Z[d] = G[s,d] . F[s] :: map(mul,intersect) reduce(add,union, over=s);");
    ASSERT_INT_EQ(0, yyparse());
    AstNode *es = parse_result->data.program.statements[0];
    ASSERT_INT_EQ(NODE_EINSUM, es->type);
    ASSERT_NOT_NULL(es->data.einsum.reduce_over);
    ASSERT_STR_EQ("s", es->data.einsum.reduce_over);
    ast_free(parse_result); parse_result = NULL; yy_delete_buffer(buf);
}

TEST(test_parse_fused_push) {
    YY_BUFFER_STATE buf = yy_scan_string(
        "F[d] = (G[s,d] .1 F[s]) .2 P[d] "
        ":: map.1(and,intersect) reduce.1(or,union) map.2(take_left,intersect);");
    ASSERT_INT_EQ(0, yyparse());
    AstNode *es = parse_result->data.program.statements[0];
    ASSERT_INT_EQ(NODE_EINSUM_FUSED, es->type);
    EinsumFusedData *f = &es->data.einsum_fused;
    ASSERT_STR_EQ("F", f->output.tensor_name);
    ASSERT_STR_EQ("G", f->inner1.tensor_name);
    ASSERT_STR_EQ("F", f->inner2.tensor_name);
    ASSERT_STR_EQ("P", f->outer.tensor_name);
    ASSERT_INT_EQ(1, f->inter_is_left2);
    ASSERT_INT_EQ(OP_AND, f->map1_op);
    ASSERT_INT_EQ(MERGE_INTERSECT, f->map1_merge);
    ASSERT_INT_EQ(1, f->a1_has_reduce);
    ASSERT_INT_EQ(OP_OR, f->reduce1_op);
    ASSERT_INT_EQ(OP_TAKE_LEFT, f->map2_op);
    ASSERT_INT_EQ(0, f->a2_has_reduce);
    ast_free(parse_result); parse_result = NULL; yy_delete_buffer(buf);
}

TEST(test_parse_fused_pull_paren_right) {
    YY_BUFFER_STATE buf = yy_scan_string(
        "F[d] = P[d] .2 (G[s,d] .1 F[s]) "
        ":: map.1(and,intersect) map.2(and,intersect) reduce.2(or,union, over=s);");
    ASSERT_INT_EQ(0, yyparse());
    AstNode *es = parse_result->data.program.statements[0];
    ASSERT_INT_EQ(NODE_EINSUM_FUSED, es->type);
    EinsumFusedData *f = &es->data.einsum_fused;
    ASSERT_INT_EQ(0, f->inter_is_left2);
    ASSERT_INT_EQ(0, f->a1_has_reduce);
    ASSERT_INT_EQ(1, f->a2_has_reduce);
    ASSERT_STR_EQ("s", f->over2);
    ast_free(parse_result); parse_result = NULL; yy_delete_buffer(buf);
}

TEST(test_parse_fused_missing_map_fails) {
    YY_BUFFER_STATE buf = yy_scan_string(
        "F[d] = (G[s,d] .1 F[s]) .2 P[d] :: reduce.1(or,union) map.2(take_left,intersect);");
    ASSERT_INT_EQ(1, yyparse());            /* no map.1 -> parse error */
    if (parse_result) { ast_free(parse_result); parse_result = NULL; }
    yy_delete_buffer(buf);
}
TEST(test_parse_fused_bad_action_label_fails) {
    YY_BUFFER_STATE buf = yy_scan_string(
        "F[d] = (G[s,d] .1 F[s]) .2 P[d] :: map.1(and,intersect) map.3(take_left,intersect);");
    ASSERT_INT_EQ(1, yyparse());            /* map.3 invalid label */
    if (parse_result) { ast_free(parse_result); parse_result = NULL; }
    yy_delete_buffer(buf);
}
TEST(test_parse_fused_duplicate_map_fails) {
    YY_BUFFER_STATE buf = yy_scan_string(
        "F[d] = (G[s,d] .1 F[s]) .2 P[d] :: map.1(and,intersect) map.1(or,intersect) map.2(take_left,intersect);");
    ASSERT_INT_EQ(1, yyparse());            /* duplicate map.1 */
    if (parse_result) { ast_free(parse_result); parse_result = NULL; }
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
    ASSERT_INT_EQ(STOP_STABLE, s->data.stop.kind);
    ASSERT_STR_EQ("F", s->data.stop.stable_tensor);
    ast_free(parse_result);
    parse_result = NULL;
    yy_delete_buffer(buf);
}

TEST(test_parse_stop_occupancy) {
    YY_BUFFER_STATE buf = yy_scan_string("stop: ||F[i+1]|| == 0;");
    ASSERT_INT_EQ(0, yyparse());
    AstNode *s = parse_result->data.program.statements[0];
    ASSERT_INT_EQ(NODE_STOP, s->type);
    ASSERT_INT_EQ(STOP_OCCUPANCY, s->data.stop.kind);
    ASSERT_STR_EQ("F", s->data.stop.lhs.tensor_name);
    ASSERT_INT_EQ(1, s->data.stop.lhs.num_vars);
    ASSERT_STR_EQ("i", s->data.stop.lhs.rank_vars[0]);
    ASSERT_INT_EQ(1, s->data.stop.lhs.gen_offset);
    ASSERT_INT_EQ(0, s->data.stop.occupancy_rhs);
    ast_free(parse_result);
    parse_result = NULL;
    yy_delete_buffer(buf);
}

TEST(test_parse_stop_occupancy_nonzero_parses) {
    /* Parses; rejected later by semantic analysis. */
    YY_BUFFER_STATE buf = yy_scan_string("stop: ||F[i+1]|| == 3;");
    ASSERT_INT_EQ(0, yyparse());
    AstNode *s = parse_result->data.program.statements[0];
    ASSERT_INT_EQ(3, s->data.stop.occupancy_rhs);
    ast_free(parse_result);
    parse_result = NULL;
    yy_delete_buffer(buf);
}

TEST(test_parse_stop_equal) {
    YY_BUFFER_STATE buf = yy_scan_string("stop: P[i+1] == P[i];");
    ASSERT_INT_EQ(0, yyparse());
    AstNode *s = parse_result->data.program.statements[0];
    ASSERT_INT_EQ(STOP_EQUAL, s->data.stop.kind);
    ASSERT_STR_EQ("P", s->data.stop.lhs.tensor_name);
    ASSERT_INT_EQ(1, s->data.stop.lhs.gen_offset);
    ASSERT_STR_EQ("P", s->data.stop.rhs.tensor_name);
    ASSERT_STR_EQ("i", s->data.stop.rhs.rank_vars[0]);
    ASSERT_INT_EQ(0, s->data.stop.rhs.gen_offset);
    ast_free(parse_result);
    parse_result = NULL;
    yy_delete_buffer(buf);
}

TEST(test_parse_stop_unknown_predicate_fails) {
    YY_BUFFER_STATE buf = yy_scan_string("stop: settled(P);");
    ASSERT_TRUE(yyparse() != 0);
    if (parse_result) { ast_free(parse_result); parse_result = NULL; }
    yy_delete_buffer(buf);
}

TEST(test_parse_inferred_dim) {
    YY_BUFFER_STATE buf = yy_scan_string("dim N = *;\n");
    ASSERT_INT_EQ(0, yyparse());
    ASSERT_NOT_NULL(parse_result);
    AstNode *dim = parse_result->data.program.statements[0];
    ASSERT_INT_EQ(NODE_DIM_DECL, dim->type);
    ASSERT_INT_EQ(1, dim->data.dim_decl.is_inferred);
    ast_free(parse_result);
    parse_result = NULL;
    yy_delete_buffer(buf);
}

TEST(test_parse_init_load) {
    YY_BUFFER_STATE buf = yy_scan_string(
        "dim N = *;\n"
        "tensor G[R=N, C=N] : float, empty = 0.0;\n"
        "init G = load(\"graph.mtx\");\n");
    ASSERT_INT_EQ(0, yyparse());
    ASSERT_NOT_NULL(parse_result);
    AstNode *prog = parse_result;
    AstNode *init = prog->data.program.statements[2];
    ASSERT_INT_EQ(NODE_INIT, init->type);
    ASSERT_INT_EQ(1, init->data.init.is_load);
    ASSERT_STR_EQ("graph.mtx", init->data.init.path);
    ast_free(prog);
    parse_result = NULL;
    yy_delete_buffer(buf);
}

TEST(test_parse_init_seed) {
    YY_BUFFER_STATE buf = yy_scan_string("init D = seed(source=2, value=0.0);");
    ASSERT_INT_EQ(0, yyparse());
    ASSERT_NOT_NULL(parse_result);
    AstNode *init = parse_result->data.program.statements[0];
    ASSERT_INT_EQ(NODE_INIT, init->type);
    ASSERT_STR_EQ("D", init->data.init.tensor_name);
    ASSERT_INT_EQ(0, init->data.init.is_load);
    ASSERT_INT_EQ(1, init->data.init.is_seed);
    ASSERT_INT_EQ(2, init->data.init.seed_source);
    ASSERT_TRUE(init->data.init.seed_value == 0.0);
    ast_free(parse_result);
    parse_result = NULL;
    yy_delete_buffer(buf);
}

TEST(test_parse_output_directive) {
    YY_BUFFER_STATE buf = yy_scan_string("output: B, Z;\n");
    ASSERT_INT_EQ(0, yyparse());
    ASSERT_NOT_NULL(parse_result);
    AstNode *o = parse_result->data.program.statements[0];
    ASSERT_INT_EQ(NODE_OUTPUT, o->type);
    ASSERT_INT_EQ(2, o->data.output.count);
    ASSERT_STR_EQ("B", o->data.output.names[0]);
    ASSERT_STR_EQ("Z", o->data.output.names[1]);
    ast_free(parse_result);
    parse_result = NULL;
    yy_delete_buffer(buf);
}

TEST(test_parse_output_empty_list_fails) {
    YY_BUFFER_STATE buf = yy_scan_string("output: ;\n");
    ASSERT_TRUE(yyparse() != 0);
    if (parse_result) { ast_free(parse_result); parse_result = NULL; }
    yy_delete_buffer(buf);
}

TEST(test_parse_output_offset_fails) {
    /* output: takes plain tensor names; a rank offset is not a tensor name. */
    YY_BUFFER_STATE buf = yy_scan_string("output: B+1;\n");
    ASSERT_TRUE(yyparse() != 0);
    if (parse_result) { ast_free(parse_result); parse_result = NULL; }
    yy_delete_buffer(buf);
}

TEST(test_parse_save_directive) {
    YY_BUFFER_STATE buf = yy_scan_string("save: Z -> \"out.mtx\";\n");
    ASSERT_INT_EQ(0, yyparse());
    ASSERT_NOT_NULL(parse_result);
    AstNode *save = parse_result->data.program.statements[0];
    ASSERT_INT_EQ(NODE_SAVE, save->type);
    ASSERT_STR_EQ("Z", save->data.save.tensor_name);
    ASSERT_STR_EQ("out.mtx", save->data.save.path);
    ast_free(parse_result);
    parse_result = NULL;
    yy_delete_buffer(buf);
}

int main(void) {
    RUN_TEST(test_parse_dim_decl);
    RUN_TEST(test_parse_tensor_decl);
    RUN_TEST(test_parse_init);
    RUN_TEST(test_parse_einsum);
    RUN_TEST(test_parse_einsum_negated_operand);
    RUN_TEST(test_parse_einsum_negated_left_operand);
    RUN_TEST(test_parse_single_reduce_over);
    RUN_TEST(test_parse_fused_push);
    RUN_TEST(test_parse_fused_pull_paren_right);
    RUN_TEST(test_parse_fused_missing_map_fails);
    RUN_TEST(test_parse_fused_bad_action_label_fails);
    RUN_TEST(test_parse_fused_duplicate_map_fails);
    RUN_TEST(test_parse_full_program);
    RUN_TEST(test_parse_einsum_with_merge);
    RUN_TEST(test_parse_einsum_default_merge);
    RUN_TEST(test_parse_generational_rank);
    RUN_TEST(test_parse_offset_coord);
    RUN_TEST(test_parse_stop_stmt);
    RUN_TEST(test_parse_stop_occupancy);
    RUN_TEST(test_parse_stop_occupancy_nonzero_parses);
    RUN_TEST(test_parse_stop_equal);
    RUN_TEST(test_parse_stop_unknown_predicate_fails);
    RUN_TEST(test_parse_inferred_dim);
    RUN_TEST(test_parse_init_load);
    RUN_TEST(test_parse_init_seed);
    RUN_TEST(test_parse_save_directive);
    RUN_TEST(test_parse_output_directive);
    RUN_TEST(test_parse_output_empty_list_fails);
    RUN_TEST(test_parse_output_offset_fails);
    return test_report();
}
