#include "test_helpers.h"
#include "ast.h"
#include "semantic.h"
#include <limits.h>

typedef struct yy_buffer_state *YY_BUFFER_STATE;
extern YY_BUFFER_STATE yy_scan_string(const char *str);
extern void yy_delete_buffer(YY_BUFFER_STATE buf);
extern int yyparse(void);
extern AstNode *parse_result;

static AstNode *parse(const char *src) {
    YY_BUFFER_STATE buf = yy_scan_string(src);
    yyparse();
    yy_delete_buffer(buf);
    AstNode *result = parse_result;
    parse_result = NULL;
    return result;
}

static SemanticResult analyze_source(const char *src) {
    AstNode *ast = parse(src);
    SemanticResult res = semantic_analyze(ast);
    ast_free(ast);
    return res;
}

TEST(test_valid_gemm) {
    AstNode *ast = parse(
        "dim M = 2; dim K = 3; dim N = 2;\n"
        "tensor Z[R=M, C=N] : float, empty = 0.0;\n"
        "tensor A[R=K, C=M] : float, empty = 0.0;\n"
        "tensor B[R=K, C=N] : float, empty = 0.0;\n"
        "init A = [[1, 2], [3, 4], [5, 6]];\n"
        "init B = [[7, 8], [9, 10], [11, 12]];\n"
        "Z[m,n] = A[k,m] . B[k,n] :: map(mul) reduce(add);\n"
    );
    SemanticResult res = semantic_analyze(ast);
    ASSERT_INT_EQ(0, res.error_count);
    ASSERT_NOT_NULL(res.symtab);
    ASSERT_INT_EQ(2, res.symtab->num_free_indices);
    ASSERT_INT_EQ(1, res.symtab->num_reduction_indices);
    semantic_result_free(&res);
    ast_free(ast);
}

TEST(test_undeclared_dim) {
    AstNode *ast = parse("tensor Z[R=M] : float, empty = 0.0;\n");
    SemanticResult res = semantic_analyze(ast);
    ASSERT_TRUE(res.error_count > 0);
    semantic_result_free(&res);
    ast_free(ast);
}

TEST(test_undeclared_tensor_in_init) {
    AstNode *ast = parse("init X = [1, 2, 3];\n");
    SemanticResult res = semantic_analyze(ast);
    ASSERT_TRUE(res.error_count > 0);
    semantic_result_free(&res);
    ast_free(ast);
}

TEST(test_dimension_mismatch) {
    AstNode *ast = parse(
        "dim M = 2; dim K = 3; dim N = 4; dim P = 5;\n"
        "tensor Z[R=M, C=N] : float, empty = 0.0;\n"
        "tensor A[R=K, C=M] : float, empty = 0.0;\n"
        "tensor B[R=K, C=P] : float, empty = 0.0;\n"
        "Z[m,n] = A[k,m] . B[k,n] :: map(mul) reduce(add);\n"
    );
    SemanticResult res = semantic_analyze(ast);
    ASSERT_TRUE(res.error_count > 0);
    semantic_result_free(&res);
    ast_free(ast);
}

TEST(test_init_shape_mismatch) {
    AstNode *ast = parse(
        "dim M = 2; dim N = 3;\n"
        "tensor A[R=M, C=N] : float, empty = 0.0;\n"
        "init A = [[1, 2], [3, 4]];\n"
    );
    SemanticResult res = semantic_analyze(ast);
    ASSERT_TRUE(res.error_count > 0);
    semantic_result_free(&res);
    ast_free(ast);
}

TEST(test_identity_add_float) {
    ASSERT_TRUE(compute_op_identity(OP_ADD, TYPE_FLOAT) == 0.0);
}
TEST(test_identity_mul_float) {
    ASSERT_TRUE(compute_op_identity(OP_MUL, TYPE_FLOAT) == 1.0);
}
TEST(test_identity_min_float) {
    ASSERT_TRUE(compute_op_identity(OP_MIN, TYPE_FLOAT) > 1e308); /* +inf */
}
TEST(test_identity_max_float) {
    ASSERT_TRUE(compute_op_identity(OP_MAX, TYPE_FLOAT) < -1e308); /* -inf */
}
TEST(test_identity_min_int) {
    ASSERT_TRUE(compute_op_identity(OP_MIN, TYPE_INTEGER) == (double)INT_MAX);
}
TEST(test_identity_and_bool) {
    ASSERT_TRUE(compute_op_identity(OP_AND, TYPE_BOOLEAN) == 1.0);
}
TEST(test_identity_or_bool) {
    ASSERT_TRUE(compute_op_identity(OP_OR, TYPE_BOOLEAN) == 0.0);
}

TEST(test_generational_valid) {
    AstNode *ast = parse(
        "dim N = 4;\n"
        "tensor G[R=N, C=N]   : float, empty = 0.0;\n"
        "tensor F[gen I, R=N] : float, empty = 0.0;\n"
        "init F = [1.0, 0.0, 0.0, 0.0];\n"
        "F[i+1, d] = G[s,d] . F[i, s] :: map(mul, intersect) reduce(max, union);\n"
        "stop: stable(F);\n"
    );
    SemanticResult res = semantic_analyze(ast);
    ASSERT_INT_EQ(0, res.error_count);
    ASSERT_INT_EQ(1, res.symtab->is_iterative);
    ASSERT_STR_EQ("F", res.symtab->gen_tensor);
    ASSERT_STR_EQ("i", res.symtab->gen_rank_var);
    ASSERT_STR_EQ("F", res.symtab->stop_watch);
    ASSERT_INT_EQ(1, res.symtab->num_free_indices);       /* d */
    ASSERT_INT_EQ(1, res.symtab->num_reduction_indices);  /* s */
    ASSERT_STR_EQ("d", res.symtab->free_indices[0].name);
    ASSERT_STR_EQ("s", res.symtab->reduction_indices[0].name);
    semantic_result_free(&res);
    ast_free(ast);
}

TEST(test_generational_stop_wrong_tensor) {
    AstNode *ast = parse(
        "dim N = 4;\n"
        "tensor G[R=N, C=N]   : float, empty = 0.0;\n"
        "tensor F[gen I, R=N] : float, empty = 0.0;\n"
        "F[i+1, d] = G[s,d] . F[i, s] :: map(mul, intersect) reduce(max, union);\n"
        "stop: stable(G);\n"
    );
    SemanticResult res = semantic_analyze(ast);
    ASSERT_TRUE(res.error_count > 0);
    semantic_result_free(&res);
    ast_free(ast);
}

TEST(test_generational_missing_stop) {
    AstNode *ast = parse(
        "dim N = 4;\n"
        "tensor G[R=N, C=N]   : float, empty = 0.0;\n"
        "tensor F[gen I, R=N] : float, empty = 0.0;\n"
        "F[i+1, d] = G[s,d] . F[i, s] :: map(mul, intersect) reduce(max, union);\n"
    );
    SemanticResult res = semantic_analyze(ast);
    ASSERT_TRUE(res.error_count > 0);
    semantic_result_free(&res);
    ast_free(ast);
}

TEST(test_generational_bad_offset) {
    AstNode *ast = parse(
        "dim N = 4;\n"
        "tensor G[R=N, C=N]   : float, empty = 0.0;\n"
        "tensor F[gen I, R=N] : float, empty = 0.0;\n"
        "F[i+2, d] = G[s,d] . F[i, s] :: map(mul, intersect) reduce(max, union);\n"
        "stop: stable(F);\n"
    );
    SemanticResult res = semantic_analyze(ast);
    ASSERT_TRUE(res.error_count > 0);
    semantic_result_free(&res);
    ast_free(ast);
}

TEST(test_generational_init_spatial_shape) {
    AstNode *ast = parse(
        "dim N = 4;\n"
        "tensor G[R=N, C=N]   : float, empty = 0.0;\n"
        "tensor F[gen I, R=N] : float, empty = 0.0;\n"
        "init F = [1.0, 0.0, 0.0, 0.0];\n"
        "F[i+1, d] = G[s,d] . F[i, s] :: map(mul, intersect) reduce(max, union);\n"
        "stop: stable(F);\n"
    );
    SemanticResult res = semantic_analyze(ast);
    ASSERT_INT_EQ(0, res.error_count);
    semantic_result_free(&res);
    ast_free(ast);
}

TEST(test_generational_two_gen_tensors) {
    AstNode *ast = parse(
        "dim N = 4;\n"
        "tensor F[gen I, R=N] : float, empty = 0.0;\n"
        "tensor H[gen J, R=N] : float, empty = 0.0;\n"
        "F[i+1, d] = H[j, d] . F[i, d] :: map(mul, intersect) reduce(max, union);\n"
        "stop: stable(F);\n"
    );
    SemanticResult res = semantic_analyze(ast);
    ASSERT_TRUE(res.error_count > 0);
    semantic_result_free(&res);
    ast_free(ast);
}

TEST(test_cascade_valid_bellman_ford) {
    AstNode *ast = parse(
        "dim N = 4;\n"
        "tensor W[R=N, C=N]   : float, empty = inf;\n"
        "tensor T[gen I, R=N] : float, empty = inf;\n"
        "tensor D[gen I, R=N] : float, empty = inf;\n"
        "init D = [0.0, inf, inf, inf];\n"
        "T[i, d]   = W[s,d] . D[i, s] :: map(add, intersect) reduce(min, union);\n"
        "D[i+1, d] = D[i, d] . T[i, d] :: map(min, union) reduce(min, union);\n"
        "stop: stable(D);\n"
    );
    SemanticResult res = semantic_analyze(ast);
    ASSERT_INT_EQ(0, res.error_count);
    ASSERT_INT_EQ(1, res.symtab->is_iterative);
    ASSERT_STR_EQ("D", res.symtab->gen_tensor);
    ASSERT_STR_EQ("i", res.symtab->gen_rank_var);
    ASSERT_STR_EQ("D", res.symtab->stop_watch);
    ASSERT_INT_EQ(1, res.symtab->num_intermediates);
    ASSERT_STR_EQ("T", res.symtab->intermediates[0]);
    ASSERT_INT_EQ(2, res.symtab->num_einsums);
    ASSERT_INT_EQ(1, res.symtab->einsum_class[0].num_free_indices);
    ASSERT_STR_EQ("d", res.symtab->einsum_class[0].free_indices[0].name);
    ASSERT_INT_EQ(1, res.symtab->einsum_class[0].num_reduction_indices);
    ASSERT_STR_EQ("s", res.symtab->einsum_class[0].reduction_indices[0].name);
    ASSERT_INT_EQ(1, res.symtab->einsum_class[1].num_free_indices);
    ASSERT_STR_EQ("d", res.symtab->einsum_class[1].free_indices[0].name);
    ASSERT_INT_EQ(0, res.symtab->einsum_class[1].num_reduction_indices);
    semantic_result_free(&res);
    ast_free(ast);
}

TEST(test_cascade_two_cross_gen_tensors) {
    /* Reachability BFS: F and P are both cross-generational (written at i+1).
       Under C1 this is valid, and P[i+1] reads the just-written F[i+1]. */
    AstNode *ast = parse(
        "dim N = 6;\n"
        "tensor G[R=N, C=N]   : float, empty = 0.0;\n"
        "tensor F[gen I, R=N] : float, empty = 0.0;\n"
        "tensor P[gen I, R=N] : float, empty = 0.0;\n"
        "init G = [[0.0,1.0,1.0,0.0,0.0,0.0],[0.0,0.0,0.0,1.0,0.0,0.0],"
        "[0.0,0.0,0.0,1.0,1.0,0.0],[0.0,0.0,0.0,0.0,0.0,1.0],"
        "[0.0,0.0,0.0,0.0,0.0,1.0],[0.0,0.0,0.0,0.0,0.0,0.0]];\n"
        "init F = [1.0,0.0,0.0,0.0,0.0,0.0];\n"
        "init P = [1.0,0.0,0.0,0.0,0.0,0.0];\n"
        "F[i+1, d] = G[s,d] . F[i,s]   :: map(and, intersect) reduce(or, union);\n"
        "P[i+1, d] = P[i,d] . F[i+1,d] :: map(or, union) reduce(or, union);\n"
        "stop: stable(P);\n"
    );
    SemanticResult res = semantic_analyze(ast);
    ASSERT_INT_EQ(0, res.error_count);
    ASSERT_INT_EQ(2, res.symtab->num_gen_tensors);
    semantic_result_free(&res);
    ast_free(ast);
}

TEST(test_cascade_i1_read_before_write) {
    /* P reads F[i+1] but F is written by a LATER Einsum -> SSA error. */
    AstNode *ast = parse(
        "dim N = 2;\n"
        "tensor G[R=N, C=N]   : float, empty = 0.0;\n"
        "tensor F[gen I, R=N] : float, empty = 0.0;\n"
        "tensor P[gen I, R=N] : float, empty = 0.0;\n"
        "init F = [1.0,0.0];\n"
        "init P = [1.0,0.0];\n"
        "P[i+1, d] = P[i,d] . F[i+1,d] :: map(or, union) reduce(or, union);\n"
        "F[i+1, d] = G[s,d] . F[i,s]   :: map(and, intersect) reduce(or, union);\n"
        "stop: stable(P);\n"
    );
    SemanticResult res = semantic_analyze(ast);
    ASSERT_TRUE(res.error_count > 0);
    semantic_result_free(&res);
    ast_free(ast);
}

TEST(test_cascade_cross_gen_missing_init) {
    /* Two cross-gen tensors but P has no init -> error. */
    AstNode *ast = parse(
        "dim N = 2;\n"
        "tensor G[R=N, C=N]   : float, empty = 0.0;\n"
        "tensor F[gen I, R=N] : float, empty = 0.0;\n"
        "tensor P[gen I, R=N] : float, empty = 0.0;\n"
        "init F = [1.0,0.0];\n"
        "F[i+1, d] = G[s,d] . F[i,s]   :: map(and, intersect) reduce(or, union);\n"
        "P[i+1, d] = P[i,d] . F[i+1,d] :: map(or, union) reduce(or, union);\n"
        "stop: stable(P);\n"
    );
    SemanticResult res = semantic_analyze(ast);
    ASSERT_TRUE(res.error_count > 0);
    semantic_result_free(&res);
    ast_free(ast);
}

TEST(test_cascade_intermediate_with_init) {
    AstNode *ast = parse(
        "dim N = 4;\n"
        "tensor W[R=N, C=N]   : float, empty = inf;\n"
        "tensor T[gen I, R=N] : float, empty = inf;\n"
        "tensor D[gen I, R=N] : float, empty = inf;\n"
        "init D = [0.0, inf, inf, inf];\n"
        "init T = [0.0, inf, inf, inf];\n"
        "T[i, d]   = W[s,d] . D[i, s] :: map(add, intersect) reduce(min, union);\n"
        "D[i+1, d] = D[i, d] . T[i, d] :: map(min, union) reduce(min, union);\n"
        "stop: stable(D);\n"
    );
    SemanticResult res = semantic_analyze(ast);
    ASSERT_TRUE(res.error_count > 0);
    semantic_result_free(&res);
    ast_free(ast);
}

TEST(test_cascade_intermediate_read_before_write) {
    AstNode *ast = parse(
        "dim N = 4;\n"
        "tensor W[R=N, C=N]   : float, empty = inf;\n"
        "tensor T[gen I, R=N] : float, empty = inf;\n"
        "tensor D[gen I, R=N] : float, empty = inf;\n"
        "init D = [0.0, inf, inf, inf];\n"
        "D[i+1, d] = D[i, d] . T[i, d] :: map(min, union) reduce(min, union);\n"
        "T[i, d]   = W[s,d] . D[i, s] :: map(add, intersect) reduce(min, union);\n"
        "stop: stable(D);\n"
    );
    SemanticResult res = semantic_analyze(ast);
    ASSERT_TRUE(res.error_count > 0);
    semantic_result_free(&res);
    ast_free(ast);
}

TEST(test_cascade_cross_gen_not_read) {
    AstNode *ast = parse(
        "dim N = 4;\n"
        "tensor W[R=N, C=N]   : float, empty = inf;\n"
        "tensor D[gen I, R=N] : float, empty = inf;\n"
        "init D = [0.0, inf, inf, inf];\n"
        "D[i+1, d] = W[s,d] . W[s,d] :: map(add, intersect) reduce(min, union);\n"
        "stop: stable(D);\n"
    );
    SemanticResult res = semantic_analyze(ast);
    ASSERT_TRUE(res.error_count > 0);
    semantic_result_free(&res);
    ast_free(ast);
}

TEST(test_take_left_map_valid) {
    AstNode *ast = parse(
        "dim N = 3;\n"
        "tensor Z[R=N] : float, empty = inf;\n"
        "tensor A[R=N] : float, empty = inf;\n"
        "tensor B[R=N] : float, empty = inf;\n"
        "init A = [0.0, 2.0, inf];\n"
        "init B = [inf, 5.0, inf];\n"
        "Z[n] = A[n] . B[n] :: map(take_left, take_left_only) reduce(min);\n"
    );
    SemanticResult res = semantic_analyze(ast);
    ASSERT_INT_EQ(0, res.error_count);
    semantic_result_free(&res);
    ast_free(ast);
}

TEST(test_take_left_reduce_is_error) {
    AstNode *ast = parse(
        "dim K = 3; dim M = 2;\n"
        "tensor Z[R=M] : float, empty = inf;\n"
        "tensor A[R=K, C=M] : float, empty = inf;\n"
        "tensor B[R=K, C=M] : float, empty = inf;\n"
        "Z[m] = A[k,m] . B[k,m] :: map(mul, intersect) reduce(take_left, union);\n"
    );
    SemanticResult res = semantic_analyze(ast);
    ASSERT_TRUE(res.error_count > 0);
    semantic_result_free(&res);
    ast_free(ast);
}

TEST(test_semantic_records_inferred_dim) {
    SemanticResult r = analyze_source(
        "dim N = *;\n"
        "tensor G[R=N, C=N] : float, empty = 0.0;\n"
        "tensor F[R=N] : float, empty = 0.0;\n"
        "tensor Z[R=N] : float, empty = 0.0;\n"
        "init G = load(\"g.mtx\");\n"
        "init F = load(\"f.mtx\");\n"
        "Z[d] = G[s,d] . F[s] :: map(mul, intersect) reduce(add, union);\n");
    ASSERT_INT_EQ(0, r.error_count);
    semantic_result_free(&r);
}

TEST(test_semantic_literal_init_on_inferred_dim_ok) {
    /* A literal init on a tensor whose dim is inferred must NOT be shape-checked
       at compile time (size is only known at runtime from the loaded tensor). */
    SemanticResult r = analyze_source(
        "dim N = *;\n"
        "tensor G[R=N, C=N] : float, empty = 0.0;\n"
        "tensor F[R=N] : float, empty = 0.0;\n"
        "tensor Z[R=N] : float, empty = 0.0;\n"
        "init G = load(\"g.mtx\");\n"
        "init F = [1.0, 0.0, 0.0, 0.0];\n"
        "Z[d] = G[s,d] . F[s] :: map(mul, intersect) reduce(add, union);\n");
    ASSERT_INT_EQ(0, r.error_count);
    semantic_result_free(&r);
}

TEST(test_semantic_unpinned_inferred_dim_errors) {
    SemanticResult r = analyze_source(
        "dim N = *;\n"
        "tensor F[R=N] : float, empty = 0.0;\n"
        "init F = [1.0, 0.0];\n"
        "Z[d] = F[d] . F[d] :: map(add, union) reduce(add, union);\n");
    ASSERT_TRUE(r.error_count > 0);
    semantic_result_free(&r);
}

TEST(test_semantic_save_unknown_tensor_errors) {
    SemanticResult r = analyze_source(
        "dim N = 2;\n"
        "tensor F[R=N] : float, empty = 0.0;\n"
        "init F = [1.0, 0.0];\n"
        "Z[d] = F[d] . F[d] :: map(add, union) reduce(add, union);\n"
        "save: Q -> \"out.mtx\";\n");
    ASSERT_TRUE(r.error_count > 0);
    semantic_result_free(&r);
}

TEST(test_semantic_seed_rank1_ok) {
    AstNode *ast = parse(
        "dim N = *;\n"
        "tensor G[R=N, C=N] : float, empty = 0.0;\n"
        "tensor D[R=N]      : float, empty = inf;\n"
        "init G = load(\"g.mtx\");\n"
        "init D = seed(source=0, value=0.0);\n"
        "D[d] = G[s,d] . D[s] :: map(add, intersect) reduce(min, union);\n"
    );
    SemanticResult sem = semantic_analyze(ast);
    ASSERT_INT_EQ(0, sem.error_count);
    semantic_result_free(&sem);
    ast_free(ast);
}

TEST(test_semantic_seed_2d_rejected) {
    AstNode *ast = parse(
        "dim N = *;\n"
        "tensor G[R=N, C=N] : float, empty = 0.0;\n"
        "tensor D[R=N, C=N] : float, empty = inf;\n"
        "init G = load(\"g.mtx\");\n"
        "init D = seed(source=0, value=0.0);\n"
        "D[a,b] = G[a,b] . G[a,b] :: map(add, intersect) reduce(min, union);\n"
    );
    SemanticResult sem = semantic_analyze(ast);
    ASSERT_TRUE(sem.error_count > 0);
    semantic_result_free(&sem);
    ast_free(ast);
}

/* A generational seed tensor D[gen I, R=N] is data-1-D (num_ranks==2 counting
   the gen axis) and must be accepted. */
TEST(test_semantic_seed_generational_ok) {
    AstNode *ast = parse(
        "dim N = *;\n"
        "tensor W[R=N, C=N]   : float, empty = 0.0;\n"
        "tensor D[gen I, R=N] : float, empty = inf;\n"
        "init W = load(\"g.mtx\");\n"
        "init D = seed(source=0, value=0.0);\n"
        "D[i+1, d] = W[s,d] . D[i, s] :: map(add, intersect) reduce(min, union);\n"
        "stop: stable(D);\n"
    );
    SemanticResult sem = semantic_analyze(ast);
    ASSERT_INT_EQ(0, sem.error_count);
    semantic_result_free(&sem);
    ast_free(ast);
}

TEST(test_semantic_fused_push_classification) {
    AstNode *ast = parse(
        "dim N = 3;\n"
        "tensor F[R=N] : float, empty = 0.0;\n"
        "tensor G[R=N, C=N] : float, empty = 0.0;\n"
        "tensor P[R=N] : float, empty = 0.0;\n"
        "init G = [[0,0,0],[0,0,0],[0,0,0]];\n"
        "init F = [1.0,0.0,0.0];\n"
        "init P = [1.0,0.0,0.0];\n"
        "F[d] = (G[s,d] .1 F[s]) .2 P[d] "
        ":: map.1(and,intersect) reduce.1(or,union) map.2(take_left,intersect);\n");
    SemanticResult sem = semantic_analyze(ast);
    ASSERT_INT_EQ(0, sem.error_count);
    EinsumClass *ec = &sem.symtab->einsum_class[0];
    ASSERT_INT_EQ(1, ec->is_fused);
    ASSERT_INT_EQ(1, ec->num_free_indices);
    ASSERT_STR_EQ("d", ec->free_indices[0].name);
    ASSERT_INT_EQ(1, ec->num_a1_reduce);
    ASSERT_STR_EQ("s", ec->a1_reduce[0].name);
    ASSERT_INT_EQ(0, ec->num_a2_reduce);
    semantic_result_free(&sem); ast_free(ast);
}
TEST(test_semantic_fused_missing_reduce_errors) {
    AstNode *ast = parse(
        "dim N = 3;\n"
        "tensor F[R=N] : float, empty = 0.0;\n"
        "tensor G[R=N, C=N] : float, empty = 0.0;\n"
        "tensor P[R=N] : float, empty = 0.0;\n"
        "init G = [[0,0,0],[0,0,0],[0,0,0]];\n"
        "init F = [1.0,0.0,0.0];\n"
        "init P = [1.0,0.0,0.0];\n"
        "F[d] = (G[s,d] .1 F[s]) .2 P[d] "
        ":: map.1(and,intersect) map.2(take_left,intersect);\n");
    SemanticResult sem = semantic_analyze(ast);
    ASSERT_TRUE(sem.error_count > 0);
    semantic_result_free(&sem); ast_free(ast);
}
TEST(test_semantic_fused_bad_over_errors) {
    AstNode *ast = parse(
        "dim N = 3;\n"
        "tensor F[R=N] : float, empty = 0.0;\n"
        "tensor G[R=N, C=N] : float, empty = 0.0;\n"
        "tensor P[R=N] : float, empty = 0.0;\n"
        "init G = [[0,0,0],[0,0,0],[0,0,0]];\n"
        "init F = [1.0,0.0,0.0];\n"
        "init P = [1.0,0.0,0.0];\n"
        "F[d] = (G[s,d] .1 F[s]) .2 P[d] "
        ":: map.1(and,intersect) reduce.1(or,union, over=d) map.2(take_left,intersect);\n");
    SemanticResult sem = semantic_analyze(ast);
    ASSERT_TRUE(sem.error_count > 0);
    semantic_result_free(&sem); ast_free(ast);
}

int main(void) {
    RUN_TEST(test_valid_gemm);
    RUN_TEST(test_undeclared_dim);
    RUN_TEST(test_undeclared_tensor_in_init);
    RUN_TEST(test_dimension_mismatch);
    RUN_TEST(test_init_shape_mismatch);
    RUN_TEST(test_identity_add_float);
    RUN_TEST(test_identity_mul_float);
    RUN_TEST(test_identity_min_float);
    RUN_TEST(test_identity_max_float);
    RUN_TEST(test_identity_min_int);
    RUN_TEST(test_identity_and_bool);
    RUN_TEST(test_identity_or_bool);
    RUN_TEST(test_generational_valid);
    RUN_TEST(test_generational_stop_wrong_tensor);
    RUN_TEST(test_generational_missing_stop);
    RUN_TEST(test_generational_bad_offset);
    RUN_TEST(test_generational_init_spatial_shape);
    RUN_TEST(test_generational_two_gen_tensors);
    RUN_TEST(test_cascade_valid_bellman_ford);
    RUN_TEST(test_cascade_two_cross_gen_tensors);
    RUN_TEST(test_cascade_i1_read_before_write);
    RUN_TEST(test_cascade_cross_gen_missing_init);
    RUN_TEST(test_cascade_intermediate_with_init);
    RUN_TEST(test_cascade_intermediate_read_before_write);
    RUN_TEST(test_cascade_cross_gen_not_read);
    RUN_TEST(test_take_left_map_valid);
    RUN_TEST(test_take_left_reduce_is_error);
    RUN_TEST(test_semantic_records_inferred_dim);
    RUN_TEST(test_semantic_literal_init_on_inferred_dim_ok);
    RUN_TEST(test_semantic_unpinned_inferred_dim_errors);
    RUN_TEST(test_semantic_save_unknown_tensor_errors);
    RUN_TEST(test_semantic_seed_rank1_ok);
    RUN_TEST(test_semantic_seed_2d_rejected);
    RUN_TEST(test_semantic_seed_generational_ok);
    RUN_TEST(test_semantic_fused_push_classification);
    RUN_TEST(test_semantic_fused_missing_reduce_errors);
    RUN_TEST(test_semantic_fused_bad_over_errors);
    return test_report();
}
