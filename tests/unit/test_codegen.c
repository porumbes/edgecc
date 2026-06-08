#include "test_helpers.h"
#include "ast.h"
#include "semantic.h"
#include "ir.h"
#include "codegen.h"

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

TEST(test_codegen_gemm) {
    AstNode *ast = parse(
        "dim M = 2; dim K = 3; dim N = 2;\n"
        "tensor Z[R=M, C=N] : float, empty = 0.0;\n"
        "tensor A[R=K, C=M] : float, empty = 0.0;\n"
        "tensor B[R=K, C=N] : float, empty = 0.0;\n"
        "init A = [[1, 2], [3, 4], [5, 6]];\n"
        "init B = [[7, 8], [9, 10], [11, 12]];\n"
        "Z[m,n] = A[k,m] . B[k,n] :: map(mul) reduce(add);\n"
    );
    SemanticResult sem = semantic_analyze(ast);
    IRProgram *ir = ir_lower(ast, sem.symtab);
    char *code = codegen_python(ir);
    ASSERT_NOT_NULL(code);
    ASSERT_TRUE(strstr(code, "import math") != NULL);
    ASSERT_TRUE(strstr(code, "M = 2") != NULL);
    ASSERT_TRUE(strstr(code, "__edgecc_meta__") != NULL);
    ASSERT_TRUE(strstr(code, "\"inputs\": [\"A\", \"B\"]") != NULL);
    ASSERT_TRUE(strstr(code, "\"output\": \"Z\"") != NULL);
    ASSERT_TRUE(strstr(code, "def run(A, B):") != NULL);
    ASSERT_TRUE(strstr(code, "for m in range(") != NULL);
    ASSERT_TRUE(strstr(code, "if True:") != NULL);                 /* passthrough merge */
    ASSERT_TRUE(strstr(code, "a_exists = a_val") != NULL);
    ASSERT_TRUE(strstr(code, "map_val = a_op * b_op") != NULL);
    ASSERT_TRUE(strstr(code, "return Z") != NULL);
    ASSERT_TRUE(strstr(code, "if __name__ == \"__main__\":") != NULL);
    ASSERT_TRUE(strstr(code, "A = [[1, 2], [3, 4], [5, 6]]") != NULL);
    ASSERT_TRUE(strstr(code, "Z = run(A, B)") != NULL);
    ASSERT_TRUE(strstr(code, "print(Z)") != NULL);
    free(code);
    ir_free(ir);
    semantic_result_free(&sem);
    ast_free(ast);
}

TEST(test_codegen_intersect_union) {
    AstNode *ast = parse(
        "dim S = 2; dim D = 2;\n"
        "tensor F1[R=D] : float, empty = inf;\n"
        "tensor G[R=S, C=D] : float, empty = 0.0;\n"
        "tensor F[R=S]  : float, empty = inf;\n"
        "init G = [[0,1],[1,0]];\n"
        "init F = [0, inf];\n"
        "F1[d] = G[s,d] . F[s] :: map(add, intersect) reduce(min, union);\n"
    );
    SemanticResult sem = semantic_analyze(ast);
    IRProgram *ir = ir_lower(ast, sem.symtab);
    char *code = codegen_python(ir);
    ASSERT_NOT_NULL(code);
    ASSERT_TRUE(strstr(code, "if a_exists and b_exists:") != NULL);     /* intersect */
    ASSERT_TRUE(strstr(code, "if b_s or b_m:") != NULL);                /* union */
    ASSERT_TRUE(strstr(code, "math.inf") != NULL);
    ASSERT_TRUE(strstr(code, "map_val = a_op + b_op") != NULL);
    free(code);
    ir_free(ir);
    semantic_result_free(&sem);
    ast_free(ast);
}

TEST(test_codegen_generational) {
    AstNode *ast = parse(
        "dim N = 4;\n"
        "tensor G[R=N, C=N]   : float, empty = 0.0;\n"
        "tensor F[gen I, R=N] : float, empty = 0.0;\n"
        "init G = [[1.0,1.0,0.0,0.0],[0.0,1.0,1.0,0.0],[0.0,0.0,1.0,0.0],[0.0,0.0,0.0,1.0]];\n"
        "init F = [1.0, 0.0, 0.0, 0.0];\n"
        "F[i+1, d] = G[s,d] . F[i, s] :: map(mul, intersect) reduce(max, union);\n"
        "stop: stable(F);\n"
    );
    SemanticResult sem = semantic_analyze(ast);
    ASSERT_INT_EQ(0, sem.error_count);
    IRProgram *ir = ir_lower(ast, sem.symtab);
    char *code = codegen_python(ir);
    ASSERT_NOT_NULL(code);
    ASSERT_TRUE(strstr(code, "__EDGECC_MAX_GEN__ = 1000") != NULL);
    ASSERT_TRUE(strstr(code, "def run(G, F):") != NULL);
    ASSERT_TRUE(strstr(code, "F_prev = F") != NULL);
    ASSERT_TRUE(strstr(code, "while True:") != NULL);
    ASSERT_TRUE(strstr(code, "F_next = [0.0] * N") != NULL);
    ASSERT_TRUE(strstr(code, "b_val = F_prev[s]") != NULL);
    ASSERT_TRUE(strstr(code, "state = F_next[d]") != NULL);
    ASSERT_TRUE(strstr(code, "if F_next == F_prev:") != NULL);
    ASSERT_TRUE(strstr(code, "break") != NULL);
    ASSERT_TRUE(strstr(code, "F_prev = F_next") != NULL);
    ASSERT_TRUE(strstr(code, "raise RuntimeError") != NULL);
    ASSERT_TRUE(strstr(code, "return F_prev") != NULL);
    free(code);
    ir_free(ir);
    semantic_result_free(&sem);
    ast_free(ast);
}

TEST(test_codegen_cascade_bellman_ford) {
    AstNode *ast = parse(
        "dim N = 4;\n"
        "tensor W[R=N, C=N]   : float, empty = inf;\n"
        "tensor T[gen I, R=N] : float, empty = inf;\n"
        "tensor D[gen I, R=N] : float, empty = inf;\n"
        "init W = [[inf,1.0,4.0,inf],[inf,inf,2.0,6.0],[inf,inf,inf,3.0],[inf,inf,inf,inf]];\n"
        "init D = [0.0, inf, inf, inf];\n"
        "T[i, d]   = W[s,d] . D[i, s] :: map(add, intersect) reduce(min, union);\n"
        "D[i+1, d] = D[i, d] . T[i, d] :: map(min, union) reduce(min, union);\n"
        "stop: stable(D);\n"
    );
    SemanticResult sem = semantic_analyze(ast);
    ASSERT_INT_EQ(0, sem.error_count);
    IRProgram *ir = ir_lower(ast, sem.symtab);
    char *code = codegen_python(ir);
    ASSERT_NOT_NULL(code);
    ASSERT_TRUE(strstr(code, "def run(W, D):") != NULL);
    ASSERT_TRUE(strstr(code, "\"output\": \"D\"") != NULL);
    ASSERT_TRUE(strstr(code, "D_prev = D") != NULL);
    ASSERT_TRUE(strstr(code, "while True:") != NULL);
    ASSERT_TRUE(strstr(code, "D_next = [math.inf] * N") != NULL);
    ASSERT_TRUE(strstr(code, "T = [math.inf] * N") != NULL);
    ASSERT_TRUE(strstr(code, "b_val = T[d]") != NULL);
    ASSERT_TRUE(strstr(code, "a_val = D_prev[d]") != NULL);
    ASSERT_TRUE(strstr(code, "if D_next == D_prev:") != NULL);
    ASSERT_TRUE(strstr(code, "return D_prev") != NULL);
    free(code);
    ir_free(ir);
    semantic_result_free(&sem);
    ast_free(ast);
}

TEST(test_codegen_two_cross_gen) {
    AstNode *ast = parse(
        "dim N = 2;\n"
        "tensor G[R=N, C=N]   : float, empty = 0.0;\n"
        "tensor F[gen I, R=N] : float, empty = 0.0;\n"
        "tensor P[gen I, R=N] : float, empty = 0.0;\n"
        "init F = [1.0,0.0];\n"
        "init P = [1.0,0.0];\n"
        "F[i+1, d] = G[s,d] . F[i,s]   :: map(and, intersect) reduce(or, union);\n"
        "P[i+1, d] = P[i,d] . F[i+1,d] :: map(or, union) reduce(or, union);\n"
        "stop: stable(P);\n"
    );
    SemanticResult sem = semantic_analyze(ast);
    IRProgram *ir = ir_lower(ast, sem.symtab);
    char *code = codegen_python(ir);
    ASSERT_NOT_NULL(code);

    ASSERT_TRUE(strstr(code, "def run(G, F, P):") != NULL);
    ASSERT_TRUE(strstr(code, "F_prev = F") != NULL);
    ASSERT_TRUE(strstr(code, "P_prev = P") != NULL);
    ASSERT_TRUE(strstr(code, "F_next = ") != NULL);
    ASSERT_TRUE(strstr(code, "P_next = ") != NULL);
    ASSERT_TRUE(strstr(code, "b_val = F_next[d]") != NULL);  /* P einsum reads F at i+1 (next) */
    ASSERT_TRUE(strstr(code, "_done = P_next == P_prev") != NULL);
    ASSERT_TRUE(strstr(code, "F_prev = F_next") != NULL);    /* both cross-gen buffers swap */
    ASSERT_TRUE(strstr(code, "P_prev = P_next") != NULL);
    ASSERT_TRUE(strstr(code, "return P_prev") != NULL);

    free(code);
    ir_free(ir);
    semantic_result_free(&sem);
    ast_free(ast);
}

/* Two INDEPENDENT cross-gen tensors (no next-reads-next): both are double-
   buffered and advanced each generation, but the loop terminates on the single
   watched tensor (A). Guards the "advance all, gate watched" epilogue path
   independently of the coupled BFS shape. */
TEST(test_codegen_independent_cross_gen) {
    AstNode *ast = parse(
        "dim N = 2;\n"
        "tensor G[R=N, C=N]   : float, empty = 0.0;\n"
        "tensor A[gen I, R=N] : float, empty = 0.0;\n"
        "tensor B[gen I, R=N] : float, empty = 0.0;\n"
        "init A = [1.0,0.0];\n"
        "init B = [0.0,1.0];\n"
        "A[i+1, d] = G[s,d] . A[i,s] :: map(and, intersect) reduce(or, union);\n"
        "B[i+1, d] = G[s,d] . B[i,s] :: map(and, intersect) reduce(or, union);\n"
        "stop: stable(A);\n"
    );
    SemanticResult sem = semantic_analyze(ast);
    ASSERT_INT_EQ(0, sem.error_count);
    IRProgram *ir = ir_lower(ast, sem.symtab);
    char *code = codegen_python(ir);
    ASSERT_NOT_NULL(code);

    ASSERT_TRUE(strstr(code, "def run(G, A, B):") != NULL);
    ASSERT_TRUE(strstr(code, "A_next = ") != NULL);
    ASSERT_TRUE(strstr(code, "B_next = ") != NULL);
    ASSERT_TRUE(strstr(code, "_done = A_next == A_prev") != NULL);  /* gate on watched A */
    ASSERT_TRUE(strstr(code, "A_prev = A_next") != NULL);           /* advance both buffers */
    ASSERT_TRUE(strstr(code, "B_prev = B_next") != NULL);
    ASSERT_TRUE(strstr(code, "_done = B_next") == NULL);            /* B is NOT the gate */
    ASSERT_TRUE(strstr(code, "return A_prev") != NULL);

    free(code);
    ir_free(ir);
    semantic_result_free(&sem);
    ast_free(ast);
}

TEST(test_codegen_take_left_map) {
    AstNode *ast = parse(
        "dim N = 3;\n"
        "tensor Z[R=N] : float, empty = inf;\n"
        "tensor A[R=N] : float, empty = inf;\n"
        "tensor B[R=N] : float, empty = inf;\n"
        "init A = [0.0, 2.0, inf];\n"
        "init B = [inf, 5.0, inf];\n"
        "Z[n] = A[n] . B[n] :: map(take_left, take_left_only) reduce(min);\n"
    );
    SemanticResult sem = semantic_analyze(ast);
    ASSERT_INT_EQ(0, sem.error_count);
    IRProgram *ir = ir_lower(ast, sem.symtab);
    char *code = codegen_python(ir);
    ASSERT_NOT_NULL(code);
    ASSERT_TRUE(strstr(code, "map_val = a_op") != NULL);   /* left operand verbatim */
    free(code);
    ir_free(ir);
    semantic_result_free(&sem);
    ast_free(ast);
}

TEST(test_codegen_emits_load_helper) {
    AstNode *ast = parse(
        "dim N = *;\n"
        "tensor G[R=N, C=N] : float, empty = 0.0;\n"
        "tensor F[R=N] : float, empty = 0.0;\n"
        "tensor Z[R=N] : float, empty = 0.0;\n"
        "init G = load(\"g.mtx\");\n"
        "init F = [1.0, 0.0, 0.0, 0.0];\n"
        "Z[d] = G[s,d] . F[s] :: map(mul, intersect) reduce(add, union);\n"
    );
    SemanticResult sem = semantic_analyze(ast);
    ASSERT_INT_EQ(0, sem.error_count);
    IRProgram *ir = ir_lower(ast, sem.symtab);
    char *code = codegen_python(ir);
    ASSERT_NOT_NULL(code);
    ASSERT_TRUE(strstr(code, "def _edgecc_load(") != NULL);
    ASSERT_TRUE(strstr(code, "_edgecc_load(\"g.mtx\")") != NULL);
    free(code);
    ir_free(ir);
    semantic_result_free(&sem);
    ast_free(ast);
}

TEST(test_codegen_inferred_dim_from_input_shape) {
    AstNode *ast = parse(
        "dim N = *;\n"
        "tensor G[R=N, C=N] : float, empty = 0.0;\n"
        "tensor F[R=N] : float, empty = 0.0;\n"
        "tensor Z[R=N] : float, empty = 0.0;\n"
        "init G = load(\"g.mtx\");\n"
        "init F = [1.0, 0.0, 0.0, 0.0];\n"
        "Z[d] = G[s,d] . F[s] :: map(mul, intersect) reduce(add, union);\n"
    );
    SemanticResult sem = semantic_analyze(ast);
    ASSERT_INT_EQ(0, sem.error_count);
    IRProgram *ir = ir_lower(ast, sem.symtab);
    char *code = codegen_python(ir);
    ASSERT_NOT_NULL(code);
    ASSERT_TRUE(strstr(code, "N = len(") != NULL);
    free(code);
    ir_free(ir);
    semantic_result_free(&sem);
    ast_free(ast);
}

TEST(test_codegen_emits_save) {
    AstNode *ast = parse(
        "dim N = 2;\n"
        "tensor F[R=N] : float, empty = 0.0;\n"
        "tensor Z[R=N] : float, empty = 0.0;\n"
        "init F = [1.0, 0.0];\n"
        "Z[d] = F[d] . F[d] :: map(add, union) reduce(add, union);\n"
        "save: Z -> \"out.mtx\";\n"
    );
    SemanticResult sem = semantic_analyze(ast);
    ASSERT_INT_EQ(0, sem.error_count);
    IRProgram *ir = ir_lower(ast, sem.symtab);
    char *code = codegen_python(ir);
    ASSERT_NOT_NULL(code);
    ASSERT_TRUE(strstr(code, "def _edgecc_save(") != NULL);
    ASSERT_TRUE(strstr(code, "_edgecc_save(Z, \"out.mtx\")") != NULL);
    free(code);
    ir_free(ir);
    semantic_result_free(&sem);
    ast_free(ast);
}

TEST(test_codegen_shared_dim_assert) {
    AstNode *ast = parse(
        "dim N = *;\n"
        "tensor G[R=N, C=N] : float, empty = 0.0;\n"
        "tensor F[R=N]      : float, empty = 0.0;\n"
        "tensor Z[R=N]      : float, empty = 0.0;\n"
        "init G = load(\"g.mtx\");\n"
        "init F = load(\"f.mtx\");\n"
        "Z[d] = G[s,d] . F[s] :: map(mul, intersect) reduce(add, union);\n"
    );
    SemanticResult sem = semantic_analyze(ast);
    ASSERT_INT_EQ(0, sem.error_count);
    IRProgram *ir = ir_lower(ast, sem.symtab);
    char *code = codegen_python(ir);
    ASSERT_NOT_NULL(code);
    ASSERT_TRUE(strstr(code, "raise ValueError(\"edgecc: input 'F'") != NULL);
    free(code);
    ir_free(ir);
    semantic_result_free(&sem);
    ast_free(ast);
}

int main(void) {
    RUN_TEST(test_codegen_gemm);
    RUN_TEST(test_codegen_intersect_union);
    RUN_TEST(test_codegen_generational);
    RUN_TEST(test_codegen_cascade_bellman_ford);
    RUN_TEST(test_codegen_two_cross_gen);
    RUN_TEST(test_codegen_independent_cross_gen);
    RUN_TEST(test_codegen_take_left_map);
    RUN_TEST(test_codegen_emits_load_helper);
    RUN_TEST(test_codegen_inferred_dim_from_input_shape);
    RUN_TEST(test_codegen_emits_save);
    RUN_TEST(test_codegen_shared_dim_assert);
    return test_report();
}
