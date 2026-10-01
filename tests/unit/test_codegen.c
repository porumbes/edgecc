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

TEST(test_codegen_negated_operand) {
    AstNode *ast = parse(
        "dim N = 3;\n"
        "tensor Z[R=N] : float, empty = 0.0;\n"
        "tensor A[R=N] : float, empty = 0.0;\n"
        "tensor B[R=N] : float, empty = 0.0;\n"
        "init A = [1.0, 0.0, 1.0];\n"
        "init B = [0.0, 1.0, 0.0];\n"
        "Z[d] = A[d] . !B[d] :: map(take_left, intersect) reduce(or, union);\n"
    );
    SemanticResult sem = semantic_analyze(ast);
    IRProgram *ir = ir_lower(ast, sem.symtab);
    char *code = codegen_python(ir);
    ASSERT_NOT_NULL(code);
    ASSERT_TRUE(strstr(code, "a_exists = a_val != 0.0") != NULL);
    ASSERT_TRUE(strstr(code, "b_exists = b_val == 0.0") != NULL);
    ASSERT_TRUE(strstr(code, "b_op = 1.0 if b_exists else 0.0") != NULL);
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

/* Multi-dim generational buffers (per-generation slices appended each
   iteration, for cross-gen tensors and intermediates alike) must be
   sized by their declared dim names, per axis -- not by matching sizes. */
TEST(test_codegen_generational_2d_alloc_uses_dim_names) {
    AstNode *ast = parse(
        "dim M = 3;\n"
        "dim N = 3;\n"
        "tensor G[R=N, C=N]        : float, empty = 0.0;\n"
        "tensor U[gen I, R=N, C=M] : float, empty = 0.0;\n"
        "tensor T[gen I, R=N, C=M] : float, empty = 0.0;\n"
        "init G = [[0.0,1.0,0.0],[0.0,0.0,1.0],[0.0,0.0,0.0]];\n"
        "init T = [[0.0,1.0,0.0],[0.0,0.0,1.0],[0.0,0.0,0.0]];\n"
        "U[i, a, c]   = G[a,b] . T[i,b,c] :: map(and,intersect) reduce(or,union);\n"
        "T[i+1, a, c] = U[i,a,c] . T[i,a,c] :: map(or,union) reduce(or,union);\n"
        "stop: stable(T);\n"
    );
    SemanticResult sem = semantic_analyze(ast);
    ASSERT_INT_EQ(0, sem.error_count);
    IRProgram *ir = ir_lower(ast, sem.symtab);
    char *code = codegen_python(ir);
    ASSERT_NOT_NULL(code);
    ASSERT_TRUE(strstr(code, "T.append([[0.0] * M for _ in range(N)])") != NULL);
    ASSERT_TRUE(strstr(code, "U.append([[0.0] * M for _ in range(N)])") != NULL);
    free(code);
    ir_free(ir);
    semantic_result_free(&sem);
    ast_free(ast);
}

/* Same, with an inferred dim: the size is only known at runtime. */
TEST(test_codegen_generational_2d_alloc_inferred_dim) {
    AstNode *ast = parse(
        "dim N = *;\n"
        "tensor G[R=N, C=N]        : float, empty = 0.0;\n"
        "tensor T[gen I, R=N, C=N] : float, empty = 0.0;\n"
        "init G = load(\"graph.mtx\");\n"
        "init T = load(\"graph.mtx\");\n"
        "T[i+1, a, c] = G[a,b] . T[i,b,c] :: map(and,intersect) reduce(or,union);\n"
        "stop: stable(T);\n"
    );
    SemanticResult sem = semantic_analyze(ast);
    ASSERT_INT_EQ(0, sem.error_count);
    IRProgram *ir = ir_lower(ast, sem.symtab);
    char *code = codegen_python(ir);
    ASSERT_NOT_NULL(code);
    ASSERT_TRUE(strstr(code, "T.append([[0.0] * N for _ in range(N)])") != NULL);
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
    ASSERT_TRUE(strstr(code, "F = [F]") != NULL);
    ASSERT_TRUE(strstr(code, "_i = 0") != NULL);
    ASSERT_TRUE(strstr(code, "while True:") != NULL);
    ASSERT_TRUE(strstr(code, "F.append([0.0] * N)") != NULL);
    ASSERT_TRUE(strstr(code, "b_val = F[_i][s]") != NULL);
    ASSERT_TRUE(strstr(code, "state = F[_i + 1][d]") != NULL);
    ASSERT_TRUE(strstr(code, "if F[_i + 1] == F[_i]:") != NULL);
    ASSERT_TRUE(strstr(code, "_i += 1") != NULL);
    ASSERT_TRUE(strstr(code, "raise RuntimeError(\"generational loop did not stop\")") != NULL);
    ASSERT_TRUE(strstr(code, "return F\n") != NULL);
    ASSERT_TRUE(strstr(code, "_prev") == NULL);
    ASSERT_TRUE(strstr(code, "_next") == NULL);
    ASSERT_TRUE(strstr(code, "_edgecc_occupancy") == NULL);   /* only when used */
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
    ASSERT_TRUE(strstr(code, "D = [D]") != NULL);
    ASSERT_TRUE(strstr(code, "T = []") != NULL);
    ASSERT_TRUE(strstr(code, "D.append([math.inf] * N)") != NULL);
    ASSERT_TRUE(strstr(code, "T.append([math.inf] * N)") != NULL);
    ASSERT_TRUE(strstr(code, "state = T[_i][d]") != NULL);     /* T written at i */
    ASSERT_TRUE(strstr(code, "b_val = T[_i][d]") != NULL);     /* T read at i */
    ASSERT_TRUE(strstr(code, "a_val = D[_i][d]") != NULL);
    ASSERT_TRUE(strstr(code, "if D[_i + 1] == D[_i]:") != NULL);
    ASSERT_TRUE(strstr(code, "return D\n") != NULL);
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
    ASSERT_TRUE(strstr(code, "F = [F]") != NULL);
    ASSERT_TRUE(strstr(code, "P = [P]") != NULL);
    ASSERT_TRUE(strstr(code, "F.append(") != NULL);
    ASSERT_TRUE(strstr(code, "P.append(") != NULL);
    ASSERT_TRUE(strstr(code, "b_val = F[_i + 1][d]") != NULL);  /* P einsum reads F at i+1 */
    ASSERT_TRUE(strstr(code, "if P[_i + 1] == P[_i]:") != NULL);
    ASSERT_TRUE(strstr(code, "return P\n") != NULL);

    free(code);
    ir_free(ir);
    semantic_result_free(&sem);
    ast_free(ast);
}

/* Two INDEPENDENT cross-gen tensors (no next-reads-next): both are stored in
   full; the loop stops on the single watched tensor (A). */
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
    ASSERT_TRUE(strstr(code, "A.append(") != NULL);
    ASSERT_TRUE(strstr(code, "B.append(") != NULL);
    ASSERT_TRUE(strstr(code, "if A[_i + 1] == A[_i]:") != NULL);   /* gate on watched A */
    ASSERT_TRUE(strstr(code, "if B[") == NULL);                    /* B is NOT the gate */
    ASSERT_TRUE(strstr(code, "return A\n") != NULL);

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

TEST(test_codegen_default_no_trace) {
    AstNode *ast = parse(
        "dim M = 2; dim K = 3; dim N = 2;\n"
        "tensor Z[R=M, C=N] : float, empty = 0.0;\n"
        "tensor A[R=K, C=M] : float, empty = 0.0;\n"
        "tensor B[R=K, C=N] : float, empty = 0.0;\n"
        "init A = [[1, 2], [3, 4], [5, 6]];\n"
        "init B = [[7, 8], [9, 10], [11, 12]];\n"
        "Z[m,n] = A[k,m] . B[k,n] :: map(mul) reduce(add);\n");
    SemanticResult sem = semantic_analyze(ast);
    IRProgram *ir = ir_lower(ast, sem.symtab);
    char *code = codegen_python_ex(ir, 0, "gemm");
    ASSERT_TRUE(strstr(code, "_trace") == NULL);
    free(code);
    ir_free(ir); semantic_result_free(&sem); ast_free(ast);
}

TEST(test_codegen_trace_on_nongenerational_warns_no_scaffold) {
    AstNode *ast = parse(
        "dim M = 2; dim K = 3; dim N = 2;\n"
        "tensor Z[R=M, C=N] : float, empty = 0.0;\n"
        "tensor A[R=K, C=M] : float, empty = 0.0;\n"
        "tensor B[R=K, C=N] : float, empty = 0.0;\n"
        "init A = [[1, 2], [3, 4], [5, 6]];\n"
        "init B = [[7, 8], [9, 10], [11, 12]];\n"
        "Z[m,n] = A[k,m] . B[k,n] :: map(mul) reduce(add);\n");
    SemanticResult sem = semantic_analyze(ast);
    IRProgram *ir = ir_lower(ast, sem.symtab);
    char *code = codegen_python_ex(ir, 1, "gemm");
    ASSERT_TRUE(strstr(code, "_trace") == NULL);
    free(code);
    ir_free(ir); semantic_result_free(&sem); ast_free(ast);
}

TEST(test_codegen_trace_scaffolding) {
    AstNode *ast = parse(
        "dim N = 4;\n"
        "tensor G[R=N, C=N]   : float, empty = 0.0;\n"
        "tensor F[gen I, R=N] : float, empty = 0.0;\n"
        "tensor P[gen I, R=N] : float, empty = 0.0;\n"
        "init G = [[0.0,1.0,0.0,0.0],[0.0,0.0,1.0,0.0],[0.0,0.0,0.0,1.0],[0.0,0.0,0.0,0.0]];\n"
        "init F = [1.0, 0.0, 0.0, 0.0];\n"
        "init P = [1.0, 0.0, 0.0, 0.0];\n"
        "F[i+1, d] = G[s,d] . F[i,s]   :: map(and, intersect) reduce(or, union);\n"
        "P[i+1, d] = P[i,d] . F[i+1,d] :: map(or, union) reduce(or, union);\n"
        "stop: stable(P);\n");
    SemanticResult sem = semantic_analyze(ast);
    IRProgram *ir = ir_lower(ast, sem.symtab);
    char *code = codegen_python_ex(ir, 1, "reachability_bfs");
    ASSERT_TRUE(strstr(code, "_trace=None") != NULL);
    ASSERT_TRUE(strstr(code, "\"kind\": \"generation\", \"gen\": 0") != NULL);
    ASSERT_TRUE(strstr(code, "for _g in range(1, _i + 2):") != NULL);
    ASSERT_TRUE(strstr(code, "\"P\": P[_g]") != NULL);
    ASSERT_TRUE(strstr(code, "\"schema\": \"edgecc-trace/1\"") != NULL);
    ASSERT_TRUE(strstr(code, "\"dims\": {\"N\": len(P[0])}") != NULL);
    ASSERT_TRUE(strstr(code, ".trace.json") != NULL);
    ASSERT_TRUE(strstr(code, "_edgecc_snapshot") == NULL);   /* history needs no copies */
    free(code);
    ir_free(ir); semantic_result_free(&sem); ast_free(ast);
}

TEST(test_codegen_seed_init) {
    AstNode *ast = parse(
        "dim N = *;\n"
        "tensor G[R=N, C=N] : float, empty = 0.0;\n"
        "tensor D[R=N]      : float, empty = inf;\n"
        "init G = load(\"g.mtx\");\n"
        "init D = seed(source=2, value=0.0);\n"
        "D[d] = G[s,d] . D[s] :: map(add, intersect) reduce(min, union);\n"
    );
    SemanticResult sem = semantic_analyze(ast);
    IRProgram *ir = ir_lower(ast, sem.symtab);
    char *code = codegen_python(ir);
    ASSERT_NOT_NULL(code);
    ASSERT_TRUE(strstr(code, "D = [math.inf] * len(G)") != NULL);
    ASSERT_TRUE(strstr(code, "if not (0 <= 2 < len(D)):") != NULL);
    ASSERT_TRUE(strstr(code, "D[2] = 0.0") != NULL);
    free(code);
    ir_free(ir);
    semantic_result_free(&sem);
    ast_free(ast);
}

/* Generational seed tensor D[gen I, R=N]: the gen axis must be skipped when
   sizing (D = [..] * len(W), not via the gen index) and in the inferred-dim
   shape assert (len(D), not len(D[0])). */
TEST(test_codegen_seed_generational) {
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
    IRProgram *ir = ir_lower(ast, sem.symtab);
    char *code = codegen_python(ir);
    ASSERT_NOT_NULL(code);
    ASSERT_TRUE(strstr(code, "D = [math.inf] * len(W)") != NULL);
    ASSERT_TRUE(strstr(code, "if len(D) != N:") != NULL);
    ASSERT_TRUE(strstr(code, "if len(D[0]) != N:") == NULL);
    free(code);
    ir_free(ir);
    semantic_result_free(&sem);
    ast_free(ast);
}

TEST(test_codegen_fused_push) {
    AstNode *ast = parse(
        "dim N = 3;\n"
        "tensor F[R=N] : float, empty = 0.0;\n"
        "tensor G[R=N, C=N] : float, empty = 0.0;\n"
        "tensor P[R=N] : float, empty = 0.0;\n"
        "init G = [[0,0,0],[0,0,0],[0,0,0]];\n"
        "init F = [1.0,0.0,0.0];\n"
        "init P = [1.0,0.0,0.0];\n"
        "F[d] = (G[s,d] .1 F[s]) .2 !P[d] "
        ":: map.1(and,intersect) reduce.1(or,union) map.2(take_left,intersect);\n");
    SemanticResult sem = semantic_analyze(ast);
    IRProgram *ir = ir_lower(ast, sem.symtab);
    char *code = codegen_python(ir);
    ASSERT_NOT_NULL(code);
    ASSERT_TRUE(strstr(code, "_acc1 =") != NULL);
    ASSERT_TRUE(strstr(code, "_inter =") != NULL);
    ASSERT_TRUE(strstr(code, "c_exists = c_val == 0.0") != NULL);   /* !P existence flip */
    ASSERT_TRUE(strstr(code, "NotImplementedError") == NULL);       /* stub replaced */
    free(code); ir_free(ir); semantic_result_free(&sem); ast_free(ast);
}

TEST(test_codegen_fused_pull) {
    AstNode *ast = parse(
        "dim N = 3;\n"
        "tensor F[R=N] : float, empty = 0.0;\n"
        "tensor G[R=N, C=N] : float, empty = 0.0;\n"
        "tensor P[R=N] : float, empty = 0.0;\n"
        "init G = [[0,0,0],[0,0,0],[0,0,0]];\n"
        "init F = [1.0,0.0,0.0];\n"
        "init P = [1.0,0.0,0.0];\n"
        "F[d] = (G[s,d] .1 !P[d]) .2 F[s] "
        ":: map.1(and,intersect) map.2(and,intersect) reduce.2(or,union);\n");
    SemanticResult sem = semantic_analyze(ast);
    IRProgram *ir = ir_lower(ast, sem.symtab);
    char *code = codegen_python(ir);
    ASSERT_NOT_NULL(code);
    ASSERT_TRUE(strstr(code, "_acc2 =") != NULL);       /* action-2 reduction accumulator */
    ASSERT_TRUE(strstr(code, "_inter_emit") != NULL);   /* unified intermediate flag */
    free(code); ir_free(ir); semantic_result_free(&sem); ast_free(ast);
}

TEST(test_codegen_occupancy_stop) {
    AstNode *ast = parse(
        "dim N = 3;\n"
        "tensor G[R=N, C=N]   : float, empty = 0.0;\n"
        "tensor T[gen I, R=N] : float, empty = inf;\n"
        "tensor F[gen I, R=N] : float, empty = inf;\n"
        "tensor P[gen I, R=N] : float, empty = 0.0;\n"
        "init G = [[0.0,1.0,0.0],[0.0,0.0,1.0],[0.0,0.0,0.0]];\n"
        "init F = [0.0, inf, inf];\n"
        "init P = [1.0, 0.0, 0.0];\n"
        "T[i, d]   = G[s,d] . F[i,s]   :: map(add, intersect) reduce(min, union);\n"
        "F[i+1, d] = T[i,d] . !P[i,d]  :: map(take_left, intersect) reduce(min, union);\n"
        "P[i+1, d] = P[i,d] . F[i+1,d] :: map(or, union) reduce(or, union);\n"
        "stop: ||F[i+1]|| == 0;\n");
    SemanticResult sem = semantic_analyze(ast);
    ASSERT_INT_EQ(0, sem.error_count);
    IRProgram *ir = ir_lower(ast, sem.symtab);
    char *code = codegen_python(ir);
    ASSERT_NOT_NULL(code);
    ASSERT_TRUE(strstr(code, "def _edgecc_occupancy(x, empty):") != NULL);
    ASSERT_TRUE(strstr(code, "if _edgecc_occupancy(F[_i + 1], math.inf) == 0:") != NULL);
    ASSERT_TRUE(strstr(code, "\"output\": \"F\"") != NULL);
    ASSERT_TRUE(strstr(code, "return F\n") != NULL);
    free(code); ir_free(ir); semantic_result_free(&sem); ast_free(ast);
}

TEST(test_codegen_fused_generational_indexing) {
    AstNode *ast = parse(
        "dim N = 3;\n"
        "tensor G[R=N, C=N]   : float, empty = 0.0;\n"
        "tensor F[gen I, R=N] : float, empty = 0.0;\n"
        "tensor P[gen I, R=N] : float, empty = 0.0;\n"
        "init G = [[0.0,1.0,0.0],[0.0,0.0,1.0],[0.0,0.0,0.0]];\n"
        "init F = [1.0, 0.0, 0.0];\n"
        "init P = [1.0, 0.0, 0.0];\n"
        "F[i+1, d] = (G[s,d] .1 F[i,s]) .2 !P[i,d] "
        ":: map.1(and,intersect) reduce.1(or,union) map.2(take_left,intersect);\n"
        "P[i+1, d] = P[i,d] . F[i+1,d] :: map(or,union) reduce(or,union);\n"
        "stop: stable(P);\n");
    SemanticResult sem = semantic_analyze(ast);
    ASSERT_INT_EQ(0, sem.error_count);
    IRProgram *ir = ir_lower(ast, sem.symtab);
    char *code = codegen_python(ir);
    ASSERT_NOT_NULL(code);
    ASSERT_TRUE(strstr(code, "F[_i][s]") != NULL);        /* inner operand at i */
    ASSERT_TRUE(strstr(code, "P[_i][d]") != NULL);        /* outer operand at i */
    ASSERT_TRUE(strstr(code, "F[_i + 1][d]") != NULL);    /* fused output at i+1 */
    ASSERT_TRUE(strstr(code, "_prev") == NULL);
    ASSERT_TRUE(strstr(code, "_next") == NULL);
    free(code); ir_free(ir); semantic_result_free(&sem); ast_free(ast);
}

int main(void) {
    RUN_TEST(test_codegen_fused_push);
    RUN_TEST(test_codegen_fused_pull);
    RUN_TEST(test_codegen_gemm);
    RUN_TEST(test_codegen_negated_operand);
    RUN_TEST(test_codegen_intersect_union);
    RUN_TEST(test_codegen_generational);
    RUN_TEST(test_codegen_generational_2d_alloc_uses_dim_names);
    RUN_TEST(test_codegen_generational_2d_alloc_inferred_dim);
    RUN_TEST(test_codegen_cascade_bellman_ford);
    RUN_TEST(test_codegen_two_cross_gen);
    RUN_TEST(test_codegen_independent_cross_gen);
    RUN_TEST(test_codegen_take_left_map);
    RUN_TEST(test_codegen_emits_load_helper);
    RUN_TEST(test_codegen_inferred_dim_from_input_shape);
    RUN_TEST(test_codegen_emits_save);
    RUN_TEST(test_codegen_shared_dim_assert);
    RUN_TEST(test_codegen_seed_init);
    RUN_TEST(test_codegen_seed_generational);
    RUN_TEST(test_codegen_default_no_trace);
    RUN_TEST(test_codegen_trace_on_nongenerational_warns_no_scaffold);
    RUN_TEST(test_codegen_trace_scaffolding);
    RUN_TEST(test_codegen_occupancy_stop);
    RUN_TEST(test_codegen_fused_generational_indexing);
    return test_report();
}
