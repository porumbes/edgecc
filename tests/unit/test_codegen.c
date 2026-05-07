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

int main(void) {
    RUN_TEST(test_codegen_gemm);
    RUN_TEST(test_codegen_intersect_union);
    return test_report();
}
