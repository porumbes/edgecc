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

    /* Check key fragments are present */
    ASSERT_TRUE(strstr(code, "M = 2") != NULL);
    ASSERT_TRUE(strstr(code, "K = 3") != NULL);
    ASSERT_TRUE(strstr(code, "N = 2") != NULL);
    ASSERT_TRUE(strstr(code, "for m in range(") != NULL);
    ASSERT_TRUE(strstr(code, "for n in range(") != NULL);
    ASSERT_TRUE(strstr(code, "for k in range(") != NULL);
    ASSERT_TRUE(strstr(code, "A = [[1, 2], [3, 4], [5, 6]]") != NULL);
    ASSERT_TRUE(strstr(code, "print(Z)") != NULL);

    free(code);
    ir_free(ir);
    semantic_result_free(&sem);
    ast_free(ast);
}

int main(void) {
    RUN_TEST(test_codegen_gemm);
    return test_report();
}
