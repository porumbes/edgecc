#include "test_helpers.h"
#include "ast.h"
#include "semantic.h"

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

int main(void) {
    RUN_TEST(test_valid_gemm);
    RUN_TEST(test_undeclared_dim);
    RUN_TEST(test_undeclared_tensor_in_init);
    RUN_TEST(test_dimension_mismatch);
    RUN_TEST(test_init_shape_mismatch);
    return test_report();
}
