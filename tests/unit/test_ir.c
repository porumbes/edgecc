#include "test_helpers.h"
#include "ast.h"
#include "semantic.h"
#include "ir.h"

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

TEST(test_ir_gemm_structure) {
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
    ASSERT_INT_EQ(0, sem.error_count);

    IRProgram *ir = ir_lower(ast, sem.symtab);
    ASSERT_NOT_NULL(ir);
    ASSERT_TRUE(ir->count > 0);

    int dim_defs = 0, allocs = 0, inits = 0;
    int loop_begins = 0, loop_ends = 0;
    int loads = 0, maps = 0, reduces = 0, stores = 0;
    for (int i = 0; i < ir->count; i++) {
        switch (ir->instrs[i].type) {
            case IR_DIM_DEF:       dim_defs++; break;
            case IR_TENSOR_ALLOC:  allocs++; break;
            case IR_TENSOR_INIT:   inits++; break;
            case IR_LOOP_BEGIN:    loop_begins++; break;
            case IR_LOOP_END:      loop_ends++; break;
            case IR_LOAD:          loads++; break;
            case IR_MAP:           maps++; break;
            case IR_REDUCE:        reduces++; break;
            case IR_STORE:         stores++; break;
        }
    }
    ASSERT_INT_EQ(3, dim_defs);
    ASSERT_INT_EQ(3, allocs);
    ASSERT_INT_EQ(2, inits);
    ASSERT_INT_EQ(3, loop_begins);
    ASSERT_INT_EQ(3, loop_ends);
    ASSERT_INT_EQ(2, loads);
    ASSERT_INT_EQ(1, maps);
    ASSERT_INT_EQ(1, reduces);
    ASSERT_INT_EQ(1, stores);

    ir_free(ir);
    semantic_result_free(&sem);
    ast_free(ast);
}

TEST(test_ir_loop_order) {
    AstNode *ast = parse(
        "dim M = 2; dim K = 3; dim N = 2;\n"
        "tensor Z[R=M, C=N] : float, empty = 0.0;\n"
        "tensor A[R=K, C=M] : float, empty = 0.0;\n"
        "tensor B[R=K, C=N] : float, empty = 0.0;\n"
        "Z[m,n] = A[k,m] . B[k,n] :: map(mul) reduce(add);\n"
    );
    SemanticResult sem = semantic_analyze(ast);
    IRProgram *ir = ir_lower(ast, sem.symtab);

    int loop_count = 0;
    char *loop_vars[3];
    for (int i = 0; i < ir->count && loop_count < 3; i++) {
        if (ir->instrs[i].type == IR_LOOP_BEGIN) {
            loop_vars[loop_count++] = ir->instrs[i].data.loop_begin.index_var;
        }
    }
    ASSERT_INT_EQ(3, loop_count);
    ASSERT_STR_EQ("m", loop_vars[0]);
    ASSERT_STR_EQ("n", loop_vars[1]);
    ASSERT_STR_EQ("k", loop_vars[2]);

    ir_free(ir);
    semantic_result_free(&sem);
    ast_free(ast);
}

TEST(test_ir_carries_merge_and_identity) {
    AstNode *ast = parse(
        "dim S = 3; dim D = 3;\n"
        "tensor F1[R=D] : float, empty = inf;\n"
        "tensor G[R=S, C=D] : float, empty = 0.0;\n"
        "tensor F[R=S]  : float, empty = inf;\n"
        "F1[d] = G[s,d] . F[s] :: map(add, intersect) reduce(min, union);\n"
    );
    SemanticResult sem = semantic_analyze(ast);
    IRProgram *ir = ir_lower(ast, sem.symtab);
    int found_map = 0, found_red = 0;
    for (int i = 0; i < ir->count; i++) {
        if (ir->instrs[i].type == IR_MAP) {
            ASSERT_INT_EQ(MERGE_INTERSECT, ir->instrs[i].data.map.merge_op);
            ASSERT_TRUE(ir->instrs[i].data.map.empty_a == 0.0);
            ASSERT_TRUE(ir->instrs[i].data.map.empty_b > 1e308);   /* inf */
            ASSERT_TRUE(ir->instrs[i].data.map.empty_z > 1e308);   /* inf */
            found_map = 1;
        }
        if (ir->instrs[i].type == IR_REDUCE) {
            ASSERT_INT_EQ(MERGE_UNION, ir->instrs[i].data.reduce.merge_op);
            ASSERT_TRUE(ir->instrs[i].data.reduce.identity > 1e308); /* +inf for min */
            ASSERT_TRUE(ir->instrs[i].data.reduce.empty_z  > 1e308);
            found_red = 1;
        }
    }
    ASSERT_TRUE(found_map);
    ASSERT_TRUE(found_red);
    ir_free(ir);
    semantic_result_free(&sem);
    ast_free(ast);
}

TEST(test_ir_default_merge_is_passthrough) {
    AstNode *ast = parse(
        "dim M = 2; dim K = 2; dim N = 2;\n"
        "tensor Z[R=M, C=N] : float, empty = 0.0;\n"
        "tensor A[R=K, C=M] : float, empty = 0.0;\n"
        "tensor B[R=K, C=N] : float, empty = 0.0;\n"
        "Z[m,n] = A[k,m] . B[k,n] :: map(mul) reduce(add);\n"
    );
    SemanticResult sem = semantic_analyze(ast);
    IRProgram *ir = ir_lower(ast, sem.symtab);
    for (int i = 0; i < ir->count; i++) {
        if (ir->instrs[i].type == IR_MAP) {
            ASSERT_INT_EQ(MERGE_PASSTHROUGH, ir->instrs[i].data.map.merge_op);
        }
        if (ir->instrs[i].type == IR_REDUCE) {
            ASSERT_INT_EQ(MERGE_PASSTHROUGH, ir->instrs[i].data.reduce.merge_op);
        }
    }
    ir_free(ir);
    semantic_result_free(&sem);
    ast_free(ast);
}

int main(void) {
    RUN_TEST(test_ir_gemm_structure);
    RUN_TEST(test_ir_loop_order);
    RUN_TEST(test_ir_carries_merge_and_identity);
    RUN_TEST(test_ir_default_merge_is_passthrough);
    return test_report();
}
