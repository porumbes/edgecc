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
            default: break;
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

TEST(test_ir_load_carries_negate) {
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
    ASSERT_INT_EQ(0, sem.error_count);
    IRProgram *ir = ir_lower(ast, sem.symtab);
    ASSERT_NOT_NULL(ir);
    int saw_neg = 0, saw_plain = 0;
    for (int i = 0; i < ir->count; i++) {
        if (ir->instrs[i].type != IR_LOAD) continue;
        if (ir->instrs[i].data.load.negate) saw_neg = 1;
        else saw_plain = 1;
    }
    ASSERT_INT_EQ(1, saw_neg);
    ASSERT_INT_EQ(1, saw_plain);
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

TEST(test_ir_generational_structure) {
    AstNode *ast = parse(
        "dim N = 4;\n"
        "tensor G[R=N, C=N]   : float, empty = 0.0;\n"
        "tensor F[gen I, R=N] : float, empty = 0.0;\n"
        "init F = [1.0, 0.0, 0.0, 0.0];\n"
        "F[i+1, d] = G[s,d] . F[i, s] :: map(mul, intersect) reduce(max, union);\n"
        "stop: stable(F);\n"
    );
    SemanticResult sem = semantic_analyze(ast);
    ASSERT_INT_EQ(0, sem.error_count);
    IRProgram *ir = ir_lower(ast, sem.symtab);
    ASSERT_NOT_NULL(ir);

    int gen_begin = 0, gen_end = 0, gen_reads = 0, gen_writes = 0;
    int loop_begins = 0;
    for (int i = 0; i < ir->count; i++) {
        switch (ir->instrs[i].type) {
            case IR_GEN_LOOP_BEGIN:
                gen_begin++;
                ASSERT_STR_EQ("F", ir->instrs[i].data.gen_loop.cross_gens[0].name);
                ASSERT_INT_EQ(1, ir->instrs[i].data.gen_loop.cross_gens[0].spatial_ndims);
                ASSERT_INT_EQ(4, ir->instrs[i].data.gen_loop.cross_gens[0].spatial_dims[0]);
                break;
            case IR_GEN_LOOP_END: gen_end++; break;
            case IR_LOOP_BEGIN:   loop_begins++; break;
            case IR_LOAD:
                if (ir->instrs[i].data.load.gen_read) {
                    gen_reads++;
                    ASSERT_INT_EQ(1, ir->instrs[i].data.load.ncoords);
                    ASSERT_STR_EQ("s", ir->instrs[i].data.load.coords[0]);
                }
                break;
            case IR_STORE:
                if (ir->instrs[i].data.store.gen_write) {
                    gen_writes++;
                    ASSERT_INT_EQ(1, ir->instrs[i].data.store.ncoords);
                    ASSERT_STR_EQ("d", ir->instrs[i].data.store.coords[0]);
                }
                break;
            default: break;
        }
    }
    ASSERT_INT_EQ(1, gen_begin);
    ASSERT_INT_EQ(1, gen_end);
    ASSERT_INT_EQ(1, gen_reads);
    ASSERT_INT_EQ(1, gen_writes);
    ASSERT_INT_EQ(2, loop_begins);   /* spatial: d (free) + s (reduction) */

    ir_free(ir);
    semantic_result_free(&sem);
    ast_free(ast);
}

TEST(test_ir_cascade_structure) {
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
    SemanticResult sem = semantic_analyze(ast);
    ASSERT_INT_EQ(0, sem.error_count);
    IRProgram *ir = ir_lower(ast, sem.symtab);
    ASSERT_NOT_NULL(ir);

    int gen_begin = 0, gen_end = 0, stores = 0, loop_begins = 0;
    int cross_writes = 0;
    for (int i = 0; i < ir->count; i++) {
        switch (ir->instrs[i].type) {
            case IR_GEN_LOOP_BEGIN:
                gen_begin++;
                ASSERT_STR_EQ("D", ir->instrs[i].data.gen_loop.cross_gens[0].name);
                ASSERT_INT_EQ(1, ir->instrs[i].data.gen_loop.cross_gens[0].spatial_ndims);
                ASSERT_INT_EQ(4, ir->instrs[i].data.gen_loop.cross_gens[0].spatial_dims[0]);
                ASSERT_INT_EQ(1, ir->instrs[i].data.gen_loop.num_intermediates);
                ASSERT_STR_EQ("T", ir->instrs[i].data.gen_loop.intermediates[0].name);
                break;
            case IR_GEN_LOOP_END:  gen_end++; break;
            case IR_LOOP_BEGIN:    loop_begins++; break;
            case IR_STORE:
                stores++;
                if (ir->instrs[i].data.store.gen_write) cross_writes++;
                break;
            default: break;
        }
    }
    ASSERT_INT_EQ(1, gen_begin);
    ASSERT_INT_EQ(1, gen_end);
    ASSERT_INT_EQ(2, stores);
    ASSERT_INT_EQ(1, cross_writes);
    ASSERT_INT_EQ(3, loop_begins);

    ir_free(ir);
    semantic_result_free(&sem);
    ast_free(ast);
}

TEST(test_ir_two_cross_gen) {
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
    ASSERT_INT_EQ(0, sem.error_count);
    IRProgram *ir = ir_lower(ast, sem.symtab);
    ASSERT_NOT_NULL(ir);

    int found_begin = 0, f_next_read = 0, p_prev_read = 0, stores = 0;
    for (int i = 0; i < ir->count; i++) {
        IRInstr *in = &ir->instrs[i];
        if (in->type == IR_GEN_LOOP_BEGIN) {
            found_begin = 1;
            ASSERT_INT_EQ(2, in->data.gen_loop.num_cross_gens);
            ASSERT_NOT_NULL(in->data.gen_loop.watch);
            ASSERT_STR_EQ("P", in->data.gen_loop.watch);
        }
        if (in->type == IR_LOAD) {
            if (strcmp(in->data.load.tensor_name, "F") == 0 && in->data.load.gen_read == 2)
                f_next_read = 1;
            if (strcmp(in->data.load.tensor_name, "P") == 0 && in->data.load.gen_read == 1)
                p_prev_read = 1;
        }
        if (in->type == IR_STORE && in->data.store.gen_write == 1) stores++;
    }
    ASSERT_INT_EQ(1, found_begin);
    ASSERT_INT_EQ(1, f_next_read);
    ASSERT_INT_EQ(1, p_prev_read);
    ASSERT_INT_EQ(2, stores);

    ir_free(ir);
    semantic_result_free(&sem);
    ast_free(ast);
}

static IRProgram *lower_source(const char *src) {
    AstNode *ast = parse(src);
    SemanticResult sem = semantic_analyze(ast);
    if (sem.error_count > 0) return NULL;
    IRProgram *ir = ir_lower(ast, sem.symtab);
    semantic_result_free(&sem);
    ast_free(ast);
    return ir;
}

TEST(test_ir_load_init_carries_path) {
    IRProgram *ir = lower_source(
        "dim N = *;\n"
        "tensor G[R=N, C=N] : float, empty = 0.0;\n"
        "tensor F[R=N] : float, empty = 0.0;\n"
        "tensor Z[R=N] : float, empty = 0.0;\n"
        "init G = load(\"g.mtx\");\n"
        "init F = [1.0, 0.0, 0.0, 0.0];\n"
        "Z[d] = G[s,d] . F[s] :: map(mul, intersect) reduce(add, union);\n");
    ASSERT_NOT_NULL(ir);
    int found = 0;
    for (int i = 0; i < ir->count; i++) {
        if (ir->instrs[i].type == IR_TENSOR_INIT &&
            strcmp(ir->instrs[i].data.tensor_init.name, "G") == 0) {
            ASSERT_INT_EQ(1, ir->instrs[i].data.tensor_init.is_load);
            ASSERT_STR_EQ("g.mtx", ir->instrs[i].data.tensor_init.path);
            found = 1;
        }
    }
    ASSERT_TRUE(found);
    ir_free(ir);
}

TEST(test_ir_inferred_dim_flag) {
    IRProgram *ir = lower_source(
        "dim N = *;\n"
        "tensor G[R=N, C=N] : float, empty = 0.0;\n"
        "tensor F[R=N] : float, empty = 0.0;\n"
        "tensor Z[R=N] : float, empty = 0.0;\n"
        "init G = load(\"g.mtx\");\n"
        "init F = [1.0, 0.0, 0.0, 0.0];\n"
        "Z[d] = G[s,d] . F[s] :: map(mul, intersect) reduce(add, union);\n");
    ASSERT_NOT_NULL(ir);
    int seen = 0;
    for (int i = 0; i < ir->count; i++)
        if (ir->instrs[i].type == IR_DIM_DEF &&
            strcmp(ir->instrs[i].data.dim_def.name, "N") == 0) {
            ASSERT_INT_EQ(1, ir->instrs[i].data.dim_def.is_inferred);
            seen = 1;
        }
    ASSERT_TRUE(seen);
    ir_free(ir);
}

TEST(test_ir_save_instruction) {
    IRProgram *ir = lower_source(
        "dim N = 2;\n"
        "tensor F[R=N] : float, empty = 0.0;\n"
        "tensor Z[R=N] : float, empty = 0.0;\n"
        "init F = [1.0, 0.0];\n"
        "Z[d] = F[d] . F[d] :: map(add, union) reduce(add, union);\n"
        "save: Z -> \"out.mtx\";\n");
    ASSERT_NOT_NULL(ir);
    int found = 0;
    for (int i = 0; i < ir->count; i++)
        if (ir->instrs[i].type == IR_SAVE) {
            ASSERT_STR_EQ("Z", ir->instrs[i].data.save.tensor_name);
            ASSERT_STR_EQ("out.mtx", ir->instrs[i].data.save.path);
            found = 1;
        }
    ASSERT_TRUE(found);
    ir_free(ir);
}

int main(void) {
    RUN_TEST(test_ir_gemm_structure);
    RUN_TEST(test_ir_load_carries_negate);
    RUN_TEST(test_ir_loop_order);
    RUN_TEST(test_ir_carries_merge_and_identity);
    RUN_TEST(test_ir_default_merge_is_passthrough);
    RUN_TEST(test_ir_generational_structure);
    RUN_TEST(test_ir_cascade_structure);
    RUN_TEST(test_ir_two_cross_gen);
    RUN_TEST(test_ir_load_init_carries_path);
    RUN_TEST(test_ir_inferred_dim_flag);
    RUN_TEST(test_ir_save_instruction);
    return test_report();
}
