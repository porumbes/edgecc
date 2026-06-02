#ifndef IR_H
#define IR_H

#include <stdio.h>
#include "ast.h"
#include "semantic.h"

typedef enum {
    IR_DIM_DEF,
    IR_TENSOR_ALLOC,
    IR_TENSOR_INIT,
    IR_LOOP_BEGIN,
    IR_LOOP_END,
    IR_LOAD,
    IR_MAP,
    IR_REDUCE,
    IR_STORE,
    IR_GEN_LOOP_BEGIN,
    IR_GEN_LOOP_END
} IRType;

typedef struct {
    char *name;
    int size;
} IRDimDef;

typedef struct {
    char *name;
    int dim_sizes[8];
    int ndims;
    double empty_val;
} IRTensorAlloc;

typedef struct {
    char *name;
    LiteralData *data;  /* borrowed pointer, not owned */
} IRTensorInit;

typedef struct {
    char *index_var;
    int range;
} IRLoopBegin;

typedef struct {
    int reg;
    char *tensor_name;
    char **coords;
    int ncoords;
    int gen_read;   /* 1 if this loads the generational tensor's prev slice */
} IRLoad;

typedef struct {
    int dst_reg;
    OperatorType op;
    MergeOp      merge_op;
    int src_reg1;
    int src_reg2;
    double empty_a;        /* empty value of left operand */
    double empty_b;        /* empty value of right operand */
    double empty_z;        /* empty value of output (for maptmp emission check) */
} IRMap;

typedef struct {
    int dst_reg;
    OperatorType op;
    MergeOp      merge_op;
    int src_reg;
    int acc_reg;
    double identity;       /* compute-op identity for reduction state */
    double empty_z;        /* empty value of output */
} IRReduce;

typedef struct {
    char *tensor_name;
    char **coords;
    int ncoords;
    int src_reg;
    int gen_write;  /* 1 if this stores into the generational tensor's next slice */
} IRStore;

typedef struct {
    char *tensor_name;       /* the generational tensor */
    int spatial_dims[8];     /* shape excluding the generational axis */
    int spatial_ndims;
    double empty_val;        /* for the fresh slice */
    int max_gen;             /* safety iteration cap */
} IRGenLoop;

typedef struct {
    IRType type;
    union {
        IRDimDef dim_def;
        IRTensorAlloc tensor_alloc;
        IRTensorInit tensor_init;
        IRLoopBegin loop_begin;
        IRLoad load;
        IRMap map;
        IRReduce reduce;
        IRStore store;
        IRGenLoop gen_loop;
    } data;
} IRInstr;

typedef struct {
    IRInstr *instrs;
    int count;
    int capacity;
} IRProgram;

IRProgram *ir_lower(AstNode *program, SymbolTable *symtab);
void ir_dump(IRProgram *ir, FILE *out);
void ir_free(IRProgram *ir);

#endif /* IR_H */
