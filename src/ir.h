#ifndef IR_H
#define IR_H

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
    IR_STORE
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
} IRLoad;

typedef struct {
    int dst_reg;
    OperatorType op;
    int src_reg1;
    int src_reg2;
} IRMap;

typedef struct {
    int dst_reg;
    OperatorType op;
    int src_reg;
    int acc_reg;
} IRReduce;

typedef struct {
    char *tensor_name;
    char **coords;
    int ncoords;
    int src_reg;
} IRStore;

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
    } data;
} IRInstr;

typedef struct {
    IRInstr *instrs;
    int count;
    int capacity;
} IRProgram;

IRProgram *ir_lower(AstNode *program, SymbolTable *symtab);
void ir_free(IRProgram *ir);

#endif /* IR_H */
