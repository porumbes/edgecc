#ifndef SEMANTIC_H
#define SEMANTIC_H

#include "ast.h"
#include <stdarg.h>

#define MAX_DIMS 64
#define MAX_TENSORS 64
#define MAX_INDICES 16

typedef struct {
    char *name;
    int size;
} DimEntry;

typedef struct {
    char *name;
    RankPair *ranks;
    int num_ranks;
    DataType datatype;
    double empty_val;
    int dim_sizes[8];
} TensorEntry;

typedef struct {
    char *name;
    int size;
    int is_free;
} IndexEntry;

typedef struct {
    DimEntry dims[MAX_DIMS];
    int num_dims;
    TensorEntry tensors[MAX_TENSORS];
    int num_tensors;
    IndexEntry free_indices[MAX_INDICES];
    int num_free_indices;
    IndexEntry reduction_indices[MAX_INDICES];
    int num_reduction_indices;
} SymbolTable;

typedef struct {
    int error_count;
    char **errors;
    int error_capacity;
    SymbolTable *symtab;
} SemanticResult;

SemanticResult semantic_analyze(AstNode *program);
void semantic_result_free(SemanticResult *res);

double compute_op_identity(OperatorType op, DataType dt);

#endif /* SEMANTIC_H */
