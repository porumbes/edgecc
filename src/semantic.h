#ifndef SEMANTIC_H
#define SEMANTIC_H

#include "ast.h"
#include <stdarg.h>

#define MAX_DIMS 64
#define MAX_TENSORS 64
#define MAX_INDICES 16
#define MAX_EINSUMS 16

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
    int gen_axis;   /* index of generational rank, or -1 */
    int has_init;   /* 1 if an `init` statement provides this tensor's values */
} TensorEntry;

typedef struct {
    char *name;
    int size;
    int is_free;
} IndexEntry;

typedef struct {
    IndexEntry free_indices[MAX_INDICES];
    int num_free_indices;
    IndexEntry reduction_indices[MAX_INDICES];
    int num_reduction_indices;
} EinsumClass;

typedef struct {
    DimEntry dims[MAX_DIMS];
    int num_dims;
    TensorEntry tensors[MAX_TENSORS];
    int num_tensors;
    IndexEntry free_indices[MAX_INDICES];
    int num_free_indices;
    IndexEntry reduction_indices[MAX_INDICES];
    int num_reduction_indices;
    /* Generational (iterative) program metadata. is_iterative == 0 for
       ordinary single-pass programs. */
    int is_iterative;
    /* Cross-generational tensors: gen-rank tensors written at offset i+1.
       C1 allows more than one (e.g. BFS frontier F and visited set P). */
    char *gen_tensors[MAX_TENSORS];   /* borrowed names, source order */
    int num_gen_tensors;
    char *gen_tensor;    /* alias = gen_tensors[0]; kept for diagnostics (borrowed) */
    char *gen_rank_var;  /* the index var bound to the gen rank, e.g. "i" (borrowed) */
    char *stop_watch;    /* tensor named in `stop: stable(...)` (borrowed) */
    /* Per-Einsum index classification, in source order (cascade support). */
    EinsumClass einsum_class[MAX_EINSUMS];
    int num_einsums;
    /* Intra-generation intermediate tensors (gen-rank, written at offset i). */
    char *intermediates[MAX_TENSORS];   /* borrowed names */
    int num_intermediates;
} SymbolTable;

typedef struct {
    int error_count;
    char **errors;
    int error_capacity;
    SymbolTable *symtab;
} SemanticResult;

SemanticResult semantic_analyze(AstNode *program);
void semantic_result_free(SemanticResult *res);

TensorEntry *symtab_find_tensor(SymbolTable *st, const char *name);
double compute_op_identity(OperatorType op, DataType dt);

#endif /* SEMANTIC_H */
