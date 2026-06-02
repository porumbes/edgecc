#ifndef AST_H
#define AST_H

#include <stdlib.h>

/* Forward declarations */
typedef struct AstNode AstNode;
typedef struct TensorRef TensorRef;
typedef struct RankPair RankPair;
typedef struct LiteralData LiteralData;

/* ---- Enums ---- */

typedef enum {
    NODE_PROGRAM,
    NODE_DIM_DECL,
    NODE_TENSOR_DECL,
    NODE_INIT,
    NODE_EINSUM,
    NODE_STOP
} NodeType;

typedef enum {
    TYPE_INTEGER,
    TYPE_FLOAT,
    TYPE_BOOLEAN
} DataType;

typedef enum {
    OP_ADD,
    OP_SUB,
    OP_MUL,
    OP_MIN,
    OP_MAX,
    OP_AND,
    OP_OR
} OperatorType;

typedef enum {
    MERGE_PASSTHROUGH,    /* default; also encodes "absent in source" */
    MERGE_NOPASS,
    MERGE_INTERSECT,
    MERGE_TAKE_LEFT,
    MERGE_TAKE_LEFT_ONLY,
    MERGE_TAKE_RIGHT,
    MERGE_TAKE_RIGHT_ONLY,
    MERGE_XOR,
    MERGE_UNION,
} MergeOp;

/* ---- Supporting structs ---- */

struct RankPair {
    char *rank_name;
    char *dim_name;        /* NULL when is_generational */
    int   is_generational; /* 1 if declared `gen NAME` (unbounded, no dim) */
};

struct TensorRef {
    char *tensor_name;
    char **rank_vars;
    int num_vars;
    int gen_axis;    /* index of the coord written as `var + N`, else -1 */
    int gen_offset;  /* the N in `var + N` (0 when gen_axis == -1) */
};

struct LiteralData {
    int is_list;
    union {
        struct {
            struct LiteralData **items;
            int count;
        } list;
        struct {
            double float_val;
            int int_val;
            int bool_val;
            DataType type;
        } scalar;
    } data;
};

/* ---- AST Nodes ---- */

typedef struct {
    char *name;
    int size;
} DimDeclData;

typedef struct {
    char *name;
    RankPair *ranks;
    int num_ranks;
    DataType datatype;
    double empty_val;
} TensorDeclData;

typedef struct {
    char *tensor_name;
    LiteralData *data;
} InitData;

typedef struct {
    char *watch_tensor;
} StopData;

typedef struct {
    TensorRef output;
    TensorRef input1;
    TensorRef input2;
    OperatorType map_op;
    MergeOp      map_merge_op;       /* defaults to MERGE_PASSTHROUGH */
    OperatorType reduce_op;
    MergeOp      reduce_merge_op;    /* defaults to MERGE_PASSTHROUGH */
} EinsumData;

typedef struct {
    AstNode **statements;
    int count;
    int capacity;
} ProgramData;

struct AstNode {
    NodeType type;
    union {
        ProgramData program;
        DimDeclData dim_decl;
        TensorDeclData tensor_decl;
        InitData init;
        EinsumData einsum;
        StopData stop;
    } data;
};

/* ---- Constructors ---- */
AstNode *ast_program_new(void);
void ast_program_add(AstNode *program, AstNode *stmt);

AstNode *ast_dim_decl_new(char *name, int size);
AstNode *ast_tensor_decl_new(char *name, RankPair *ranks, int num_ranks,
                              DataType datatype, double empty_val);
AstNode *ast_init_new(char *tensor_name, LiteralData *data);
AstNode *ast_einsum_new(TensorRef output, TensorRef input1, TensorRef input2,
                         OperatorType map_op, MergeOp map_merge_op,
                         OperatorType reduce_op, MergeOp reduce_merge_op);
AstNode *ast_stop_new(char *watch_tensor);

TensorRef tensor_ref_new(char *name, char **rank_vars, int num_vars);

LiteralData *literal_scalar_int(int val);
LiteralData *literal_scalar_float(double val);
LiteralData *literal_scalar_bool(int val);
LiteralData *literal_list_new(LiteralData **items, int count);

/* ---- Destructors ---- */
void ast_free(AstNode *node);
void literal_data_free(LiteralData *data);
void tensor_ref_free_contents(TensorRef *ref);

/* ---- Utilities ---- */
const char *datatype_to_str(DataType dt);
const char *operator_to_str(OperatorType op);
OperatorType operator_from_str(const char *name);
MergeOp merge_op_from_str(const char *name);     /* returns -1 on unknown */
const char *merge_op_to_str(MergeOp op);

#endif /* AST_H */
