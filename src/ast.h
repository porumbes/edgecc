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
    NODE_EINSUM_FUSED,
    NODE_STOP,
    NODE_SAVE
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
    OP_OR,
    OP_TAKE_LEFT
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
    int negate;      /* 1 if operand carries a unary `!` (existence complement) */
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
    int is_inferred;   /* 1 if declared `dim N = *;` (size from a loaded file) */
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
    int is_load;          /* 1 if `init X = load("path");` */
    int is_seed;          /* 1 if `init X = seed(source=I, value=V);` */
    LiteralData *data;    /* literal init (NULL when is_load/is_seed) */
    char *path;           /* file path (NULL unless is_load) */
    int seed_source;      /* source index (valid when is_seed) */
    double seed_value;    /* value at source (valid when is_seed) */
} InitData;

typedef enum {
    STOP_STABLE,      /* stop: stable(X);      sugar for X[i+1] == X[i], resolved in semantic */
    STOP_EQUAL,       /* stop: X[g] == X[h];                                             */
    STOP_OCCUPANCY    /* stop: ||X[g]|| == N;  N must be 0 (checked in semantic)         */
} StopKind;

typedef struct {
    StopKind  kind;
    char     *stable_tensor;  /* STOP_STABLE: the X in stable(X); else NULL */
    TensorRef lhs;            /* STOP_EQUAL / STOP_OCCUPANCY; zeroed otherwise */
    TensorRef rhs;            /* STOP_EQUAL only; zeroed otherwise */
    int       occupancy_rhs;  /* STOP_OCCUPANCY: the integer after '==' */
} StopData;

typedef struct {
    char *tensor_name;
    char *path;
} SaveData;

typedef struct {
    TensorRef output;
    TensorRef input1;
    TensorRef input2;
    OperatorType map_op;
    MergeOp      map_merge_op;       /* defaults to MERGE_PASSTHROUGH */
    OperatorType reduce_op;
    MergeOp      reduce_merge_op;    /* defaults to MERGE_PASSTHROUGH */
    char        *reduce_over;        /* optional `over=` rank for single-action reduce; NULL if none */
} EinsumData;

typedef struct {
    TensorRef output;
    TensorRef inner1;        /* paren left operand  (action 1 left) */
    TensorRef inner2;        /* paren right operand (action 1 right) */
    TensorRef outer;         /* the non-paren outer operand */
    int       inter_is_left2;/* 1 if the intermediate is action 2's LEFT operand */
    /* action 1 (inner, label .1) */
    OperatorType map1_op;    MergeOp map1_merge;
    int          a1_has_reduce;
    OperatorType reduce1_op; MergeOp reduce1_merge;
    char        *over1;      /* optional `over=` rank, else NULL */
    /* action 2 (outer, label .2) */
    OperatorType map2_op;    MergeOp map2_merge;
    int          a2_has_reduce;
    OperatorType reduce2_op; MergeOp reduce2_merge;
    char        *over2;      /* optional `over=` rank, else NULL */
} EinsumFusedData;

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
        EinsumFusedData einsum_fused;
        StopData stop;
        SaveData save;
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
AstNode *ast_einsum_fused_new(EinsumFusedData data);
AstNode *ast_stop_stable_new(char *tensor);
AstNode *ast_stop_equal_new(TensorRef lhs, TensorRef rhs);
AstNode *ast_stop_occupancy_new(TensorRef ref, int rhs_value);
AstNode *ast_init_load_new(char *tensor_name, char *path);
AstNode *ast_init_seed_new(char *tensor_name, int source, double value);
AstNode *ast_dim_decl_inferred_new(char *name);
AstNode *ast_save_new(char *tensor_name, char *path);

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
