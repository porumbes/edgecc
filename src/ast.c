#include "ast.h"
#include <stdio.h>
#include <string.h>

/* ---- Constructors ---- */

AstNode *ast_program_new(void) {
    AstNode *node = calloc(1, sizeof(AstNode));
    node->type = NODE_PROGRAM;
    node->data.program.capacity = 8;
    node->data.program.statements = calloc(8, sizeof(AstNode *));
    node->data.program.count = 0;
    return node;
}

void ast_program_add(AstNode *program, AstNode *stmt) {
    ProgramData *p = &program->data.program;
    if (p->count >= p->capacity) {
        p->capacity *= 2;
        p->statements = realloc(p->statements, p->capacity * sizeof(AstNode *));
    }
    p->statements[p->count++] = stmt;
}

AstNode *ast_dim_decl_new(char *name, int size) {
    AstNode *node = calloc(1, sizeof(AstNode));
    node->type = NODE_DIM_DECL;
    node->data.dim_decl.name = name;
    node->data.dim_decl.size = size;
    return node;
}

AstNode *ast_tensor_decl_new(char *name, RankPair *ranks, int num_ranks,
                              DataType datatype, double empty_val) {
    AstNode *node = calloc(1, sizeof(AstNode));
    node->type = NODE_TENSOR_DECL;
    node->data.tensor_decl.name = name;
    node->data.tensor_decl.ranks = ranks;
    node->data.tensor_decl.num_ranks = num_ranks;
    node->data.tensor_decl.datatype = datatype;
    node->data.tensor_decl.empty_val = empty_val;
    return node;
}

AstNode *ast_init_new(char *tensor_name, LiteralData *data) {
    AstNode *node = calloc(1, sizeof(AstNode));
    node->type = NODE_INIT;
    node->data.init.tensor_name = tensor_name;
    node->data.init.data = data;
    return node;
}

AstNode *ast_einsum_new(TensorRef output, TensorRef input1, TensorRef input2,
                         OperatorType map_op, MergeOp map_merge_op,
                         OperatorType reduce_op, MergeOp reduce_merge_op) {
    AstNode *node = calloc(1, sizeof(AstNode));
    node->type = NODE_EINSUM;
    node->data.einsum.output = output;
    node->data.einsum.input1 = input1;
    node->data.einsum.input2 = input2;
    node->data.einsum.map_op = map_op;
    node->data.einsum.map_merge_op = map_merge_op;
    node->data.einsum.reduce_op = reduce_op;
    node->data.einsum.reduce_merge_op = reduce_merge_op;
    return node;
}

TensorRef tensor_ref_new(char *name, char **rank_vars, int num_vars) {
    TensorRef ref;
    ref.tensor_name = name;
    ref.rank_vars = rank_vars;
    ref.num_vars = num_vars;
    return ref;
}

LiteralData *literal_scalar_int(int val) {
    LiteralData *d = calloc(1, sizeof(LiteralData));
    d->is_list = 0;
    d->data.scalar.type = TYPE_INTEGER;
    d->data.scalar.int_val = val;
    d->data.scalar.float_val = (double)val;
    return d;
}

LiteralData *literal_scalar_float(double val) {
    LiteralData *d = calloc(1, sizeof(LiteralData));
    d->is_list = 0;
    d->data.scalar.type = TYPE_FLOAT;
    d->data.scalar.float_val = val;
    return d;
}

LiteralData *literal_scalar_bool(int val) {
    LiteralData *d = calloc(1, sizeof(LiteralData));
    d->is_list = 0;
    d->data.scalar.type = TYPE_BOOLEAN;
    d->data.scalar.bool_val = val;
    return d;
}

LiteralData *literal_list_new(LiteralData **items, int count) {
    LiteralData *d = calloc(1, sizeof(LiteralData));
    d->is_list = 1;
    d->data.list.items = items;
    d->data.list.count = count;
    return d;
}

/* ---- Destructors ---- */

void tensor_ref_free_contents(TensorRef *ref) {
    free(ref->tensor_name);
    for (int i = 0; i < ref->num_vars; i++) {
        free(ref->rank_vars[i]);
    }
    free(ref->rank_vars);
}

void literal_data_free(LiteralData *data) {
    if (!data) return;
    if (data->is_list) {
        for (int i = 0; i < data->data.list.count; i++) {
            literal_data_free(data->data.list.items[i]);
        }
        free(data->data.list.items);
    }
    free(data);
}

void ast_free(AstNode *node) {
    if (!node) return;
    switch (node->type) {
        case NODE_PROGRAM:
            for (int i = 0; i < node->data.program.count; i++) {
                ast_free(node->data.program.statements[i]);
            }
            free(node->data.program.statements);
            break;
        case NODE_DIM_DECL:
            free(node->data.dim_decl.name);
            break;
        case NODE_TENSOR_DECL:
            free(node->data.tensor_decl.name);
            for (int i = 0; i < node->data.tensor_decl.num_ranks; i++) {
                free(node->data.tensor_decl.ranks[i].rank_name);
                free(node->data.tensor_decl.ranks[i].dim_name);
            }
            free(node->data.tensor_decl.ranks);
            break;
        case NODE_INIT:
            free(node->data.init.tensor_name);
            literal_data_free(node->data.init.data);
            break;
        case NODE_EINSUM:
            tensor_ref_free_contents(&node->data.einsum.output);
            tensor_ref_free_contents(&node->data.einsum.input1);
            tensor_ref_free_contents(&node->data.einsum.input2);
            break;
    }
    free(node);
}

/* ---- Utilities ---- */

const char *datatype_to_str(DataType dt) {
    switch (dt) {
        case TYPE_INTEGER: return "integer";
        case TYPE_FLOAT:   return "float";
        case TYPE_BOOLEAN: return "boolean";
    }
    return "unknown";
}

const char *operator_to_str(OperatorType op) {
    switch (op) {
        case OP_ADD: return "add";
        case OP_SUB: return "sub";
        case OP_MUL: return "mul";
        case OP_MIN: return "min";
        case OP_MAX: return "max";
        case OP_AND: return "and";
        case OP_OR:  return "or";
    }
    return "unknown";
}

OperatorType operator_from_str(const char *name) {
    if (strcmp(name, "add") == 0) return OP_ADD;
    if (strcmp(name, "sub") == 0) return OP_SUB;
    if (strcmp(name, "mul") == 0) return OP_MUL;
    if (strcmp(name, "min") == 0) return OP_MIN;
    if (strcmp(name, "max") == 0) return OP_MAX;
    if (strcmp(name, "and") == 0) return OP_AND;
    if (strcmp(name, "or") == 0)  return OP_OR;
    return -1;
}

MergeOp merge_op_from_str(const char *name) {
    if (strcmp(name, "passthrough") == 0)      return MERGE_PASSTHROUGH;
    if (strcmp(name, "nopass") == 0)           return MERGE_NOPASS;
    if (strcmp(name, "intersect") == 0)        return MERGE_INTERSECT;
    if (strcmp(name, "take_left") == 0)        return MERGE_TAKE_LEFT;
    if (strcmp(name, "take_left_only") == 0)   return MERGE_TAKE_LEFT_ONLY;
    if (strcmp(name, "take_right") == 0)       return MERGE_TAKE_RIGHT;
    if (strcmp(name, "take_right_only") == 0)  return MERGE_TAKE_RIGHT_ONLY;
    if (strcmp(name, "xor") == 0)              return MERGE_XOR;
    if (strcmp(name, "union") == 0)            return MERGE_UNION;
    return (MergeOp)-1;
}

const char *merge_op_to_str(MergeOp op) {
    switch (op) {
        case MERGE_PASSTHROUGH:     return "passthrough";
        case MERGE_NOPASS:          return "nopass";
        case MERGE_INTERSECT:       return "intersect";
        case MERGE_TAKE_LEFT:       return "take_left";
        case MERGE_TAKE_LEFT_ONLY:  return "take_left_only";
        case MERGE_TAKE_RIGHT:      return "take_right";
        case MERGE_TAKE_RIGHT_ONLY: return "take_right_only";
        case MERGE_XOR:             return "xor";
        case MERGE_UNION:           return "union";
    }
    return "?";
}
