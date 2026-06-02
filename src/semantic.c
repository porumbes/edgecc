#include "semantic.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdarg.h>
#include <math.h>
#include <limits.h>

/* ---- Error helpers ---- */

static void add_error(SemanticResult *res, const char *fmt, ...) {
    if (res->error_count >= res->error_capacity) {
        int new_cap = res->error_capacity == 0 ? 8 : res->error_capacity * 2;
        res->errors = realloc(res->errors, sizeof(char *) * new_cap);
        res->error_capacity = new_cap;
    }
    va_list args;
    va_start(args, fmt);
    /* determine required length */
    va_list args2;
    va_copy(args2, args);
    int len = vsnprintf(NULL, 0, fmt, args2);
    va_end(args2);
    char *buf = malloc(len + 1);
    vsnprintf(buf, len + 1, fmt, args);
    va_end(args);
    res->errors[res->error_count++] = buf;
}

/* ---- Symbol table helpers ---- */

static int find_dim(SymbolTable *st, const char *name) {
    for (int i = 0; i < st->num_dims; i++) {
        if (strcmp(st->dims[i].name, name) == 0)
            return st->dims[i].size;
    }
    return -1;
}

TensorEntry *symtab_find_tensor(SymbolTable *st, const char *name) {
    for (int i = 0; i < st->num_tensors; i++) {
        if (strcmp(st->tensors[i].name, name) == 0)
            return &st->tensors[i];
    }
    return NULL;
}

/* ---- Analysis passes ---- */

static void analyze_dim_decl(SemanticResult *res, AstNode *node) {
    DimDeclData *d = &node->data.dim_decl;
    SymbolTable *st = res->symtab;

    /* check for duplicate */
    if (find_dim(st, d->name) != -1) {
        add_error(res, "duplicate dim declaration: '%s'", d->name);
        return;
    }
    if (st->num_dims >= MAX_DIMS) {
        add_error(res, "too many dims (max %d)", MAX_DIMS);
        return;
    }
    st->dims[st->num_dims].name = d->name;
    st->dims[st->num_dims].size = d->size;
    st->num_dims++;
}

static void analyze_tensor_decl(SemanticResult *res, AstNode *node) {
    TensorDeclData *d = &node->data.tensor_decl;
    SymbolTable *st = res->symtab;

    /* check for duplicate */
    if (symtab_find_tensor(st, d->name) != NULL) {
        add_error(res, "duplicate tensor declaration: '%s'", d->name);
        return;
    }
    if (st->num_tensors >= MAX_TENSORS) {
        add_error(res, "too many tensors (max %d)", MAX_TENSORS);
        return;
    }
    if (d->num_ranks > 8) {
        add_error(res, "tensor '%s' has too many ranks (max 8)", d->name);
        return;
    }

    TensorEntry *te = &st->tensors[st->num_tensors];
    te->name = d->name;
    te->ranks = d->ranks;
    te->num_ranks = d->num_ranks;
    te->datatype = d->datatype;
    te->empty_val = d->empty_val;
    te->gen_axis = -1;

    for (int i = 0; i < d->num_ranks; i++) {
        if (d->ranks[i].is_generational) {
            if (te->gen_axis != -1) {
                add_error(res, "tensor '%s' has more than one generational rank",
                          d->name);
            }
            te->gen_axis = i;
            te->dim_sizes[i] = -1;   /* unbounded sentinel */
            continue;
        }
        int sz = find_dim(st, d->ranks[i].dim_name);
        if (sz == -1) {
            add_error(res, "tensor '%s' references undeclared dim '%s'",
                      d->name, d->ranks[i].dim_name);
            te->dim_sizes[i] = 0;
        } else {
            te->dim_sizes[i] = sz;
        }
    }
    st->num_tensors++;
}

/* Recursively validate that literal data matches expected dims.
   depth = current nesting depth (0 = outermost)
   num_ranks = total number of ranks for the tensor
   dim_sizes = array of sizes for each rank
   Returns 1 on success, 0 on failure (error already added). */
static int check_init_shape(SemanticResult *res, LiteralData *data,
                             int depth, int num_ranks, int *dim_sizes,
                             const char *tensor_name) {
    if (depth == num_ranks) {
        /* expect a scalar here */
        if (data->is_list) {
            add_error(res, "init for tensor '%s': too many nesting levels",
                      tensor_name);
            return 0;
        }
        return 1;
    }

    /* expect a list of size dim_sizes[depth] */
    if (!data->is_list) {
        add_error(res,
                  "init for tensor '%s': expected list at depth %d, got scalar",
                  tensor_name, depth);
        return 0;
    }

    int expected = dim_sizes[depth];
    int actual = data->data.list.count;
    if (actual != expected) {
        add_error(res,
                  "init for tensor '%s': at depth %d expected %d elements, got %d",
                  tensor_name, depth, expected, actual);
        return 0;
    }

    int ok = 1;
    for (int i = 0; i < actual; i++) {
        if (!check_init_shape(res, data->data.list.items[i],
                              depth + 1, num_ranks, dim_sizes, tensor_name)) {
            ok = 0;
        }
    }
    return ok;
}

static void analyze_init(SemanticResult *res, AstNode *node) {
    InitData *d = &node->data.init;
    SymbolTable *st = res->symtab;

    TensorEntry *te = symtab_find_tensor(st, d->tensor_name);
    if (te == NULL) {
        add_error(res, "init references undeclared tensor '%s'", d->tensor_name);
        return;
    }

    /* Build spatial dim sizes (excluding the generational axis, if any). */
    int spatial_sizes[8];
    int spatial_ranks = 0;
    for (int i = 0; i < te->num_ranks; i++) {
        if (i == te->gen_axis) continue;
        spatial_sizes[spatial_ranks++] = te->dim_sizes[i];
    }

    check_init_shape(res, d->data, 0, spatial_ranks, spatial_sizes,
                     d->tensor_name);
}

/* ---- Index tracking for einsum ---- */

typedef struct {
    char *name;
    int size; /* resolved size, -1 = unresolved, -2 = conflict */
} IndexInfo;

static int find_index(IndexInfo *arr, int count, const char *name) {
    for (int i = 0; i < count; i++) {
        if (strcmp(arr[i].name, name) == 0) return i;
    }
    return -1;
}

static void analyze_einsum(SemanticResult *res, AstNode *node) {
    EinsumData *d = &node->data.einsum;
    SymbolTable *st = res->symtab;

    /* validate all tensors exist */
    TensorEntry *out_te = symtab_find_tensor(st, d->output.tensor_name);
    TensorEntry *in1_te = symtab_find_tensor(st, d->input1.tensor_name);
    TensorEntry *in2_te = symtab_find_tensor(st, d->input2.tensor_name);

    if (out_te == NULL)
        add_error(res, "einsum: undeclared output tensor '%s'",
                  d->output.tensor_name);
    if (in1_te == NULL)
        add_error(res, "einsum: undeclared input tensor '%s'",
                  d->input1.tensor_name);
    if (in2_te == NULL)
        add_error(res, "einsum: undeclared input tensor '%s'",
                  d->input2.tensor_name);

    /* warn about nopass (degenerate operator that always produces empty output).
       Emitted before the undeclared-tensor early return because the warning has no
       dependency on tensor resolution — it inspects only the merge fields on the
       AST. */
    if (d->map_merge_op == MERGE_NOPASS) {
        fprintf(stderr,
            "warning: 'nopass' merge on map always produces empty output; "
            "the action has no effect\n");
    }
    if (d->reduce_merge_op == MERGE_NOPASS) {
        fprintf(stderr,
            "warning: 'nopass' merge on reduce always produces empty output; "
            "the action has no effect\n");
    }

    if (out_te == NULL || in1_te == NULL || in2_te == NULL) return;

    /* validate rank counts match num_vars */
    if (d->output.num_vars != out_te->num_ranks) {
        add_error(res,
                  "einsum: output tensor '%s' has %d ranks but %d index vars given",
                  d->output.tensor_name, out_te->num_ranks, d->output.num_vars);
    }
    if (d->input1.num_vars != in1_te->num_ranks) {
        add_error(res,
                  "einsum: input tensor '%s' has %d ranks but %d index vars given",
                  d->input1.tensor_name, in1_te->num_ranks, d->input1.num_vars);
    }
    if (d->input2.num_vars != in2_te->num_ranks) {
        add_error(res,
                  "einsum: input tensor '%s' has %d ranks but %d index vars given",
                  d->input2.tensor_name, in2_te->num_ranks, d->input2.num_vars);
    }

    /* ---- Generational rank handling ---- */
    /* The generational tensor is whichever operand's TensorEntry has gen_axis >= 0.
       For A there is exactly one such tensor; it must be the output and one input. */
    char *gen_var = NULL;   /* the index var bound to the gen rank */
    {
        TensorEntry *gen_te = NULL;
        if (out_te->gen_axis >= 0) gen_te = out_te;
        else if (in1_te->gen_axis >= 0) gen_te = in1_te;
        else if (in2_te->gen_axis >= 0) gen_te = in2_te;

        if (gen_te != NULL) {
            st->is_iterative = 1;
            st->gen_tensor = gen_te->name;

            /* Output must be the generational tensor, written at offset +1. */
            if (strcmp(d->output.tensor_name, gen_te->name) != 0) {
                add_error(res,
                    "generational tensor '%s' must be the output of the einsum",
                    gen_te->name);
            } else {
                if (d->output.gen_axis != out_te->gen_axis ||
                    d->output.gen_offset != 1) {
                    add_error(res,
                        "generational output '%s' must be written at offset +1 "
                        "on its generational rank (e.g. %s[i+1, ...])",
                        gen_te->name, gen_te->name);
                }
                if (d->output.gen_axis == out_te->gen_axis)
                    gen_var = d->output.rank_vars[out_te->gen_axis];
            }

            /* The matching read of the generational tensor must use offset 0
               (no `+N`) on its generational rank. */
            TensorRef *in_gen = NULL;
            TensorEntry *in_gen_te = NULL;
            if (strcmp(d->input1.tensor_name, gen_te->name) == 0) {
                in_gen = &d->input1; in_gen_te = in1_te;
            } else if (strcmp(d->input2.tensor_name, gen_te->name) == 0) {
                in_gen = &d->input2; in_gen_te = in2_te;
            }
            if (in_gen == NULL) {
                add_error(res,
                    "generational tensor '%s' must also appear as an input "
                    "read at generation i", gen_te->name);
            } else if (in_gen->gen_offset != 0) {
                add_error(res,
                    "generational input '%s' must be read at offset 0 "
                    "(e.g. %s[i, ...]), not with a +N offset",
                    gen_te->name, gen_te->name);
            } else if (gen_var != NULL &&
                       in_gen_te->gen_axis >= 0 &&
                       in_gen_te->gen_axis < in_gen->num_vars &&
                       strcmp(in_gen->rank_vars[in_gen_te->gen_axis], gen_var) != 0) {
                add_error(res,
                    "generational rank variable mismatch: output uses '%s' but "
                    "input uses '%s'",
                    gen_var, in_gen->rank_vars[in_gen_te->gen_axis]);
            }
        }
    }

    /* collect all indices and resolve sizes */
    IndexInfo indices[MAX_INDICES * 3];
    int num_indices = 0;

    /* helper: register an index with a size */
    /* inlined as a local macro for clarity */
#define REGISTER_INDEX(varname, sz) do {                                    \
    int _idx = find_index(indices, num_indices, (varname));                 \
    if (_idx == -1) {                                                       \
        if (num_indices >= (int)(sizeof(indices)/sizeof(indices[0]))) {     \
            add_error(res, "einsum: too many index variables");             \
            break;                                                           \
        }                                                                   \
        indices[num_indices].name = (varname);                              \
        indices[num_indices].size = (sz);                                   \
        num_indices++;                                                       \
    } else {                                                                \
        int existing = indices[_idx].size;                                  \
        if (existing != -2 && existing != (sz)) {                           \
            add_error(res,                                                   \
                      "einsum: index '%s' bound to size %d and size %d",    \
                      (varname), existing, (sz));                            \
            indices[_idx].size = -2; /* mark conflict */                    \
        }                                                                   \
    }                                                                       \
} while(0)

    for (int i = 0; i < d->output.num_vars && i < out_te->num_ranks; i++)
        if (i != out_te->gen_axis)
            REGISTER_INDEX(d->output.rank_vars[i], out_te->dim_sizes[i]);
    for (int i = 0; i < d->input1.num_vars && i < in1_te->num_ranks; i++)
        if (i != in1_te->gen_axis)
            REGISTER_INDEX(d->input1.rank_vars[i], in1_te->dim_sizes[i]);
    for (int i = 0; i < d->input2.num_vars && i < in2_te->num_ranks; i++)
        if (i != in2_te->gen_axis)
            REGISTER_INDEX(d->input2.rank_vars[i], in2_te->dim_sizes[i]);

#undef REGISTER_INDEX

    if (gen_var != NULL) st->gen_rank_var = gen_var;

    /* classify indices into free (in output) or reduction (not in output) */
    st->num_free_indices = 0;
    st->num_reduction_indices = 0;

    for (int i = 0; i < num_indices; i++) {
        char *iname = indices[i].name;
        int isize = indices[i].size < 0 ? 0 : indices[i].size;

        /* check if in output */
        int in_output = 0;
        for (int j = 0; j < d->output.num_vars; j++) {
            if (strcmp(d->output.rank_vars[j], iname) == 0) {
                in_output = 1;
                break;
            }
        }

        if (in_output) {
            if (st->num_free_indices < MAX_INDICES) {
                st->free_indices[st->num_free_indices].name = iname;
                st->free_indices[st->num_free_indices].size = isize;
                st->free_indices[st->num_free_indices].is_free = 1;
                st->num_free_indices++;
            }
        } else {
            if (st->num_reduction_indices < MAX_INDICES) {
                st->reduction_indices[st->num_reduction_indices].name = iname;
                st->reduction_indices[st->num_reduction_indices].size = isize;
                st->reduction_indices[st->num_reduction_indices].is_free = 0;
                st->num_reduction_indices++;
            }
        }
    }

}

static void analyze_stop(SemanticResult *res, AstNode *node) {
    SymbolTable *st = res->symtab;
    const char *watch = node->data.stop.watch_tensor;
    TensorEntry *te = symtab_find_tensor(st, watch);
    if (te == NULL) {
        add_error(res, "stop: undeclared tensor '%s'", watch);
        return;
    }
    if (te->gen_axis < 0) {
        add_error(res, "stop: tensor '%s' is not generational", watch);
    }
    st->stop_watch = te->name;
}

/* ---- Public API ---- */

SemanticResult semantic_analyze(AstNode *program) {
    SemanticResult res;
    memset(&res, 0, sizeof(res));

    res.symtab = calloc(1, sizeof(SymbolTable));

    if (program == NULL || program->type != NODE_PROGRAM) {
        add_error(&res, "invalid program node");
        return res;
    }

    ProgramData *pd = &program->data.program;
    for (int i = 0; i < pd->count; i++) {
        AstNode *stmt = pd->statements[i];
        switch (stmt->type) {
            case NODE_DIM_DECL:
                analyze_dim_decl(&res, stmt);
                break;
            case NODE_TENSOR_DECL:
                analyze_tensor_decl(&res, stmt);
                break;
            case NODE_INIT:
                analyze_init(&res, stmt);
                break;
            case NODE_EINSUM:
                analyze_einsum(&res, stmt);
                break;
            case NODE_STOP:
                analyze_stop(&res, stmt);
                break;
            default:
                add_error(&res, "unknown statement type %d", stmt->type);
                break;
        }
    }

    /* Iterative consistency: a generational einsum needs a stop clause that
       watches the generational tensor, and vice versa. */
    SymbolTable *st = res.symtab;
    if (st->is_iterative && st->stop_watch == NULL) {
        add_error(&res, "generational einsum on '%s' requires a 'stop: stable(...)' clause",
                  st->gen_tensor ? st->gen_tensor : "?");
    }
    if (st->stop_watch != NULL && !st->is_iterative) {
        add_error(&res, "'stop' clause present but no generational einsum");
    }
    if (st->is_iterative && st->stop_watch != NULL && st->gen_tensor != NULL &&
        strcmp(st->stop_watch, st->gen_tensor) != 0) {
        add_error(&res, "stop watches '%s' but the generational tensor is '%s'",
                  st->stop_watch, st->gen_tensor);
    }

    return res;
}

void semantic_result_free(SemanticResult *res) {
    for (int i = 0; i < res->error_count; i++) {
        free(res->errors[i]);
    }
    free(res->errors);
    free(res->symtab);
    res->errors = NULL;
    res->symtab = NULL;
    res->error_count = 0;
    res->error_capacity = 0;
}

double compute_op_identity(OperatorType op, DataType dt) {
    switch (op) {
        case OP_ADD: case OP_SUB:
            return 0.0;
        case OP_MUL:
            return 1.0;
        case OP_MIN:
            return (dt == TYPE_INTEGER) ? (double)INT_MAX : INFINITY;
        case OP_MAX:
            return (dt == TYPE_INTEGER) ? (double)INT_MIN : -INFINITY;
        case OP_AND:
            return 1.0;     /* true */
        case OP_OR:
            return 0.0;     /* false */
    }
    return 0.0;
}
