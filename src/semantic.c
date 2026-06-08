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
    st->dims[st->num_dims].is_inferred = d->is_inferred;
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
                             int *dim_inferred, const char *tensor_name) {
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
    /* Inferred dims have no compile-time size; a literal init's length is only
       checked against the loaded shape at runtime (see codegen shape assert). */
    if (!dim_inferred[depth] && actual != expected) {
        add_error(res,
                  "init for tensor '%s': at depth %d expected %d elements, got %d",
                  tensor_name, depth, expected, actual);
        return 0;
    }

    int ok = 1;
    for (int i = 0; i < actual; i++) {
        if (!check_init_shape(res, data->data.list.items[i],
                              depth + 1, num_ranks, dim_sizes, dim_inferred,
                              tensor_name)) {
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
    te->has_init = 1;
    te->loaded_from_file = d->is_load;

    /* load("...") inits have no literal data; skip shape checking. */
    if (d->is_load) return;

    /* Build spatial dim sizes (excluding the generational axis, if any),
       and a parallel flag marking which axes are inferred (`dim = *`). */
    int spatial_sizes[8];
    int spatial_inferred[8];
    int spatial_ranks = 0;
    for (int i = 0; i < te->num_ranks; i++) {
        if (i == te->gen_axis) continue;
        spatial_sizes[spatial_ranks] = te->dim_sizes[i];
        int inferred = 0;
        const char *dn = te->ranks[i].dim_name;
        if (dn) {
            for (int k = 0; k < st->num_dims; k++) {
                if (strcmp(st->dims[k].name, dn) == 0) {
                    inferred = st->dims[k].is_inferred;
                    break;
                }
            }
        }
        spatial_inferred[spatial_ranks] = inferred;
        spatial_ranks++;
    }

    check_init_shape(res, d->data, 0, spatial_ranks, spatial_sizes,
                     spatial_inferred, d->tensor_name);
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

static void analyze_einsum(SemanticResult *res, AstNode *node, int eidx) {
    EinsumData *d = &node->data.einsum;
    if (d->reduce_op == OP_TAKE_LEFT) {
        add_error(res,
            "take_left is a map-only compute operator; "
            "it cannot be used as a reduce operator");
    }
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

    /* ---- Generational rank handling (cascade-aware) ---- */
    char *gen_var = NULL;   /* the index var bound to the gen rank, this einsum */
    {
        TensorEntry *tes[3]  = { out_te, in1_te, in2_te };
        TensorRef   *refs[3] = { &d->output, &d->input1, &d->input2 };

        /* Detect the shared generational rank variable. */
        for (int k = 0; k < 3; k++) {
            if (tes[k]->gen_axis >= 0 && tes[k]->gen_axis < refs[k]->num_vars) {
                st->is_iterative = 1;
                char *v = refs[k]->rank_vars[tes[k]->gen_axis];
                if (gen_var == NULL) gen_var = v;
                if (st->gen_rank_var == NULL) {
                    st->gen_rank_var = v;
                } else if (strcmp(st->gen_rank_var, v) != 0) {
                    add_error(res,
                        "generational rank variable mismatch: '%s' vs '%s'",
                        st->gen_rank_var, v);
                }
            }
        }

        /* Classify the OUTPUT's generational role by its effective write offset
           on the declared generational axis. A `+N` offset on the ref, if
           present, must be on that axis. */
        if (out_te->gen_axis >= 0) {
            if (d->output.gen_axis >= 0 && d->output.gen_axis != out_te->gen_axis) {
                add_error(res,
                    "generational tensor '%s' written with an offset on the wrong axis",
                    out_te->name);
            } else {
                int eff_off = (d->output.gen_axis == out_te->gen_axis)
                              ? d->output.gen_offset : 0;
                if (eff_off == 1) {
                    /* cross-generational write: D[i+1, ...] — collect, allow many. */
                    int seen = 0;
                    for (int m = 0; m < st->num_gen_tensors; m++)
                        if (strcmp(st->gen_tensors[m], out_te->name) == 0) seen = 1;
                    if (!seen && st->num_gen_tensors < MAX_TENSORS) {
                        st->gen_tensors[st->num_gen_tensors++] = out_te->name;
                        if (st->gen_tensor == NULL) st->gen_tensor = out_te->name;
                    }
                } else if (eff_off == 0) {
                    /* intra-generation intermediate write: T[i, ...] */
                    int seen = 0;
                    for (int m = 0; m < st->num_intermediates; m++)
                        if (strcmp(st->intermediates[m], out_te->name) == 0) seen = 1;
                    if (!seen && st->num_intermediates < MAX_TENSORS)
                        st->intermediates[st->num_intermediates++] = out_te->name;
                } else {
                    add_error(res,
                        "generational tensor '%s' written at unsupported offset +%d "
                        "(only i and i+1 are allowed)", out_te->name, eff_off);
                }
            }
        }

        /* Generational inputs may be read at offset 0 (i) or +1 (i+1, the
           just-written cross-gen value from an earlier Einsum). Offsets >1 are
           unsupported. The "written before read at i+1" SSA rule is enforced in
           validate_cascade_ordering. */
        for (int k = 1; k < 3; k++) {
            if (tes[k]->gen_axis >= 0 && refs[k]->gen_offset > 1) {
                add_error(res,
                    "generational tensor '%s' read at unsupported offset +%d "
                    "(only i and i+1 are allowed)", tes[k]->name, refs[k]->gen_offset);
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

    /* Copy this Einsum's classification into the per-Einsum slot (cascade support).
       The global free/reduction fields remain populated for the single-Einsum path. */
    if (eidx >= 0 && eidx < MAX_EINSUMS) {
        EinsumClass *ec = &st->einsum_class[eidx];
        ec->num_free_indices = st->num_free_indices;
        for (int i = 0; i < st->num_free_indices; i++)
            ec->free_indices[i] = st->free_indices[i];
        ec->num_reduction_indices = st->num_reduction_indices;
        for (int i = 0; i < st->num_reduction_indices; i++)
            ec->reduction_indices[i] = st->reduction_indices[i];
        if (eidx + 1 > st->num_einsums) st->num_einsums = eidx + 1;
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

/* Validate SSA ordering within a generation:
   (1) an intra-generation intermediate must be written by an earlier Einsum
       before a later Einsum reads it;
   (2) a cross-generational tensor read at i+1 must already be written at i+1 by
       an earlier Einsum (next-reads-next). */
static void validate_cascade_ordering(SemanticResult *res, AstNode *program) {
    SymbolTable *st = res->symtab;
    if (!st->is_iterative) return;

    char *written[MAX_TENSORS];      int nwritten = 0;     /* intermediates written so far */
    char *cg_written[MAX_TENSORS];   int ncg_written = 0;  /* cross-gen written at i+1 so far */
    ProgramData *pd = &program->data.program;
    for (int i = 0; i < pd->count; i++) {
        AstNode *stmt = pd->statements[i];
        if (stmt->type != NODE_EINSUM) continue;
        EinsumData *e = &stmt->data.einsum;

        TensorRef *ins[2] = { &e->input1, &e->input2 };
        for (int k = 0; k < 2; k++) {
            const char *nm = ins[k]->tensor_name;
            /* (1) intermediate read-before-write */
            int is_inter = 0;
            for (int m = 0; m < st->num_intermediates; m++)
                if (strcmp(st->intermediates[m], nm) == 0) is_inter = 1;
            if (is_inter) {
                int found = 0;
                for (int w = 0; w < nwritten; w++)
                    if (strcmp(written[w], nm) == 0) found = 1;
                if (!found)
                    add_error(res, "intermediate '%s' is read before it is written", nm);
            }
            /* (2) cross-gen i+1 read-before-write */
            TensorEntry *te = symtab_find_tensor(st, nm);
            int eff = (te && ins[k]->gen_axis == te->gen_axis) ? ins[k]->gen_offset : 0;
            if (eff == 1) {
                int found = 0;
                for (int w = 0; w < ncg_written; w++)
                    if (strcmp(cg_written[w], nm) == 0) found = 1;
                if (!found)
                    add_error(res,
                        "cross-generational tensor '%s' is read at i+1 before it is written", nm);
            }
        }

        /* record this Einsum's output */
        const char *onm = e->output.tensor_name;
        for (int m = 0; m < st->num_intermediates; m++) {
            if (strcmp(st->intermediates[m], onm) == 0) {
                written[nwritten++] = st->intermediates[m]; break;
            }
        }
        for (int m = 0; m < st->num_gen_tensors; m++) {
            if (strcmp(st->gen_tensors[m], onm) == 0) {
                cg_written[ncg_written++] = st->gen_tensors[m]; break;
            }
        }
    }
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
    int einsum_index = 0;
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
                analyze_einsum(&res, stmt, einsum_index++);
                break;
            case NODE_STOP:
                analyze_stop(&res, stmt);
                break;
            default:
                add_error(&res, "unknown statement type %d", stmt->type);
                break;
        }
    }

    validate_cascade_ordering(&res, program);

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
    /* A generational program must write at least one cross-generational tensor
       (a gen-rank tensor at offset i+1); otherwise nothing crosses generations. */
    if (st->is_iterative && st->num_gen_tensors == 0) {
        add_error(&res, "generational program has no cross-generational tensor "
                        "(write one at offset i+1, e.g. F[i+1, ...])");
    }
    if (st->is_iterative && st->stop_watch != NULL && st->num_gen_tensors > 0) {
        int ok = 0;
        for (int m = 0; m < st->num_gen_tensors; m++)
            if (strcmp(st->stop_watch, st->gen_tensors[m]) == 0) ok = 1;
        if (!ok)
            add_error(&res,
                "stop watches '%s' but it is not a cross-generational tensor",
                st->stop_watch);
    }

    /* Every cross-generational tensor needs a generation-0 init. */
    for (int m = 0; m < st->num_gen_tensors; m++) {
        TensorEntry *cg = symtab_find_tensor(st, st->gen_tensors[m]);
        if (cg != NULL && !cg->has_init) {
            add_error(&res,
                "cross-generational tensor '%s' requires an init (generation 0)",
                st->gen_tensors[m]);
        }
    }
    for (int m = 0; m < st->num_intermediates; m++) {
        TensorEntry *te = symtab_find_tensor(st, st->intermediates[m]);
        if (te != NULL && te->has_init) {
            add_error(&res,
                "intermediate tensor '%s' must not have an init "
                "(it is recomputed each generation)", st->intermediates[m]);
        }
    }

    /* Every cross-generational tensor must be read at generation i somewhere
       (so its generation-0 value is wired into run() as a parameter). */
    for (int m = 0; st->is_iterative && m < st->num_gen_tensors; m++) {
        const char *gt = st->gen_tensors[m];
        int read_at_i = 0;
        for (int i = 0; i < pd->count; i++) {
            AstNode *s = pd->statements[i];
            if (s->type != NODE_EINSUM) continue;
            EinsumData *e = &s->data.einsum;
            TensorEntry *t1 = symtab_find_tensor(st, e->input1.tensor_name);
            TensorEntry *t2 = symtab_find_tensor(st, e->input2.tensor_name);
            int eff1 = (t1 && e->input1.gen_axis == t1->gen_axis) ? e->input1.gen_offset : 0;
            int eff2 = (t2 && e->input2.gen_axis == t2->gen_axis) ? e->input2.gen_offset : 0;
            if ((strcmp(e->input1.tensor_name, gt) == 0 && eff1 == 0) ||
                (strcmp(e->input2.tensor_name, gt) == 0 && eff2 == 0)) {
                read_at_i = 1; break;
            }
        }
        if (!read_at_i) {
            add_error(&res,
                "cross-generational tensor '%s' must be read at generation i "
                "somewhere in the cascade", gt);
        }
    }

    /* Save directive: target must be a declared tensor. */
    for (int i = 0; i < pd->count; i++) {
        AstNode *s = pd->statements[i];
        if (s->type != NODE_SAVE) continue;
        if (!symtab_find_tensor(st, s->data.save.tensor_name)) {
            add_error(&res, "save: unknown tensor '%s'", s->data.save.tensor_name);
            continue;
        }
        st->save_tensor = s->data.save.tensor_name;  /* borrowed */
        st->save_path   = s->data.save.path;          /* borrowed */
    }

    /* Every inferred dim must be pinned by a loaded tensor. */
    for (int d = 0; d < st->num_dims; d++) {
        if (!st->dims[d].is_inferred) continue;
        int pinned = 0;
        for (int t = 0; t < st->num_tensors && !pinned; t++) {
            if (!st->tensors[t].loaded_from_file) continue;
            for (int k = 0; k < st->tensors[t].num_ranks; k++) {
                const char *dn = st->tensors[t].ranks[k].dim_name;
                if (dn && strcmp(dn, st->dims[d].name) == 0) { pinned = 1; break; }
            }
        }
        if (!pinned)
            add_error(&res,
                      "inferred dim '%s' is not pinned by any load(...) tensor",
                      st->dims[d].name);
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
        case OP_TAKE_LEFT:
            return 0.0;     /* map-only; identity unused */
    }
    return 0.0;
}
