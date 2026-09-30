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

/* ---- Reserved identifiers ----
 * Dim names, tensor names, and rank variables are emitted verbatim into the
 * generated Python, so they must not collide with Python keywords, the
 * builtins/modules the generated code uses, or codegen's own temporaries.
 * Names starting with '_' are reserved wholesale for codegen. Keep this list
 * in sync with codegen.c -- tests/reserved/run_reserved_names.py checks it
 * against what the compiler actually emits. */
/* BEGIN PY_RESERVED */
static const char *const PY_RESERVED[] = {
    /* Python keywords */
    "False", "None", "True", "and", "as", "assert", "async", "await", "break",
    "class", "continue", "def", "del", "elif", "else", "except", "finally",
    "for", "from", "global", "if", "import", "in", "is", "lambda", "nonlocal",
    "not", "or", "pass", "raise", "return", "try", "while", "with", "yield",
    /* builtins, modules, and top-level names used by generated code */
    "ImportError", "RuntimeError", "SystemExit", "ValueError", "hasattr",
    "isinstance", "len", "list", "max", "min", "open", "print", "range",
    "json", "math", "mmread", "mmwrite", "np", "run", "sys",
    /* codegen temporaries */
    "a_exists", "a_op", "a_val", "b_exists", "b_m", "b_op", "b_s", "b_val",
    "c_exists", "c_val", "i_op", "map_emitted", "map_val", "new_state",
    "o_op", "state", "val_for_red",
};
/* END PY_RESERVED */

static int is_py_reserved(const char *name) {
    if (name[0] == '_') return 1;
    for (size_t i = 0; i < sizeof(PY_RESERVED) / sizeof(PY_RESERVED[0]); i++)
        if (strcmp(name, PY_RESERVED[i]) == 0) return 1;
    return 0;
}

/* A generational tensor T owns the generated names T_prev and T_next. */
static int is_gen_buffer_name(AstNode *program, const char *name) {
    ProgramData *pd = &program->data.program;
    for (int i = 0; i < pd->count; i++) {
        AstNode *s = pd->statements[i];
        if (s->type != NODE_TENSOR_DECL) continue;
        TensorDeclData *t = &s->data.tensor_decl;
        int gen = 0;
        for (int r = 0; r < t->num_ranks; r++) gen |= t->ranks[r].is_generational;
        size_t n = strlen(t->name);
        if (gen && strncmp(name, t->name, n) == 0 &&
            (strcmp(name + n, "_prev") == 0 || strcmp(name + n, "_next") == 0))
            return 1;
    }
    return 0;
}

static AstNode *find_decl(AstNode *program, NodeType type, const char *name) {
    ProgramData *pd = &program->data.program;
    for (int i = 0; i < pd->count; i++) {
        AstNode *s = pd->statements[i];
        if (s->type != type) continue;
        const char *n = type == NODE_DIM_DECL ? s->data.dim_decl.name : s->data.tensor_decl.name;
        if (strcmp(n, name) == 0) return s;
    }
    return NULL;
}

/* Names already reported, so a name used in several places yields one
   error, not one per occurrence. */
typedef struct { const char *names[64]; int n; } ReportedNames;

static int first_report(ReportedNames *seen, const char *name) {
    for (int i = 0; i < seen->n; i++)
        if (strcmp(seen->names[i], name) == 0) return 0;
    if (seen->n < 64) seen->names[seen->n++] = name;
    return 1;
}

static void check_reserved(SemanticResult *res, ReportedNames *seen, AstNode *program,
                           const char *what, const char *name) {
    const char *why = is_py_reserved(name) ? "keyword, builtin, or compiler temporary"
                    : is_gen_buffer_name(program, name) ? "buffer of a generational tensor"
                    : NULL;
    if (why && first_report(seen, name))
        add_error(res, "%s '%s' is reserved in the generated Python (%s); rename it",
                  what, name, why);
}

static void check_ref_vars(SemanticResult *res, ReportedNames *seen,
                           ReportedNames *shadowed, AstNode *program, const TensorRef *r) {
    AstNode *decl = find_decl(program, NODE_TENSOR_DECL, r->tensor_name);
    for (int k = 0; k < r->num_vars; k++) {
        /* The generational coordinate is never emitted (the loop counter is
           _it), so its variable cannot collide with anything. */
        if (decl && k < decl->data.tensor_decl.num_ranks &&
            decl->data.tensor_decl.ranks[k].is_generational)
            continue;
        const char *v = r->rank_vars[k];
        check_reserved(res, seen, program, "rank variable", v);
        /* Other rank variables become loop variables, which would shadow a
           dim or tensor of the same name inside run(). */
        const char *kind = find_decl(program, NODE_DIM_DECL, v) ? "dim"
                         : find_decl(program, NODE_TENSOR_DECL, v) ? "tensor" : NULL;
        if (kind && first_report(shadowed, v))
            add_error(res, "rank variable '%s' shadows %s '%s' in the generated Python; rename it",
                      v, kind, v);
    }
}

static void validate_identifiers(SemanticResult *res, AstNode *program) {
    ProgramData *pd = &program->data.program;
    ReportedNames seen = { {0}, 0 }, shadowed = { {0}, 0 };
    for (int i = 0; i < pd->count; i++) {
        AstNode *s = pd->statements[i];
        switch (s->type) {
            case NODE_DIM_DECL:
                check_reserved(res, &seen, program, "dim", s->data.dim_decl.name);
                break;
            case NODE_TENSOR_DECL: {
                const char *name = s->data.tensor_decl.name;
                check_reserved(res, &seen, program, "tensor", name);
                if (find_decl(program, NODE_DIM_DECL, name))
                    add_error(res, "tensor '%s' has the same name as dim '%s' "
                                   "(both become Python names); rename one", name, name);
                break;
            }
            case NODE_EINSUM:
                check_ref_vars(res, &seen, &shadowed, program, &s->data.einsum.output);
                check_ref_vars(res, &seen, &shadowed, program, &s->data.einsum.input1);
                check_ref_vars(res, &seen, &shadowed, program, &s->data.einsum.input2);
                break;
            case NODE_EINSUM_FUSED:
                check_ref_vars(res, &seen, &shadowed, program, &s->data.einsum_fused.output);
                check_ref_vars(res, &seen, &shadowed, program, &s->data.einsum_fused.inner1);
                check_ref_vars(res, &seen, &shadowed, program, &s->data.einsum_fused.inner2);
                check_ref_vars(res, &seen, &shadowed, program, &s->data.einsum_fused.outer);
                break;
            default:
                break;
        }
    }
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

    /* seed(source=I, value=V) inits have no literal data; validate and skip
       shape checking. The seed materializes a single scalar at index I. */
    if (d->is_seed) {
        /* Count spatial (non-generational) ranks: a generational seed tensor
           like D[gen I, R=N] is data-1-D even though num_ranks == 2. */
        int spatial = te->num_ranks - (te->gen_axis >= 0 ? 1 : 0);
        if (spatial != 1)
            add_error(res, "seed: tensor '%s' must be 1-D", d->tensor_name);
        if (d->seed_source < 0)
            add_error(res, "seed: source index must be non-negative");
        return;
    }

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

/* Collect a ref's spatial index names (skipping its gen axis); returns count. */
static int ref_spatial_indices(TensorRef *r, TensorEntry *te, char *names[], int *sizes) {
    int n = 0;
    for (int a = 0; a < r->num_vars && a < te->num_ranks; a++) {
        if (a == te->gen_axis) continue;
        names[n] = r->rank_vars[a];
        sizes[n] = te->dim_sizes[a];
        n++;
    }
    return n;
}
static int name_present(char *const names[], int n, const char *x) {
    for (int i = 0; i < n; i++) if (strcmp(names[i], x) == 0) return 1;
    return 0;
}

static void analyze_einsum_fused(SemanticResult *res, AstNode *node, int eidx) {
    SymbolTable *st = res->symtab;
    EinsumFusedData *f = &node->data.einsum_fused;
    TensorEntry *out_te = symtab_find_tensor(st, f->output.tensor_name);
    TensorEntry *i1_te  = symtab_find_tensor(st, f->inner1.tensor_name);
    TensorEntry *i2_te  = symtab_find_tensor(st, f->inner2.tensor_name);
    TensorEntry *ou_te  = symtab_find_tensor(st, f->outer.tensor_name);
    if (!out_te || !i1_te || !i2_te || !ou_te) { add_error(res, "fused einsum: unknown tensor"); return; }

    /* validate each ref's index count matches its tensor's declared rank count
       (mirror the single-action arity check) */
    {
        TensorRef *frefs[4] = { &f->output, &f->inner1, &f->inner2, &f->outer };
        TensorEntry *ftes[4] = { out_te, i1_te, i2_te, ou_te };
        for (int k = 0; k < 4; k++) {
            if (frefs[k]->num_vars != ftes[k]->num_ranks)
                add_error(res,
                          "fused einsum: tensor '%s' has %d ranks but %d index vars given",
                          ftes[k]->name, ftes[k]->num_ranks, frefs[k]->num_vars);
        }
    }

    /* generational metadata (mirror analyze_einsum over the four fused refs) */
    {
        TensorEntry *tes[4]  = { out_te, i1_te, i2_te, ou_te };
        TensorRef   *refs[4] = { &f->output, &f->inner1, &f->inner2, &f->outer };
        for (int k = 0; k < 4; k++) {
            if (tes[k]->gen_axis >= 0 && tes[k]->gen_axis < refs[k]->num_vars) {
                st->is_iterative = 1;
                char *v = refs[k]->rank_vars[tes[k]->gen_axis];
                if (st->gen_rank_var == NULL) st->gen_rank_var = v;
                else if (strcmp(st->gen_rank_var, v) != 0)
                    add_error(res, "generational rank variable mismatch: '%s' vs '%s'", st->gen_rank_var, v);
            }
        }
        if (out_te->gen_axis >= 0) {
            if (f->output.gen_axis >= 0 && f->output.gen_axis != out_te->gen_axis) {
                add_error(res, "generational tensor '%s' written with an offset on the wrong axis", out_te->name);
            } else {
                int eff_off = (f->output.gen_axis == out_te->gen_axis) ? f->output.gen_offset : 0;
                if (eff_off == 1) {
                    int seen = 0;
                    for (int m = 0; m < st->num_gen_tensors; m++)
                        if (strcmp(st->gen_tensors[m], out_te->name) == 0) seen = 1;
                    if (!seen && st->num_gen_tensors < MAX_TENSORS) {
                        st->gen_tensors[st->num_gen_tensors++] = out_te->name;
                        if (st->gen_tensor == NULL) st->gen_tensor = out_te->name;
                    }
                } else if (eff_off == 0) {
                    int seen = 0;
                    for (int m = 0; m < st->num_intermediates; m++)
                        if (strcmp(st->intermediates[m], out_te->name) == 0) seen = 1;
                    if (!seen && st->num_intermediates < MAX_TENSORS)
                        st->intermediates[st->num_intermediates++] = out_te->name;
                } else {
                    add_error(res, "generational tensor '%s' written at unsupported offset +%d (only i and i+1 are allowed)", out_te->name, eff_off);
                }
            }
        }
    }

    char *iname1[MAX_INDICES]; int isz1[MAX_INDICES]; int n1 = ref_spatial_indices(&f->inner1, i1_te, iname1, isz1);
    char *iname2[MAX_INDICES]; int isz2[MAX_INDICES]; int n2 = ref_spatial_indices(&f->inner2, i2_te, iname2, isz2);
    char *onm[MAX_INDICES];    int osz[MAX_INDICES];  int no = ref_spatial_indices(&f->outer,  ou_te, onm,    osz);
    char *znm[MAX_INDICES];    int zsz[MAX_INDICES];  int nz = ref_spatial_indices(&f->output, out_te, znm,    zsz);

    /* a rank name shared across operands must resolve to a single size */
    {
        char *seen_n[4 * MAX_INDICES]; int seen_s[4 * MAX_INDICES]; int seen_c = 0;
        char *all_n[4 * MAX_INDICES]; int all_s[4 * MAX_INDICES]; int all_c = 0;
        for (int i = 0; i < n1; i++) { all_n[all_c]=iname1[i]; all_s[all_c]=isz1[i]; all_c++; }
        for (int i = 0; i < n2; i++) { all_n[all_c]=iname2[i]; all_s[all_c]=isz2[i]; all_c++; }
        for (int i = 0; i < no; i++) { all_n[all_c]=onm[i];    all_s[all_c]=osz[i];  all_c++; }
        for (int i = 0; i < nz; i++) { all_n[all_c]=znm[i];    all_s[all_c]=zsz[i];  all_c++; }
        for (int i = 0; i < all_c; i++) {
            int found = -1;
            for (int j = 0; j < seen_c; j++) if (strcmp(seen_n[j], all_n[i])==0) found = j;
            if (found < 0) { seen_n[seen_c]=all_n[i]; seen_s[seen_c]=all_s[i]; seen_c++; }
            else if (seen_s[found] != all_s[i])
                add_error(res, "fused einsum: index '%s' bound to size %d and size %d", all_n[i], seen_s[found], all_s[i]);
        }
    }

    char *inner[2 * MAX_INDICES]; int innersz[2 * MAX_INDICES]; int ni = 0;
    for (int i = 0; i < n1; i++) if (!name_present(inner, ni, iname1[i])) { inner[ni]=iname1[i]; innersz[ni]=isz1[i]; ni++; }
    for (int i = 0; i < n2; i++) if (!name_present(inner, ni, iname2[i])) { inner[ni]=iname2[i]; innersz[ni]=isz2[i]; ni++; }

    EinsumClass *ec = &st->einsum_class[eidx];
    memset(ec, 0, sizeof(*ec));
    ec->is_fused = 1;
    ec->inter_is_left2 = f->inter_is_left2;

    for (int i = 0; i < nz; i++) {
        ec->free_indices[ec->num_free_indices].name = znm[i];
        ec->free_indices[ec->num_free_indices].size = zsz[i];
        ec->free_indices[ec->num_free_indices].is_free = 1;
        ec->num_free_indices++;
    }
    for (int i = 0; i < ni; i++) {
        int downstream = name_present(onm, no, inner[i]) || name_present(znm, nz, inner[i]);
        if (!downstream) {
            ec->a1_reduce[ec->num_a1_reduce].name = inner[i];
            ec->a1_reduce[ec->num_a1_reduce].size = innersz[i];
            ec->num_a1_reduce++;
        }
    }
    char *inter[2 * MAX_INDICES]; int intersz[2 * MAX_INDICES]; int nint = 0;
    for (int i = 0; i < ni; i++) {
        int reduced = 0;
        for (int j = 0; j < ec->num_a1_reduce; j++) if (strcmp(ec->a1_reduce[j].name, inner[i])==0) reduced = 1;
        if (!reduced) { inter[nint]=inner[i]; intersz[nint]=innersz[i]; nint++; }
    }
    char *a2in[2 * MAX_INDICES]; int a2sz[2 * MAX_INDICES]; int na2 = 0;
    for (int i = 0; i < nint; i++) if (!name_present(a2in, na2, inter[i])) { a2in[na2]=inter[i]; a2sz[na2]=intersz[i]; na2++; }
    for (int i = 0; i < no;   i++) if (!name_present(a2in, na2, onm[i]))   { a2in[na2]=onm[i];   a2sz[na2]=osz[i];   na2++; }
    for (int i = 0; i < na2; i++) {
        if (!name_present(znm, nz, a2in[i])) {
            ec->a2_reduce[ec->num_a2_reduce].name = a2in[i];
            ec->a2_reduce[ec->num_a2_reduce].size = a2sz[i];
            ec->num_a2_reduce++;
        }
    }

    if (ec->num_a1_reduce > 0 && !f->a1_has_reduce)
        add_error(res, "fused einsum: action 1 reduces an index but has no reduce.1");
    if (ec->num_a1_reduce == 0 && f->a1_has_reduce)
        add_error(res, "fused einsum: reduce.1 given but action 1 reduces nothing");
    if (ec->num_a2_reduce > 0 && !f->a2_has_reduce)
        add_error(res, "fused einsum: action 2 reduces an index but has no reduce.2");
    if (ec->num_a2_reduce == 0 && f->a2_has_reduce)
        add_error(res, "fused einsum: reduce.2 given but action 2 reduces nothing");
    if (ec->num_a1_reduce > 0 && ec->num_a2_reduce > 0)
        add_error(res, "fused einsum: reductions on both actions (matrix-chain) are not yet supported");
    if (f->over1) {
        int ok = 0; for (int j=0;j<ec->num_a1_reduce;j++) if (strcmp(ec->a1_reduce[j].name,f->over1)==0) ok=1;
        if (!ok) add_error(res, "fused einsum: reduce.1 over= names a rank not reduced by action 1");
    }
    if (f->over2) {
        int ok = 0; for (int j=0;j<ec->num_a2_reduce;j++) if (strcmp(ec->a2_reduce[j].name,f->over2)==0) ok=1;
        if (!ok) add_error(res, "fused einsum: reduce.2 over= names a rank not reduced by action 2");
    }
    if (f->a1_has_reduce && f->reduce1_op == OP_TAKE_LEFT) add_error(res, "fused einsum: take_left cannot be a reduce operator");
    if (f->a2_has_reduce && f->reduce2_op == OP_TAKE_LEFT) add_error(res, "fused einsum: take_left cannot be a reduce operator");

    if (eidx + 1 > st->num_einsums) st->num_einsums = eidx + 1;
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

    /* Single-action reduce over= must name one of the reduction indices. */
    if (d->reduce_over != NULL) {
        int ok = 0;
        for (int i = 0; i < st->num_reduction_indices; i++)
            if (strcmp(st->reduction_indices[i].name, d->reduce_over) == 0) ok = 1;
        if (!ok)
            add_error(res, "reduce over= names a rank that is not reduced");
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
    StopData *sd = &node->data.stop;
    const char *watch = sd->kind == STOP_STABLE ? sd->stable_tensor : sd->lhs.tensor_name;
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
        TensorRef *ins[3]; int nins = 0; const char *onm = NULL;
        if (stmt->type == NODE_EINSUM) {
            ins[0] = &stmt->data.einsum.input1; ins[1] = &stmt->data.einsum.input2; nins = 2;
            onm = stmt->data.einsum.output.tensor_name;
        } else if (stmt->type == NODE_EINSUM_FUSED) {
            ins[0] = &stmt->data.einsum_fused.inner1;
            ins[1] = &stmt->data.einsum_fused.inner2;
            ins[2] = &stmt->data.einsum_fused.outer;
            nins = 3;
            onm = stmt->data.einsum_fused.output.tensor_name;
        } else continue;

        for (int k = 0; k < nins; k++) {
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

    validate_identifiers(&res, program);

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
            case NODE_EINSUM_FUSED:
                analyze_einsum_fused(&res, stmt, einsum_index++);
                break;
            case NODE_STOP:
                analyze_stop(&res, stmt);
                break;
            case NODE_SAVE:
                /* Validated in a dedicated pass below (needs all tensors). */
                break;
            default:
                add_error(&res, "unknown statement type %d", stmt->type);
                break;
        }
    }

    if (einsum_index == 0)
        add_error(&res, "program has no Einsum statement (nothing to compute)");

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
        for (int i = 0; i < pd->count && !read_at_i; i++) {
            AstNode *s = pd->statements[i];
            TensorRef *ins[3]; int nins = 0;
            if (s->type == NODE_EINSUM) {
                ins[0] = &s->data.einsum.input1; ins[1] = &s->data.einsum.input2; nins = 2;
            } else if (s->type == NODE_EINSUM_FUSED) {
                ins[0] = &s->data.einsum_fused.inner1;
                ins[1] = &s->data.einsum_fused.inner2;
                ins[2] = &s->data.einsum_fused.outer;
                nins = 3;
            } else continue;
            for (int k = 0; k < nins; k++) {
                TensorEntry *t = symtab_find_tensor(st, ins[k]->tensor_name);
                int eff = (t && ins[k]->gen_axis == t->gen_axis) ? ins[k]->gen_offset : 0;
                if (strcmp(ins[k]->tensor_name, gt) == 0 && eff == 0) { read_at_i = 1; break; }
            }
        }
        if (!read_at_i) {
            add_error(&res,
                "cross-generational tensor '%s' must be read at generation i "
                "somewhere in the cascade", gt);
        }
    }

    /* Save directive: target must be a tensor that some Einsum writes (an
       output), so that run() defines it. At most one save directive. */
    int num_saves = 0;
    for (int i = 0; i < pd->count; i++) {
        AstNode *s = pd->statements[i];
        if (s->type != NODE_SAVE) continue;
        if (++num_saves > 1) {
            add_error(&res, "save: only one save directive is supported");
            break;
        }
        const char *tgt = s->data.save.tensor_name;
        if (!symtab_find_tensor(st, tgt)) {
            add_error(&res, "save: unknown tensor '%s'", tgt);
            continue;
        }
        int is_output = 0;
        for (int j = 0; j < pd->count; j++) {
            AstNode *st_j = pd->statements[j];
            if (st_j->type == NODE_EINSUM &&
                strcmp(st_j->data.einsum.output.tensor_name, tgt) == 0) {
                is_output = 1; break;
            }
        }
        if (!is_output) {
            add_error(&res, "save: tensor '%s' is not a computed output", tgt);
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
