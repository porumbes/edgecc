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

/* Record a warning (kept for tests) and print it to stderr. */
static void add_warning(SemanticResult *res, const char *fmt, ...) {
    if (res->warning_count >= res->warning_capacity) {
        int new_cap = res->warning_capacity == 0 ? 4 : res->warning_capacity * 2;
        res->warnings = realloc(res->warnings, sizeof(char *) * new_cap);
        res->warning_capacity = new_cap;
    }
    va_list args;
    va_start(args, fmt);
    va_list args2;
    va_copy(args2, args);
    int len = vsnprintf(NULL, 0, fmt, args2);
    va_end(args2);
    char *buf = malloc(len + 1);
    vsnprintf(buf, len + 1, fmt, args);
    va_end(args);
    res->warnings[res->warning_count++] = buf;
    fprintf(stderr, "warning: %s\n", buf);
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
    "ImportError", "OSError", "RuntimeError", "SystemExit", "ValueError", "any", "float",
    "hasattr", "int", "isinstance", "len", "list", "max", "min", "open",
    "print", "range", "str", "sum", "zip",
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

static void check_reserved(SemanticResult *res, ReportedNames *seen,
                           const char *what, const char *name) {
    if (is_py_reserved(name) && first_report(seen, name))
        add_error(res, "%s '%s' is reserved in the generated Python "
                  "(keyword, builtin, or compiler temporary); rename it", what, name);
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
        check_reserved(res, seen, "rank variable", v);
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
                check_reserved(res, &seen, "dim", s->data.dim_decl.name);
                break;
            case NODE_TENSOR_DECL: {
                const char *name = s->data.tensor_decl.name;
                check_reserved(res, &seen, "tensor", name);
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
    if (!d->is_inferred && d->size <= 0)
        add_error(res, "dim '%s' must be positive, got %d", d->name, d->size);
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
            if (i != 0)
                add_error(res, "generational rank '%s' must be the first rank of '%s'",
                          d->ranks[i].rank_name, d->name);
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

/* 1 if `path` ends with `ext` (case-sensitive). */
static int path_has_ext(const char *path, const char *ext) {
    size_t lp = strlen(path), le = strlen(ext);
    return lp >= le && strcmp(path + lp - le, ext) == 0;
}

static void analyze_init(SemanticResult *res, AstNode *node) {
    InitData *d = &node->data.init;
    SymbolTable *st = res->symtab;

    TensorEntry *te = symtab_find_tensor(st, d->tensor_name);
    if (te == NULL) {
        add_error(res, "init references undeclared tensor '%s'", d->tensor_name);
        return;
    }
    if (te->has_init) {
        add_error(res, "duplicate init for tensor '%s'", d->tensor_name);
        return;
    }
    te->has_init = 1;
    te->loaded_from_file = d->is_load;

    /* load("...") inits have no literal data; skip shape checking. The format
       is chosen by extension. Matrix Market is 2-D and is densified as-is, so
       an .mtx target must be 2-D (the generational axis, if any, is not a data
       dimension); .npy carries its own shape and any rank. */
    if (d->is_load) {
        int is_mtx = path_has_ext(d->path, ".mtx");
        if (!is_mtx && !path_has_ext(d->path, ".npy")) {
            add_error(res, "load: unsupported file extension in \"%s\" (supported: .mtx, .npy)",
                      d->path);
            return;
        }
        int spatial = te->num_ranks - (te->gen_axis >= 0 ? 1 : 0);
        if (is_mtx && spatial != 2)
            add_error(res, "load: tensor '%s' must be 2-D (Matrix Market is 2-D), got %d-D",
                      d->tensor_name, spatial);
        return;
    }

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
                } else if (eff_off > 1) {
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
                } else if (eff_off > 1) {
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

/* Validate SSA ordering within a generation:
   (1) an intra-generation intermediate must be written by an earlier Einsum
       before a later Einsum reads it;
   (2) a cross-generational tensor read at i+1 must already be written at i+1 by
       an earlier Einsum (next-reads-next). */
/* Collect an Einsum statement's operand refs and output ref; 0 if not an Einsum. */
static int einsum_operands(AstNode *stmt, TensorRef *ins[3], TensorRef **out) {
    if (stmt->type == NODE_EINSUM) {
        ins[0] = &stmt->data.einsum.input1; ins[1] = &stmt->data.einsum.input2;
        *out = &stmt->data.einsum.output;
        return 2;
    }
    if (stmt->type == NODE_EINSUM_FUSED) {
        ins[0] = &stmt->data.einsum_fused.inner1;
        ins[1] = &stmt->data.einsum_fused.inner2;
        ins[2] = &stmt->data.einsum_fused.outer;
        *out = &stmt->data.einsum_fused.output;
        return 3;
    }
    return 0;
}

/* A `var+N` offset is only meaningful on a tensor's generational rank, and N
   must be non-negative (i or i+1; larger offsets are rejected per Einsum).
   Offsets anywhere else used to be silently ignored. A generational output
   written with an offset on the wrong axis is reported by the per-Einsum check. */
static void validate_rank_offsets(SemanticResult *res, AstNode *program) {
    ProgramData *pd = &program->data.program;
    for (int i = 0; i < pd->count; i++) {
        TensorRef *refs[4]; TensorRef *out;
        int n = einsum_operands(pd->statements[i], refs, &out);
        if (n == 0) continue;
        refs[n] = out;
        for (int k = 0; k <= n; k++) {
            TensorRef *r = refs[k];
            if (r->gen_axis < 0) continue;
            TensorEntry *te = symtab_find_tensor(res->symtab, r->tensor_name);
            if (te == NULL) continue;
            if (r->gen_offset < 0) {
                add_error(res, "negative rank offset on '%s' (only i and i+1 are allowed)",
                          r->tensor_name);
            } else if (r->gen_axis != te->gen_axis && !(r == out && te->gen_axis >= 0)) {
                add_error(res, "rank offset on '%s' is only allowed on its generational rank",
                          r->tensor_name);
            }
        }
    }
}

/* A tensor that an Einsum reads but that has no `init` and is never written
   becomes a run() input that __main__ fills with its empty value. That is
   legal (a driver may supply it), but is usually a typo or a missing init. */
static void warn_uninitialized_reads(SemanticResult *res, AstNode *program) {
    SymbolTable *st = res->symtab;
    ProgramData *pd = &program->data.program;
    const char *warned[MAX_TENSORS]; int nwarned = 0;
    for (int i = 0; i < pd->count; i++) {
        TensorRef *ins[3]; TensorRef *out;
        int nins = einsum_operands(pd->statements[i], ins, &out);
        for (int k = 0; k < nins; k++) {
            const char *nm = ins[k]->tensor_name;
            TensorEntry *te = symtab_find_tensor(st, nm);
            if (te == NULL || te->has_init) continue;
            int written = 0;
            for (int j = 0; j < pd->count && !written; j++) {
                TensorRef *unused[3]; TensorRef *w;
                if (einsum_operands(pd->statements[j], unused, &w) && strcmp(w->tensor_name, nm) == 0)
                    written = 1;
            }
            if (written || name_present((char *const *)warned, nwarned, nm)) continue;
            if (nwarned < MAX_TENSORS) warned[nwarned++] = nm;
            add_warning(res, "tensor '%s' is read but never initialized or written; "
                             "__main__ will pass it as empty", nm);
        }
    }
}

/* Output tensors start empty (paper, §Initialization): a tensor written by an
   Einsum is not an input. For non-generational tensors:
     E1  it may not have an init;
     E2  it may not be read before the first Einsum that writes it;
     E4  an Einsum may not read its own output (the loop would overwrite
         elements it still reads; an in-place update needs a gen rank).
   E3: in a generational program every Einsum writes a generational tensor
   (hoisting loop-invariant Einsums out of the loop is future work).
   Each error is reported once per tensor. */
static int written_by_some_einsum(ProgramData *pd, const char *name) {
    for (int j = 0; j < pd->count; j++) {
        TensorRef *unused[3]; TensorRef *w;
        if (einsum_operands(pd->statements[j], unused, &w) &&
            strcmp(w->tensor_name, name) == 0)
            return 1;
    }
    return 0;
}

static void validate_written_tensors(SemanticResult *res, AstNode *program) {
    SymbolTable *st = res->symtab;
    ProgramData *pd = &program->data.program;
    char *seen[MAX_EINSUMS];   int nseen = 0;       /* written so far, source order */
    char *e2_done[MAX_TENSORS]; int ne2 = 0;        /* E2 reported once per tensor */
    char *e3_done[MAX_TENSORS]; int ne3 = 0;        /* E3 reported once per tensor */
    char *e4_done[MAX_TENSORS]; int ne4 = 0;        /* E4 reported once per tensor */
    for (int i = 0; i < pd->count; i++) {
        TensorRef *ins[3]; TensorRef *out;
        int nins = einsum_operands(pd->statements[i], ins, &out);
        if (nins == 0) continue;

        if (!st->is_iterative) {
            for (int k = 0; k < nins; k++) {
                char *nm = ins[k]->tensor_name;
                TensorEntry *te = symtab_find_tensor(st, nm);
                if (te == NULL || te->gen_axis >= 0) continue;
                if (strcmp(nm, out->tensor_name) == 0) {   /* E4 (supersedes E2) */
                    if (name_present(e4_done, ne4, nm)) continue;
                    if (ne4 < MAX_TENSORS) e4_done[ne4++] = nm;
                    add_error(res, "tensor '%s' is read and written by the same Einsum "
                                   "(use a generational rank)", nm);
                    continue;
                }
                if (name_present(seen, nseen, nm) || name_present(e2_done, ne2, nm)) continue;
                if (!written_by_some_einsum(pd, nm)) continue;
                if (ne2 < MAX_TENSORS) e2_done[ne2++] = nm;
                add_error(res, "tensor '%s' is read before the Einsum that writes it", nm);
            }
        }

        TensorEntry *ote = symtab_find_tensor(st, out->tensor_name);
        if (ote == NULL || ote->gen_axis >= 0) continue;
        if (st->is_iterative) {
            if (name_present(e3_done, ne3, out->tensor_name)) continue;
            if (ne3 < MAX_TENSORS) e3_done[ne3++] = out->tensor_name;
            add_error(res, "in a generational program every Einsum must write a "
                           "generational tensor; '%s' is not (compute it in a "
                           "separate program)", out->tensor_name);
            continue;
        }
        if (name_present(seen, nseen, out->tensor_name)) continue;
        if (ote->has_init)
            add_error(res, "tensor '%s' is written by an Einsum and must not have an "
                           "init (output tensors start empty)", out->tensor_name);
        if (nseen < MAX_EINSUMS) seen[nseen++] = out->tensor_name;
    }
}

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

static int name_in(char *const *list, int n, const char *name) {
    for (int i = 0; i < n; i++)
        if (strcmp(list[i], name) == 0) return 1;
    return 0;
}

/* A generational tensor is written by some Einsum iff it is a
   cross-generational tensor (written at i+1) or an intermediate (at i). */
static int stop_tensor_written(const SymbolTable *st, const char *name) {
    return name_in(st->gen_tensors, st->num_gen_tensors, name) ||
           name_in(st->intermediates, st->num_intermediates, name);
}

/* Resolve one predicate reference X[g]. X must be a declared generational
   tensor, the reference must name a whole generation using the program's
   generational rank variable, and g must be i or i+1 (only i for an
   intermediate, which has no generation i+1 when the predicate runs).
   Returns the offset (0 or 1), or -1 after reporting an error. */
static int resolve_stop_ref(SemanticResult *res, const TensorRef *r, TensorEntry **te_out) {
    SymbolTable *st = res->symtab;
    TensorEntry *te = symtab_find_tensor(st, r->tensor_name);
    *te_out = te;
    if (!te) {
        add_error(res, "stop predicate: undeclared tensor '%s'", r->tensor_name);
        return -1;
    }
    if (te->gen_axis < 0) {
        add_error(res, "stop predicate: tensor '%s' is not generational", r->tensor_name);
        return -1;
    }
    if (!stop_tensor_written(st, te->name)) {
        add_error(res, "stop predicate: generational tensor '%s' is not written by any Einsum",
                  r->tensor_name);
        return -1;
    }
    if (r->num_vars != 1) {
        add_error(res, "stop predicate: '%s' must name a whole generation, e.g. %s[%s+1]",
                  r->tensor_name, r->tensor_name, st->gen_rank_var ? st->gen_rank_var : "i");
        return -1;
    }
    const char *v = r->rank_vars[0];
    if (st->gen_rank_var && strcmp(v, st->gen_rank_var) != 0) {
        add_error(res, "stop predicate uses '%s', but the generational rank variable is '%s'",
                  v, st->gen_rank_var);
        return -1;
    }
    int off = r->gen_offset;
    if (off != 0 && off != 1) {
        add_error(res, "stop predicate: generation must be '%s' or '%s+1'", v, v);
        return -1;
    }
    if (off == 1 && name_in(st->intermediates, st->num_intermediates, te->name)) {
        add_error(res, "intermediate '%s' has only generation '%s' when the stop predicate is checked",
                  te->name, v);
        return -1;
    }
    return off;
}

/* Find the (single) stop clause, check it against the
   program's iterativeness, and resolve it into symtab->stop_*. */
static void resolve_stop(SemanticResult *res, AstNode *program) {
    SymbolTable *st = res->symtab;
    ProgramData *pd = &program->data.program;
    AstNode *stop = NULL;
    int num_stops = 0;
    for (int i = 0; i < pd->count; i++)
        if (pd->statements[i]->type == NODE_STOP) { stop = pd->statements[i]; num_stops++; }
    if (num_stops > 1) {
        add_error(res, "only one 'stop:' clause is allowed");
        return;
    }

    if (!stop) {
        if (st->is_iterative)
            add_error(res, "generational einsum on '%s' requires a 'stop:' clause",
                      st->gen_tensor ? st->gen_tensor : "?");
        return;
    }
    if (!st->is_iterative) {
        add_error(res, "'stop' clause present but no generational einsum");
        return;
    }

    StopData *sd = &stop->data.stop;
    TensorEntry *ta = NULL, *tb = NULL;
    switch (sd->kind) {
        case STOP_STABLE: {
            /* stable(X) is sugar for X[i+1] == X[i]. */
            ta = symtab_find_tensor(st, sd->stable_tensor);
            if (!ta) {
                add_error(res, "stop predicate: undeclared tensor '%s'", sd->stable_tensor);
                return;
            }
            if (ta->gen_axis < 0) {
                add_error(res, "stop predicate: tensor '%s' is not generational", sd->stable_tensor);
                return;
            }
            if (!stop_tensor_written(st, ta->name)) {
                add_error(res, "stop predicate: generational tensor '%s' is not written by any Einsum",
                          sd->stable_tensor);
                return;
            }
            if (name_in(st->intermediates, st->num_intermediates, ta->name)) {
                add_error(res, "intermediate '%s' has only generation '%s' when the stop predicate is checked",
                          ta->name, st->gen_rank_var ? st->gen_rank_var : "i");
                return;
            }
            st->stop_kind = STOP_EQUAL;
            st->stop_gen_a = 1;
            st->stop_gen_b = 0;
            break;
        }
        case STOP_OCCUPANCY: {
            int a = resolve_stop_ref(res, &sd->lhs, &ta);
            if (sd->occupancy_rhs != 0) {
                add_error(res, "stop predicate: only '||X[g]|| == 0' is supported");
                return;
            }
            if (a < 0) return;
            st->stop_kind = STOP_OCCUPANCY;
            st->stop_gen_a = a;
            break;
        }
        case STOP_EQUAL: {
            int a = resolve_stop_ref(res, &sd->lhs, &ta);
            if (a < 0) return;
            int b = resolve_stop_ref(res, &sd->rhs, &tb);
            if (b < 0) return;
            if (ta != tb) {
                add_error(res, "stop predicate: '==' compares two generations of the same tensor "
                               "('%s' vs '%s')", ta->name, tb->name);
                return;
            }
            if (a == b) {
                add_error(res, "stop predicate compares a generation with itself");
                return;
            }
            st->stop_kind = STOP_EQUAL;
            st->stop_gen_a = a;
            st->stop_gen_b = b;
            break;
        }
    }
    st->stop_watch = ta->name;
}

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
            case NODE_EINSUM_FUSED:
                if (einsum_index >= MAX_EINSUMS) {
                    if (einsum_index == MAX_EINSUMS)
                        add_error(&res, "too many Einsums (max %d)", MAX_EINSUMS);
                    einsum_index++;
                    break;
                }
                if (stmt->type == NODE_EINSUM)
                    analyze_einsum(&res, stmt, einsum_index++);
                else
                    analyze_einsum_fused(&res, stmt, einsum_index++);
                break;
            case NODE_STOP:
                /* Resolved after all Einsums (needs the generational rank variable). */
                break;
            case NODE_SAVE:
                /* Validated in a dedicated pass below (needs all tensors). */
                break;
            case NODE_OUTPUT:
                /* Validated with the save directives after all Einsums. */
                break;
            default:
                add_error(&res, "unknown statement type %d", stmt->type);
                break;
        }
    }

    if (einsum_index == 0)
        add_error(&res, "program has no Einsum statement (nothing to compute)");
    /* The whole-program passes below keep per-Einsum buffers sized for at most
       MAX_EINSUMS statements; the "too many Einsums" error is already recorded. */
    if (einsum_index > MAX_EINSUMS) return res;

    validate_rank_offsets(&res, program);
    validate_written_tensors(&res, program);
    validate_cascade_ordering(&res, program);
    resolve_stop(&res, program);
    warn_uninitialized_reads(&res, program);

    /* Stop-clause presence and target checks live in resolve_stop above. */
    SymbolTable *st = res.symtab;
    /* A generational program must write at least one cross-generational tensor
       (a gen-rank tensor at offset i+1); otherwise nothing crosses generations. */
    if (st->is_iterative && st->num_gen_tensors == 0) {
        add_error(&res, "generational program has no cross-generational tensor "
                        "(write one at offset i+1, e.g. F[i+1, ...])");
    }
    if (st->num_gen_tensors > MAX_GEN_TENSORS)
        add_error(&res, "too many cross-generational tensors (max %d)", MAX_GEN_TENSORS);
    if (st->num_intermediates > MAX_GEN_TENSORS)
        add_error(&res, "too many intermediate tensors in a generational cascade (max %d)",
                  MAX_GEN_TENSORS);
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

    /* run()'s outputs: the `output:` list if present, else the default (the
       stop-watched tensor, or the output of the last Einsum). */
    AstNode *out_dir = NULL;
    for (int i = 0; i < pd->count; i++) {
        AstNode *s = pd->statements[i];
        if (s->type != NODE_OUTPUT) continue;
        if (out_dir) {
            add_error(&res, "output: only one output directive is allowed");
            continue;
        }
        out_dir = s;
    }
    if (out_dir) {
        for (int k = 0; k < out_dir->data.output.count; k++) {
            char *nm = out_dir->data.output.names[k];
            if (!symtab_find_tensor(st, nm))
                add_error(&res, "output: unknown tensor '%s'", nm);
            else if (!written_by_some_einsum(pd, nm))
                add_error(&res, "output: '%s' is not written by any Einsum "
                                "(inputs are not outputs)", nm);
            else if (name_present(st->outputs, st->num_outputs, nm))
                add_error(&res, "output: '%s' listed twice", nm);
            else if (st->num_outputs < MAX_TENSORS)
                st->outputs[st->num_outputs++] = nm;
        }
    } else {
        char *def = NULL;
        if (st->is_iterative) {
            def = st->stop_watch;
        } else {
            for (int j = 0; j < pd->count; j++) {
                AstNode *s = pd->statements[j];
                if (s->type == NODE_EINSUM)
                    def = s->data.einsum.output.tensor_name;
                else if (s->type == NODE_EINSUM_FUSED)
                    def = s->data.einsum_fused.output.tensor_name;
            }
        }
        if (def) st->outputs[st->num_outputs++] = def;
    }

    /* save: each target is one of run()'s outputs, saved once, to its own path. */
    for (int i = 0; i < pd->count; i++) {
        AstNode *s = pd->statements[i];
        if (s->type != NODE_SAVE) continue;
        char *tgt = s->data.save.tensor_name;
        char *path = s->data.save.path;
        TensorEntry *ste = symtab_find_tensor(st, tgt);
        if (!ste) {
            add_error(&res, "save: unknown tensor '%s'", tgt);
            continue;
        }
        int save_mtx = path_has_ext(path, ".mtx");
        if (!save_mtx && !path_has_ext(path, ".npy")) {
            add_error(&res, "save: unsupported file extension in \"%s\" (supported: .mtx, .npy)",
                      path);
            continue;
        }
        if (st->num_outputs == 0) continue;  /* earlier errors; don't pile on */
        if (!name_present(st->outputs, st->num_outputs, tgt)) {
            add_error(&res, "save: '%s' is not an output of run(); add it to the "
                            "output: list", tgt);
            continue;
        }
        if (name_present(st->save_tensors, st->num_saves, tgt)) {
            add_error(&res, "save: '%s' is saved twice", tgt);
            continue;
        }
        if (name_present(st->save_paths, st->num_saves, path)) {
            add_error(&res, "save: path \"%s\" is used twice", path);
            continue;
        }
        /* num_ranks counts the generations axis, which is stored. */
        if (save_mtx && ste->num_ranks > 2) {
            add_error(&res, "save: '%s' would be %d-D; Matrix Market stores at most 2-D "
                            "(save to a .npy path)", tgt, ste->num_ranks);
            continue;
        }
        if (st->num_saves < MAX_TENSORS) {
            st->save_tensors[st->num_saves] = tgt;
            st->save_paths[st->num_saves]   = path;
            st->num_saves++;
        }
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
    for (int i = 0; i < res->warning_count; i++) free(res->warnings[i]);
    free(res->warnings);
    res->warnings = NULL;
    res->warning_count = 0;
    res->warning_capacity = 0;
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
