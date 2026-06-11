#include "codegen.h"
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ---- Dynamic string buffer ---- */

typedef struct {
    char *buf;
    int len;
    int cap;
    int indent;
} CodeBuf;

static void cb_init(CodeBuf *cb) {
    cb->cap = 1024;
    cb->buf = malloc(cb->cap);
    cb->buf[0] = '\0';
    cb->len = 0;
    cb->indent = 0;
}

static void cb_grow(CodeBuf *cb, int need) {
    while (cb->len + need + 1 > cb->cap) {
        cb->cap *= 2;
    }
    cb->buf = realloc(cb->buf, cb->cap);
}

static void cb_append(CodeBuf *cb, const char *s) {
    int slen = (int)strlen(s);
    cb_grow(cb, slen);
    memcpy(cb->buf + cb->len, s, slen + 1);
    cb->len += slen;
}

static void cb_appendf(CodeBuf *cb, const char *fmt, ...) {
    char tmp[1024];
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(tmp, sizeof(tmp), fmt, ap);
    va_end(ap);
    cb_append(cb, tmp);
}

static void cb_indent(CodeBuf *cb) {
    for (int i = 0; i < cb->indent; i++) {
        cb_append(cb, "    ");
    }
}

/* ---- Helpers ---- */

/* Find a DIM_DEF name whose size matches `range`. Search backwards from `pos`.
   If ambiguous (multiple matches), return NULL. */
static const char *find_dim_name(IRProgram *ir, int pos, int range) {
    const char *found = NULL;
    int count = 0;
    for (int i = pos - 1; i >= 0; i--) {
        if (ir->instrs[i].type == IR_DIM_DEF &&
            ir->instrs[i].data.dim_def.size == range) {
            if (!found) found = ir->instrs[i].data.dim_def.name;
            count++;
        }
    }
    return (count == 1) ? found : NULL;
}

/* Emit literal data recursively as a Python expression.
   Forward declaration of format_double_literal lives below near emit_alloc;
   emit_literal sees it via the forward decl in that block. */
static char *format_double_literal(double v);

/* Emit literal data recursively as a Python expression */
static void emit_literal(CodeBuf *cb, LiteralData *data) {
    if (!data) return;
    if (data->is_list) {
        cb_append(cb, "[");
        for (int i = 0; i < data->data.list.count; i++) {
            if (i > 0) cb_append(cb, ", ");
            emit_literal(cb, data->data.list.items[i]);
        }
        cb_append(cb, "]");
    } else {
        switch (data->data.scalar.type) {
            case TYPE_INTEGER:
                cb_appendf(cb, "%d", data->data.scalar.int_val);
                break;
            case TYPE_FLOAT: {
                char *s = format_double_literal(data->data.scalar.float_val);
                cb_append(cb, s);
                free(s);
                break;
            }
            case TYPE_BOOLEAN:
                cb_append(cb, data->data.scalar.bool_val ? "True" : "False");
                break;
        }
    }
}

/* Emit tensor[c1][c2]... */
static void emit_tensor_access(CodeBuf *cb, const char *name, char **coords, int ncoords) {
    cb_append(cb, name);
    for (int i = 0; i < ncoords; i++) {
        cb_append(cb, "[");
        cb_append(cb, coords[i]);
        cb_append(cb, "]");
    }
}

/* Forward declaration: emit_general_skeleton is defined below codegen_python.
   format_double_literal is forward-declared earlier (above emit_literal). */
static void emit_general_skeleton(CodeBuf *cb,
                                   IRInstr *load1, IRInstr *load2,
                                   IRInstr *map_i, IRInstr *red_i,
                                   IRInstr *store_i);

/* Emit a tensor alloc as nested list comprehension.
   e.g. for dims [2,3] with empty_val 0.0:
   [[0.0] * 3 for _ in range(2)]
   For 1D: [0.0] * N
*/
static void emit_alloc(CodeBuf *cb, IRTensorAlloc *alloc, IRProgram *ir, int instr_pos) {
    /* Format the empty value using format_double_literal */
    char *valstr = format_double_literal(alloc->empty_val);

    if (alloc->ndims == 0) {
        cb_append(cb, valstr);
        free(valstr);
        return;
    }

    /* Build dim name strings for each dimension */
    /* For the alloc, we need to find dim names that match each dim_size.
       We search ALL DIM_DEFs, not just unique ones. We try to match by
       looking at the order of DIM_DEFs. */

    /* Collect dim names in order from the IR */
    const char *dim_names[8] = {0};
    /* For each dimension of this tensor, try to find a matching DIM_DEF */
    /* Strategy: collect all DIM_DEFs in order, then for each alloc dim_size,
       find the DIM_DEF with matching size. If multiple match, pick the first
       unused one. */
    typedef struct { const char *name; int size; int used; } DimInfo;
    DimInfo all_dims[64];
    int ndims_found = 0;
    for (int i = 0; i < ir->count && ir->instrs[i].type == IR_DIM_DEF; i++) {
        all_dims[ndims_found].name = ir->instrs[i].data.dim_def.name;
        all_dims[ndims_found].size = ir->instrs[i].data.dim_def.size;
        all_dims[ndims_found].used = 0;
        ndims_found++;
    }

    for (int d = 0; d < alloc->ndims; d++) {
        dim_names[d] = alloc->dim_names[d];   /* prefer threaded name */
        if (dim_names[d]) continue;
        for (int j = 0; j < ndims_found; j++) {
            if (!all_dims[j].used && all_dims[j].size == alloc->dim_sizes[d]) {
                dim_names[d] = all_dims[j].name;
                all_dims[j].used = 1;
                break;
            }
        }
    }

    /* Now emit nested list comprehension from outermost to innermost */
    /* For [M, N]: [[0.0] * N for _ in range(M)] */
    /* For [M, N, P]: [[[0.0] * P for _ in range(N)] for _ in range(M)] */

    /* We emit: open brackets, innermost = [val] * last_dim, then for _ in range(...) for each outer dim */
    for (int d = 0; d < alloc->ndims - 1; d++) {
        cb_append(cb, "[");
    }
    cb_appendf(cb, "[%s] * ", valstr);
    if (dim_names[alloc->ndims - 1]) {
        cb_append(cb, dim_names[alloc->ndims - 1]);
    } else {
        cb_appendf(cb, "%d", alloc->dim_sizes[alloc->ndims - 1]);
    }
    for (int d = alloc->ndims - 2; d >= 0; d--) {
        cb_append(cb, " for _ in range(");
        if (dim_names[d]) {
            cb_append(cb, dim_names[d]);
        } else {
            cb_appendf(cb, "%d", alloc->dim_sizes[d]);
        }
        cb_append(cb, ")]");
    }

    free(valstr);
}

/* ---- New v2 helpers ---- */

/* Format a double for emission as a Python literal.
 * +inf → "math.inf", -inf → "-math.inf", integral values
 * keep one decimal (e.g. "0.0"), other values use %g.
 * Returns a malloc'd string; caller frees. */
static char *format_double_literal(double v) {
    char buf[64];
    if (v > 1e308) {           /* +inf */
        snprintf(buf, sizeof(buf), "math.inf");
    } else if (v < -1e308) {   /* -inf */
        snprintf(buf, sizeof(buf), "-math.inf");
    } else if (v == (int)v && v >= -1e9 && v <= 1e9) {
        snprintf(buf, sizeof(buf), "%.1f", v);
    } else {
        snprintf(buf, sizeof(buf), "%g", v);
    }
    return strdup(buf);
}

/* Emit the merge guard expression for a given merge op into cb.
 * `lhs_name` is the left existence variable name (e.g. "a_exists" or "b_s").
 * `rhs_name` is the right existence variable name (e.g. "b_exists" or "b_m"). */
static void emit_merge_guard(CodeBuf *cb, MergeOp m,
                              const char *lhs_name, const char *rhs_name) {
    switch (m) {
        case MERGE_PASSTHROUGH:
            cb_append(cb, "True");
            return;
        case MERGE_NOPASS:
            cb_append(cb, "False");
            return;
        case MERGE_INTERSECT:
            cb_appendf(cb, "%s and %s", lhs_name, rhs_name);
            return;
        case MERGE_TAKE_LEFT:
            cb_append(cb, lhs_name);
            return;
        case MERGE_TAKE_LEFT_ONLY:
            cb_appendf(cb, "%s and not %s", lhs_name, rhs_name);
            return;
        case MERGE_TAKE_RIGHT:
            cb_append(cb, rhs_name);
            return;
        case MERGE_TAKE_RIGHT_ONLY:
            cb_appendf(cb, "%s and not %s", rhs_name, lhs_name);
            return;
        case MERGE_XOR:
            cb_appendf(cb, "%s != %s", lhs_name, rhs_name);
            return;
        case MERGE_UNION:
            cb_appendf(cb, "%s or %s", lhs_name, rhs_name);
            return;
    }
}

typedef struct {
    char *names[8];
    int count;
} TensorNameList;

/* Collect distinct input tensor names (those appearing in IR_LOAD, in load order).
 * Output tensor name is the operand of IR_STORE. Caller-owned. */
static void collect_io(IRProgram *ir, TensorNameList *inputs, char **output) {
    inputs->count = 0;
    *output = NULL;
    for (int i = 0; i < ir->count; i++) {
        IRInstr *instr = &ir->instrs[i];
        if (instr->type == IR_LOAD) {
            const char *n = instr->data.load.tensor_name;
            int seen = 0;
            for (int j = 0; j < inputs->count; j++) {
                if (strcmp(inputs->names[j], n) == 0) { seen = 1; break; }
            }
            if (!seen && inputs->count < 8) {
                inputs->names[inputs->count++] = strdup(n);
            }
        } else if (instr->type == IR_STORE) {
            if (*output) free(*output);
            *output = strdup(instr->data.store.tensor_name);
        }
    }
}

static void free_tensor_name_list(TensorNameList *l) {
    for (int i = 0; i < l->count; i++) free(l->names[i]);
    l->count = 0;
}

/* Format an operator application as a Python expression into `out`.
 * Caller-owned buffer; capped at `cap`. Returns out for convenience. */
static const char *compute_op_pyexpr(OperatorType op,
                                     const char *lhs, const char *rhs,
                                     char *out, size_t cap) {
    switch (op) {
        case OP_ADD: snprintf(out, cap, "%s + %s", lhs, rhs); break;
        case OP_SUB: snprintf(out, cap, "%s - %s", lhs, rhs); break;
        case OP_MUL: snprintf(out, cap, "%s * %s", lhs, rhs); break;
        case OP_MIN: snprintf(out, cap, "min(%s, %s)", lhs, rhs); break;
        case OP_MAX: snprintf(out, cap, "max(%s, %s)", lhs, rhs); break;
        case OP_AND: snprintf(out, cap, "%s and %s", lhs, rhs); break;
        case OP_OR:  snprintf(out, cap, "%s or %s",  lhs, rhs); break;
        case OP_TAKE_LEFT: snprintf(out, cap, "%s", lhs); break;
    }
    return out;
}

/* Emit the general skeleton for one innermost iteration body. */
static void emit_general_skeleton(CodeBuf *cb,
                                   IRInstr *load1, IRInstr *load2,
                                   IRInstr *map_i, IRInstr *red_i,
                                   IRInstr *store_i) {
    char *eA = format_double_literal(map_i->data.map.empty_a);
    char *eB = format_double_literal(map_i->data.map.empty_b);
    char *eZ = format_double_literal(map_i->data.map.empty_z);

    /* Generational tensors are double-buffered: reads use <name>_prev,
       writes use <name>_next. Non-generational tensors keep their name. */
    char a_name[128], b_name[128], z_name[128];
    if (load1->data.load.gen_read == 2)
        snprintf(a_name, sizeof(a_name), "%s_next", load1->data.load.tensor_name);
    else if (load1->data.load.gen_read == 1)
        snprintf(a_name, sizeof(a_name), "%s_prev", load1->data.load.tensor_name);
    else
        snprintf(a_name, sizeof(a_name), "%s", load1->data.load.tensor_name);
    if (load2->data.load.gen_read == 2)
        snprintf(b_name, sizeof(b_name), "%s_next", load2->data.load.tensor_name);
    else if (load2->data.load.gen_read == 1)
        snprintf(b_name, sizeof(b_name), "%s_prev", load2->data.load.tensor_name);
    else
        snprintf(b_name, sizeof(b_name), "%s", load2->data.load.tensor_name);
    if (store_i->data.store.gen_write)
        snprintf(z_name, sizeof(z_name), "%s_next", store_i->data.store.tensor_name);
    else
        snprintf(z_name, sizeof(z_name), "%s", store_i->data.store.tensor_name);

    /* LOAD operands */
    cb_indent(cb); cb_append(cb, "a_val = ");
    emit_tensor_access(cb, a_name,
                       load1->data.load.coords, load1->data.load.ncoords);
    cb_append(cb, "\n");
    cb_indent(cb); cb_append(cb, "b_val = ");
    emit_tensor_access(cb, b_name,
                       load2->data.load.coords, load2->data.load.ncoords);
    cb_append(cb, "\n");

    /* MAP existence. A negated operand (`!X`) is present exactly where X is
       empty, so its existence test flips to `== empty`. */
    cb_indent(cb);
    cb_appendf(cb, "a_exists = a_val %s %s\n",
               load1->data.load.negate ? "==" : "!=", eA);
    cb_indent(cb);
    cb_appendf(cb, "b_exists = b_val %s %s\n",
               load2->data.load.negate ? "==" : "!=", eB);

    cb_indent(cb); cb_append(cb, "if ");
    emit_merge_guard(cb, map_i->data.map.merge_op, "a_exists", "b_exists");
    cb_append(cb, ":\n");
    cb->indent++;
    cb_indent(cb);
    cb_appendf(cb, "a_op = %s if a_exists else %s\n",
               load1->data.load.negate ? "1.0" : "a_val", eA);
    cb_indent(cb);
    cb_appendf(cb, "b_op = %s if b_exists else %s\n",
               load2->data.load.negate ? "1.0" : "b_val", eB);

    char compute_buf[256];
    cb_indent(cb); cb_append(cb, "map_val = ");
    cb_append(cb, compute_op_pyexpr(map_i->data.map.op, "a_op", "b_op",
                                     compute_buf, sizeof(compute_buf)));
    cb_append(cb, "\n");
    cb_indent(cb); cb_appendf(cb, "map_emitted = map_val != %s\n", eZ);
    cb->indent--;
    cb_indent(cb); cb_append(cb, "else:\n");
    cb->indent++;
    cb_indent(cb); cb_append(cb, "map_emitted = False\n");
    cb_indent(cb); cb_appendf(cb, "map_val = %s\n", eZ);
    cb->indent--;

    /* REDUCE existence */
    cb_indent(cb); cb_append(cb, "state = ");
    emit_tensor_access(cb, z_name, store_i->data.store.coords, store_i->data.store.ncoords);
    cb_append(cb, "\n");
    cb_indent(cb); cb_appendf(cb, "b_s = state != %s\n", eZ);
    cb_indent(cb); cb_append(cb, "b_m = map_emitted\n");

    cb_indent(cb); cb_append(cb, "if ");
    emit_merge_guard(cb, red_i->data.reduce.merge_op, "b_s", "b_m");
    cb_append(cb, ":\n");
    cb->indent++;
    cb_indent(cb); cb_appendf(cb, "val_for_red = map_val if map_emitted else %s\n", eZ);
    cb_indent(cb); cb_append(cb, "new_state = ");
    cb_append(cb, compute_op_pyexpr(red_i->data.reduce.op, "state", "val_for_red",
                                     compute_buf, sizeof(compute_buf)));
    cb_append(cb, "\n");
    cb_indent(cb);
    emit_tensor_access(cb, z_name, store_i->data.store.coords, store_i->data.store.ncoords);
    cb_appendf(cb, " = new_state if new_state != %s else %s\n", eZ, eZ);
    cb->indent--;
    cb_indent(cb); cb_append(cb, "else:\n");
    cb->indent++;
    cb_indent(cb);
    emit_tensor_access(cb, z_name, store_i->data.store.coords, store_i->data.store.ncoords);
    cb_appendf(cb, " = %s\n", eZ);
    cb->indent--;

    free(eA); free(eB); free(eZ);
}

/* ---- Main codegen ---- */

/* Return 1 if `name` matches any entry in `list[0..n)`. */
static int name_in_list(const char *name, const char **list, int n) {
    for (int i = 0; i < n; i++)
        if (strcmp(name, list[i]) == 0) return 1;
    return 0;
}

/* For an inferred dim `dimname`, find the first INPUT tensor alloc that has it
   on some axis and emit `dimname = len(input)` (axis 0) or `len(input[0])`. */
/* Runtime Python subscript depth of alloc dim `d`: a generational axis (dim_size
   == -1 sentinel) is not a materialized Python dimension, so skip it. Returns 0
   or 1 for edgecc's <=2 spatial dims. */
static int python_axis_of(const IRTensorAlloc *a, int d) {
    int axis = 0;
    for (int j = 0; j < d; j++)
        if (a->dim_sizes[j] != -1) axis++;
    return axis;
}

static void emit_inferred_dim_derivation(CodeBuf *cb, IRProgram *ir,
                                         const char *dimname,
                                         TensorNameList *inputs) {
    for (int i = 0; i < ir->count; i++) {
        if (ir->instrs[i].type != IR_TENSOR_ALLOC) continue;
        IRTensorAlloc *a = &ir->instrs[i].data.tensor_alloc;
        int is_input = 0;
        for (int j = 0; j < inputs->count; j++)
            if (strcmp(a->name, inputs->names[j]) == 0) { is_input = 1; break; }
        if (!is_input) continue;
        for (int d = 0; d < a->ndims; d++) {
            if (a->dim_names[d] && strcmp(a->dim_names[d], dimname) == 0) {
                cb_indent(cb);
                if (python_axis_of(a, d) == 0) cb_appendf(cb, "%s = len(%s)\n", dimname, a->name);
                else                           cb_appendf(cb, "%s = len(%s[0])\n", dimname, a->name);
                return;
            }
        }
    }
}

/* Materialize a seed(...) init in __main__: fill a length-N vector with the
   tensor's empty value, then set the source index to the seed value. N comes
   from the loaded tensor that pins the seed tensor's (single) dim, or from a
   fixed dim's constant size. */
static void emit_seed_init(CodeBuf *cb, IRProgram *ir, const char *name,
                           int source, double value) {
    double empty_val = 0.0;
    const char *dimname = NULL;
    for (int i = 0; i < ir->count; i++) {
        if (ir->instrs[i].type != IR_TENSOR_ALLOC) continue;
        if (strcmp(ir->instrs[i].data.tensor_alloc.name, name) != 0) continue;
        IRTensorAlloc *sa = &ir->instrs[i].data.tensor_alloc;
        empty_val = sa->empty_val;
        /* Pick the spatial (data) dim, skipping a generational axis (dim_size
           == -1 sentinel) so D[gen I, R=N] resolves to N, not the gen index. */
        for (int d = 0; d < sa->ndims; d++) {
            if (sa->dim_sizes[d] == -1) continue;
            dimname = sa->dim_names[d];
            break;
        }
        break;
    }

    int inferred = 0, dimsize = 0;
    for (int i = 0; i < ir->count; i++) {
        if (ir->instrs[i].type != IR_DIM_DEF) continue;
        if (dimname && strcmp(ir->instrs[i].data.dim_def.name, dimname) == 0) {
            inferred = ir->instrs[i].data.dim_def.is_inferred;
            dimsize = ir->instrs[i].data.dim_def.size;
            break;
        }
    }

    char lenexpr[128];
    snprintf(lenexpr, sizeof(lenexpr), "0");   /* unreachable: semantic.c guarantees
                                                  an inferred seed dim is pinned by a load */
    if (!inferred) {
        snprintf(lenexpr, sizeof(lenexpr), "%d", dimsize);
    } else {
        int found = 0;
        for (int i = 0; i < ir->count && !found; i++) {
            if (ir->instrs[i].type != IR_TENSOR_INIT) continue;
            if (!ir->instrs[i].data.tensor_init.is_load) continue;
            const char *pin = ir->instrs[i].data.tensor_init.name;
            for (int j = 0; j < ir->count && !found; j++) {
                if (ir->instrs[j].type != IR_TENSOR_ALLOC) continue;
                if (strcmp(ir->instrs[j].data.tensor_alloc.name, pin) != 0) continue;
                IRTensorAlloc *pa = &ir->instrs[j].data.tensor_alloc;
                for (int d = 0; d < pa->ndims; d++) {
                    if (pa->dim_names[d] && dimname &&
                        strcmp(pa->dim_names[d], dimname) == 0) {
                        if (python_axis_of(pa, d) == 0) snprintf(lenexpr, sizeof(lenexpr), "len(%s)", pin);
                        else                            snprintf(lenexpr, sizeof(lenexpr), "len(%s[0])", pin);
                        found = 1;
                        break;
                    }
                }
            }
        }
    }

    char *emptys = format_double_literal(empty_val);
    char *vals = format_double_literal(value);
    cb_indent(cb);
    cb_appendf(cb, "%s = [%s] * %s\n", name, emptys, lenexpr);
    cb_indent(cb);
    cb_appendf(cb, "if not (0 <= %d < len(%s)):\n", source, name);
    cb->indent++;
    cb_indent(cb);
    cb_appendf(cb,
        "raise ValueError(\"edgecc: seed source %d out of range [0, %%d) for '%s'\" %% len(%s))\n",
        source, name, name);
    cb->indent--;
    cb_indent(cb);
    cb_appendf(cb, "%s[%d] = %s\n", name, source, vals);
    free(emptys);
    free(vals);
}

/* After `dimname` is derived from its first input, assert every OTHER input
   that uses `dimname` agrees on the relevant axis length. */
static void emit_inferred_dim_asserts(CodeBuf *cb, IRProgram *ir,
                                      const char *dimname,
                                      TensorNameList *inputs) {
    const char *first = NULL;
    for (int i = 0; i < ir->count; i++) {
        if (ir->instrs[i].type != IR_TENSOR_ALLOC) continue;
        IRTensorAlloc *a = &ir->instrs[i].data.tensor_alloc;
        int is_input = 0;
        for (int j = 0; j < inputs->count; j++)
            if (strcmp(a->name, inputs->names[j]) == 0) { is_input = 1; break; }
        if (!is_input) continue;
        int found = 0, axis = 0;
        for (int d = 0; d < a->ndims; d++)
            if (a->dim_names[d] && strcmp(a->dim_names[d], dimname) == 0) {
                axis = python_axis_of(a, d); found = 1; break;
            }
        if (!found) continue;
        if (!first) { first = a->name; continue; }   /* the source; skip */
        const char *suffix = (axis == 0) ? "" : "[0]";
        cb_indent(cb);
        cb_appendf(cb, "if len(%s%s) != %s:\n", a->name, suffix, dimname);
        cb->indent++;
        cb_indent(cb);
        cb_appendf(cb,
            "raise ValueError(\"edgecc: input '%s' has size %%d on dim %s "
            "but '%s' pins it to %%d\" %% (len(%s%s), %s))\n",
            a->name, dimname, first, a->name, suffix, dimname);
        cb->indent--;
    }
}

char *codegen_python(IRProgram *ir) {
    return codegen_python_ex(ir, 0, NULL);
}

char *codegen_python_ex(IRProgram *ir, int emit_trace, const char *kernel_name) {
    if (!ir) return NULL;
    if (!kernel_name) kernel_name = "kernel";
    CodeBuf cb;
    cb_init(&cb);

    cb_append(&cb, "# Generated by edgecc\n");
    cb_append(&cb, "import math\n\n");

    /* Detect whether this program uses file I/O (load init or save). */
    int uses_fileio = 0;
    for (int i = 0; i < ir->count; i++) {
        if ((ir->instrs[i].type == IR_TENSOR_INIT && ir->instrs[i].data.tensor_init.is_load) ||
            ir->instrs[i].type == IR_SAVE) { uses_fileio = 1; break; }
    }
    if (uses_fileio) {
        cb_append(&cb,
            "def _edgecc_load(path):\n"
            "    if path.endswith(\".mtx\"):\n"
            "        try:\n"
            "            from scipy.io import mmread\n"
            "        except ImportError:\n"
            "            raise ImportError(\"edgecc load/save requires scipy: pip install scipy\")\n"
            "        m = mmread(path)\n"
            "        return (m.toarray() if hasattr(m, \"toarray\") else m).tolist()\n"
            "    raise ValueError(\"edgecc load: unsupported file extension for %r (supported: .mtx)\" % path)\n"
            "\n"
            "def _edgecc_save(tensor, path):\n"
            "    if path.endswith(\".mtx\"):\n"
            "        try:\n"
            "            from scipy.io import mmwrite\n"
            "            import numpy as np\n"
            "        except ImportError:\n"
            "            raise ImportError(\"edgecc load/save requires scipy: pip install scipy\")\n"
            "        a = np.array(tensor, dtype=float)\n"
            "        if a.ndim == 1:\n"
            "            a = a.reshape(-1, 1)   # Matrix Market needs 2-D; store a vector as a column\n"
            "        mmwrite(path, a)\n"
            "        return\n"
            "    raise ValueError(\"edgecc save: unsupported file extension for %r (supported: .mtx)\" % path)\n"
            "\n");
    }

    /* Module-level dim constants (skip inferred dims; derived in run()) */
    for (int i = 0; i < ir->count; i++) {
        if (ir->instrs[i].type == IR_DIM_DEF &&
            !ir->instrs[i].data.dim_def.is_inferred) {
            cb_appendf(&cb, "%s = %d\n",
                ir->instrs[i].data.dim_def.name,
                ir->instrs[i].data.dim_def.size);
        }
    }
    cb_append(&cb, "\n");

    /* Discover inputs and output */
    TensorNameList inputs;
    char *output = NULL;
    collect_io(ir, &inputs, &output);

    /* Detect generational mode and capture the gen-loop metadata. */
    int iterative = 0;
    const char *cross_gens[8]; int num_cross_gens = 0;
    double cross_gen_empty[8];
    const char *watch = NULL;
    const char *intermediates[8];
    double intermediate_empty[8];
    int num_intermediates = 0;
    for (int i = 0; i < ir->count; i++) {
        if (ir->instrs[i].type == IR_GEN_LOOP_BEGIN) {
            IRGenLoop *gl = &ir->instrs[i].data.gen_loop;
            iterative = 1;
            num_cross_gens = gl->num_cross_gens;
            for (int j = 0; j < num_cross_gens && j < 8; j++) {
                cross_gens[j] = gl->cross_gens[j].name;
                cross_gen_empty[j] = gl->cross_gens[j].empty_val;
            }
            watch = gl->watch;
            num_intermediates = gl->num_intermediates;
            for (int j = 0; j < num_intermediates && j < 8; j++) {
                intermediates[j] = gl->intermediates[j].name;
                intermediate_empty[j] = gl->intermediates[j].empty_val;
            }
            break;
        }
    }

    /* Trace capture is only meaningful for generational kernels. */
    if (emit_trace && !iterative) {
        fprintf(stderr,
            "edgecc: --emit-trace ignored: '%s' is not a generational kernel\n",
            kernel_name);
        emit_trace = 0;
    }

    /* Trace snapshot helper (deep-copies a tensor); only when capturing. */
    if (emit_trace) {
        cb_append(&cb,
            "def _edgecc_snapshot(x):\n"
            "    return [row[:] for row in x] if (x and isinstance(x[0], list)) else list(x)\n"
            "\n");
    }

    /* Cascade I/O: intermediates are neither parameters nor the return value;
       the returned tensor is the cross-generational one. */
    if (iterative) {
        TensorNameList filtered;
        filtered.count = 0;
        for (int i = 0; i < inputs.count; i++) {
            if (name_in_list(inputs.names[i], intermediates, num_intermediates)) {
                free(inputs.names[i]); continue;
            }
            filtered.names[filtered.count++] = inputs.names[i];
        }
        inputs = filtered;
        if (watch) {
            if (output) free(output);
            output = strdup(watch);
        }
    }

    if (iterative) {
        cb_append(&cb, "__EDGECC_MAX_GEN__ = 1000\n\n");
    }

    /* __edgecc_meta__ */
    cb_append(&cb, "__edgecc_meta__ = {\n");
    cb_append(&cb, "    \"inputs\": [");
    for (int i = 0; i < inputs.count; i++) {
        if (i > 0) cb_append(&cb, ", ");
        cb_appendf(&cb, "\"%s\"", inputs.names[i]);
    }
    cb_append(&cb, "],\n");
    cb_appendf(&cb, "    \"output\": \"%s\",\n", output ? output : "");
    cb_append(&cb, "    \"file_inputs\": [");
    {
        int first = 1;
        for (int i = 0; i < ir->count; i++) {
            if (ir->instrs[i].type == IR_TENSOR_INIT &&
                ir->instrs[i].data.tensor_init.is_load) {
                if (!first) cb_append(&cb, ", ");
                cb_appendf(&cb, "\"%s\"", ir->instrs[i].data.tensor_init.name);
                first = 0;
            }
        }
    }
    cb_append(&cb, "],\n");
    cb_append(&cb, "}\n\n");

    /* def run(inputs) */
    cb_append(&cb, "def run(");
    for (int i = 0; i < inputs.count; i++) {
        if (i > 0) cb_append(&cb, ", ");
        cb_append(&cb, inputs.names[i]);
    }
    if (emit_trace) cb_append(&cb, ", _trace=None");
    cb_append(&cb, "):\n");
    cb.indent = 1;

    /* Derive inferred dims from input shapes, then assert shared dims agree. */
    for (int i = 0; i < ir->count; i++) {
        if (ir->instrs[i].type == IR_DIM_DEF &&
            ir->instrs[i].data.dim_def.is_inferred) {
            emit_inferred_dim_derivation(&cb, ir, ir->instrs[i].data.dim_def.name, &inputs);
        }
    }
    for (int i = 0; i < ir->count; i++) {
        if (ir->instrs[i].type == IR_DIM_DEF &&
            ir->instrs[i].data.dim_def.is_inferred) {
            emit_inferred_dim_asserts(&cb, ir, ir->instrs[i].data.dim_def.name, &inputs);
        }
    }

    int produced_loops = 0;
    for (int i = 0; i < ir->count; i++) {
        IRInstr *instr = &ir->instrs[i];
        switch (instr->type) {
            case IR_DIM_DEF:
                /* already emitted at module scope */
                break;
            case IR_TENSOR_ALLOC:
                /* Output tensor is allocated inside run(); but a generational
                   tensor arrives as a parameter and is buffered by the gen loop. */
                if (output && strcmp(instr->data.tensor_alloc.name, output) == 0 &&
                    !(iterative &&
                      name_in_list(instr->data.tensor_alloc.name, cross_gens, num_cross_gens))) {
                    cb_indent(&cb);
                    cb_appendf(&cb, "%s = ", instr->data.tensor_alloc.name);
                    emit_alloc(&cb, &instr->data.tensor_alloc, ir, i);
                    cb_append(&cb, "\n");
                }
                break;
            case IR_TENSOR_INIT:
                /* Init statements lift into __main__; skip inside run(). */
                break;
            case IR_LOOP_BEGIN: {
                if (!produced_loops) {
                    cb_append(&cb, "\n");
                    produced_loops = 1;
                }
                cb_indent(&cb);
                const char *var = instr->data.loop_begin.index_var;
                int range = instr->data.loop_begin.range;
                /* Prefer the dim name threaded from the IR (correct for inferred
                   dims, whose range is 0). Fall back to size-matching, then to a
                   literal bound. */
                const char *dim_name = instr->data.loop_begin.range_name;
                if (!dim_name) dim_name = find_dim_name(ir, i, range);
                if (dim_name) {
                    cb_appendf(&cb, "for %s in range(%s):\n", var, dim_name);
                } else {
                    cb_appendf(&cb, "for %s in range(%d):\n", var, range);
                }
                cb.indent++;
                break;
            }
            case IR_LOOP_END:
                cb.indent--;
                break;
            case IR_GEN_LOOP_BEGIN: {
                IRGenLoop *gloop = &instr->data.gen_loop;
                for (int c = 0; c < gloop->num_cross_gens; c++) {
                    cb_indent(&cb);
                    cb_appendf(&cb, "%s_prev = %s\n",
                               gloop->cross_gens[c].name, gloop->cross_gens[c].name);
                }
                cb_indent(&cb); cb_append(&cb, "_it = 0\n");
                if (emit_trace) {
                    cb_indent(&cb); cb_append(&cb, "if _trace is not None:\n");
                    cb.indent++;
                    cb_indent(&cb);
                    cb_append(&cb, "_trace.append({\"kind\": \"generation\", \"gen\": 0, \"tensors\": {");
                    for (int c = 0; c < gloop->num_cross_gens; c++) {
                        if (c) cb_append(&cb, ", ");
                        cb_appendf(&cb, "\"%s\": _edgecc_snapshot(%s_prev)",
                                   gloop->cross_gens[c].name, gloop->cross_gens[c].name);
                    }
                    cb_append(&cb, "}})\n");
                    cb.indent--;
                }
                cb_indent(&cb); cb_append(&cb, "while True:\n");
                cb.indent++;
                for (int c = 0; c < gloop->num_cross_gens; c++) {
                    IRTensorAlloc tmp; memset(&tmp, 0, sizeof(tmp));
                    tmp.ndims = gloop->cross_gens[c].spatial_ndims;
                    tmp.empty_val = gloop->cross_gens[c].empty_val;
                    for (int k = 0; k < tmp.ndims; k++)
                        tmp.dim_sizes[k] = gloop->cross_gens[c].spatial_dims[k];
                    cb_indent(&cb);
                    cb_appendf(&cb, "%s_next = ", gloop->cross_gens[c].name);
                    emit_alloc(&cb, &tmp, ir, i);
                    cb_append(&cb, "\n");
                }
                for (int m = 0; m < gloop->num_intermediates; m++) {
                    IRTensorAlloc tmp; memset(&tmp, 0, sizeof(tmp));
                    tmp.ndims = gloop->intermediates[m].spatial_ndims;
                    tmp.empty_val = gloop->intermediates[m].empty_val;
                    for (int k = 0; k < tmp.ndims; k++)
                        tmp.dim_sizes[k] = gloop->intermediates[m].spatial_dims[k];
                    cb_indent(&cb);
                    cb_appendf(&cb, "%s = ", gloop->intermediates[m].name);
                    emit_alloc(&cb, &tmp, ir, i);
                    cb_append(&cb, "\n");
                }
                break;
            }
            case IR_GEN_LOOP_END: {
                if (emit_trace) {
                    cb_indent(&cb); cb_append(&cb, "if _trace is not None:\n");
                    cb.indent++;
                    cb_indent(&cb);
                    cb_append(&cb, "_trace.append({\"kind\": \"generation\", \"gen\": _it + 1, \"tensors\": {");
                    int _first = 1;
                    for (int c = 0; c < num_cross_gens; c++) {
                        if (!_first) cb_append(&cb, ", ");
                        cb_appendf(&cb, "\"%s\": _edgecc_snapshot(%s_next)", cross_gens[c], cross_gens[c]);
                        _first = 0;
                    }
                    for (int m = 0; m < num_intermediates; m++) {
                        if (!_first) cb_append(&cb, ", ");
                        cb_appendf(&cb, "\"%s\": _edgecc_snapshot(%s)", intermediates[m], intermediates[m]);
                        _first = 0;
                    }
                    cb_append(&cb, "}})\n");
                    cb.indent--;
                }
                if (num_cross_gens <= 1) {
                    const char *t = watch;
                    cb_indent(&cb);
                    cb_appendf(&cb, "if %s_next == %s_prev:\n", t, t);
                    cb.indent++;
                    cb_indent(&cb); cb_append(&cb, "break\n");
                    cb.indent--;
                    cb_indent(&cb);
                    cb_appendf(&cb, "%s_prev = %s_next\n", t, t);
                    cb_indent(&cb); cb_append(&cb, "_it += 1\n");
                    cb_indent(&cb); cb_append(&cb, "if _it >= __EDGECC_MAX_GEN__:\n");
                    cb.indent++;
                    cb_indent(&cb);
                    cb_append(&cb, "raise RuntimeError(\"generational loop did not stabilize\")\n");
                    cb.indent--;
                    cb.indent--;
                } else {
                    cb_indent(&cb);
                    cb_appendf(&cb, "_done = %s_next == %s_prev\n", watch, watch);
                    for (int c = 0; c < num_cross_gens; c++) {
                        cb_indent(&cb);
                        cb_appendf(&cb, "%s_prev = %s_next\n", cross_gens[c], cross_gens[c]);
                    }
                    cb_indent(&cb); cb_append(&cb, "if _done:\n");
                    cb.indent++;
                    cb_indent(&cb); cb_append(&cb, "break\n");
                    cb.indent--;
                    cb_indent(&cb); cb_append(&cb, "_it += 1\n");
                    cb_indent(&cb); cb_append(&cb, "if _it >= __EDGECC_MAX_GEN__:\n");
                    cb.indent++;
                    cb_indent(&cb);
                    cb_append(&cb, "raise RuntimeError(\"generational loop did not stabilize\")\n");
                    cb.indent--;
                    cb.indent--;
                }
                break;
            }
            case IR_LOAD: case IR_MAP: case IR_REDUCE:
                /* Combined into STORE emission below. */
                break;
            case IR_STORE:
                if (i >= 4) {
                    IRInstr *load1 = &ir->instrs[i - 4];
                    IRInstr *load2 = &ir->instrs[i - 3];
                    IRInstr *map_i = &ir->instrs[i - 2];
                    IRInstr *red_i = &ir->instrs[i - 1];
                    if (load1->type == IR_LOAD && load2->type == IR_LOAD &&
                        map_i->type == IR_MAP && red_i->type == IR_REDUCE) {
                        emit_general_skeleton(&cb,
                            load1, load2, map_i, red_i, instr);
                    }
                }
                break;
            case IR_SAVE:
                /* Save happens in __main__, not inside run(). */
                break;
            case IR_FUSED:
                /* Codegen for fused einsums is a later task; emit a clear
                   marker so an un-wired path can't silently produce wrong
                   output. */
                cb_indent(&cb);
                cb_append(&cb, "raise NotImplementedError(\"fused einsum codegen not yet implemented\")\n");
                break;
        }
    }

    if (output) {
        cb_indent(&cb);
        if (iterative && watch) {
            cb_appendf(&cb, "return %s_prev\n", watch);
        } else {
            cb_appendf(&cb, "return %s\n", output);
        }
    }
    cb.indent = 0;
    cb_append(&cb, "\n");

    /* __main__ block: re-emit allocs+inits+call */
    cb_append(&cb, "if __name__ == \"__main__\":\n");
    cb.indent = 1;

    /* Map input index -> load path (NULL if literal). */
    const char *load_path_of[8] = {0};
    for (int i = 0; i < ir->count; i++) {
        if (ir->instrs[i].type == IR_TENSOR_INIT &&
            ir->instrs[i].data.tensor_init.is_load) {
            for (int j = 0; j < inputs.count; j++)
                if (strcmp(ir->instrs[i].data.tensor_init.name, inputs.names[j]) == 0)
                    load_path_of[j] = ir->instrs[i].data.tensor_init.path;
        }
    }

    /* Emit allocs for INPUT tensors only (the output gets recomputed by run).
       Inputs that have an init (literal or load) are assigned below, so skip
       their dense pre-alloc — it is redundant and, for inferred dims, would
       reference a dim name that is only defined inside run(). */
    for (int i = 0; i < ir->count; i++) {
        IRInstr *instr = &ir->instrs[i];
        if (instr->type != IR_TENSOR_ALLOC) continue;
        int is_input = 0;
        int input_idx = -1;
        for (int j = 0; j < inputs.count; j++) {
            if (strcmp(instr->data.tensor_alloc.name, inputs.names[j]) == 0) {
                is_input = 1; input_idx = j; break;
            }
        }
        if (!is_input) continue;
        if (input_idx >= 0 && load_path_of[input_idx]) continue;
        /* Skip the pre-alloc when this input has any init statement. */
        int has_init = 0;
        for (int k = 0; k < ir->count; k++) {
            if (ir->instrs[k].type == IR_TENSOR_INIT &&
                strcmp(ir->instrs[k].data.tensor_init.name,
                       instr->data.tensor_alloc.name) == 0) { has_init = 1; break; }
        }
        if (has_init) continue;
        if (iterative &&
            name_in_list(instr->data.tensor_alloc.name, cross_gens, num_cross_gens)) continue;
        cb_indent(&cb);
        cb_appendf(&cb, "%s = ", instr->data.tensor_alloc.name);
        emit_alloc(&cb, &instr->data.tensor_alloc, ir, i);
        cb_append(&cb, "\n");
    }

    /* Loaded inputs: argv override + usage + count validation. */
    if (uses_fileio) {
        cb_indent(&cb); cb_append(&cb, "import sys\n");
        cb_indent(&cb); cb_append(&cb, "_argv = sys.argv[1:]\n");
        cb_indent(&cb); cb_append(&cb, "if _argv and _argv[0] in (\"-h\", \"--help\"):\n");
        cb.indent++;
        cb_indent(&cb);
        cb_append(&cb, "print(\"usage: <prog> [");
        {
            int first = 1;
            for (int j = 0; j < inputs.count; j++) {
                if (!load_path_of[j]) continue;
                if (!first) cb_append(&cb, " ");
                cb_appendf(&cb, "%s.mtx", inputs.names[j]);
                first = 0;
            }
        }
        cb_append(&cb, "]\"); sys.exit(0)\n");
        cb.indent--;
        {
            int k = 0;
            for (int j = 0; j < inputs.count; j++) {
                if (!load_path_of[j]) continue;
                cb_indent(&cb);
                cb_appendf(&cb,
                    "%s = _edgecc_load(_argv[%d]) if len(_argv) > %d else _edgecc_load(\"%s\")\n",
                    inputs.names[j], k, k, load_path_of[j]);
                k++;
            }
            if (k > 0) {
                cb_indent(&cb);
                cb_appendf(&cb, "if len(_argv) > %d:\n", k);
                cb.indent++;
                cb_indent(&cb);
                cb_appendf(&cb,
                    "raise SystemExit(\"error: expected at most %d input path(s), got %%d\" %% len(_argv))\n", k);
                cb.indent--;
            }
        }
    }

    /* Emit init statements (inputs only — output is initialized inside run).
       Loaded inputs have NULL literal data; skip them (assigned above). */
    for (int i = 0; i < ir->count; i++) {
        IRInstr *instr = &ir->instrs[i];
        if (instr->type != IR_TENSOR_INIT) continue;
        if (instr->data.tensor_init.is_load) continue;
        int is_input = 0;
        for (int j = 0; j < inputs.count; j++) {
            if (strcmp(instr->data.tensor_init.name, inputs.names[j]) == 0) { is_input = 1; break; }
        }
        if (!is_input) continue;
        if (instr->data.tensor_init.is_seed) {
            emit_seed_init(&cb, ir, instr->data.tensor_init.name,
                           instr->data.tensor_init.seed_source,
                           instr->data.tensor_init.seed_value);
            continue;
        }
        cb_indent(&cb);
        cb_appendf(&cb, "%s = ", instr->data.tensor_init.name);
        emit_literal(&cb, instr->data.tensor_init.data);
        cb_append(&cb, "\n");
    }

    /* Call run and print */
    if (output) {
        if (emit_trace) { cb_indent(&cb); cb_append(&cb, "_tr = []\n"); }
        cb_indent(&cb);
        cb_appendf(&cb, "%s = run(", output);
        for (int i = 0; i < inputs.count; i++) {
            if (i > 0) cb_append(&cb, ", ");
            cb_append(&cb, inputs.names[i]);
        }
        if (emit_trace) cb_append(&cb, ", _trace=_tr");
        cb_append(&cb, ")\n");
        cb_indent(&cb);
        cb_appendf(&cb, "print(%s)\n", output);
        if (emit_trace) {
            cb_indent(&cb); cb_append(&cb, "import json\n");
            cb_indent(&cb);
            cb_appendf(&cb,
                "json.dump({\"schema\": \"edgecc-trace/1\", \"kernel\": \"%s\", "
                "\"dims\": {\"N\": len(%s)}, \"gen_tensors\": [",
                kernel_name, output);
            {
                int _first = 1;
                for (int c = 0; c < num_cross_gens; c++) {
                    if (!_first) cb_append(&cb, ", ");
                    cb_appendf(&cb, "\"%s\"", cross_gens[c]); _first = 0;
                }
                for (int m = 0; m < num_intermediates; m++) {
                    if (!_first) cb_append(&cb, ", ");
                    cb_appendf(&cb, "\"%s\"", intermediates[m]); _first = 0;
                }
            }
            /* Per-tensor empty value so the renderer can tell "active" (non-empty)
               from empty regardless of semiring (e.g. min-plus source dist 0.0). */
            cb_append(&cb, "], \"empty\": {");
            {
                int _first = 1;
                for (int c = 0; c < num_cross_gens; c++) {
                    char *ev = format_double_literal(cross_gen_empty[c]);
                    if (!_first) cb_append(&cb, ", ");
                    cb_appendf(&cb, "\"%s\": %s", cross_gens[c], ev);
                    free(ev); _first = 0;
                }
                for (int m = 0; m < num_intermediates; m++) {
                    char *ev = format_double_literal(intermediate_empty[m]);
                    if (!_first) cb_append(&cb, ", ");
                    cb_appendf(&cb, "\"%s\": %s", intermediates[m], ev);
                    free(ev); _first = 0;
                }
            }
            cb_appendf(&cb,
                "}, \"watch\": \"%s\", \"steps\": _tr}, open(\"%s.trace.json\", \"w\"))\n",
                watch ? watch : "", kernel_name);
        }
    }

    /* Emit save calls after run+print. */
    for (int i = 0; i < ir->count; i++) {
        if (ir->instrs[i].type != IR_SAVE) continue;
        cb_indent(&cb);
        cb_appendf(&cb, "_edgecc_save(%s, \"%s\")\n",
                   ir->instrs[i].data.save.tensor_name,
                   ir->instrs[i].data.save.path);
    }

    free(output);
    free_tensor_name_list(&inputs);
    return cb.buf;
}
