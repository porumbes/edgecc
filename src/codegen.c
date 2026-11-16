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

/* A generational tensor stores every generation (the paper's rank I), so a
   reference names one slice: kind 1 -> X[_i], kind 2 -> X[_i + 1]; kind 0 is
   a non-generational tensor (plain name). */
static void gen_slice_name(char *out, size_t cap, const char *name, int kind) {
    if (kind == 2)      snprintf(out, cap, "%s[_i + 1]", name);
    else if (kind == 1) snprintf(out, cap, "%s[_i]", name);
    else                snprintf(out, cap, "%s", name);
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
    char **names;   /* grows on demand; no cap on the number of inputs */
    int count;
    int cap;
} TensorNameList;

/* Append a copy of tensor name `n` to `list` unless it is already present
 * (distinct names, first-seen order). Used for both the read and the written
 * name lists. Caller-owned. */
static void add_input_name(TensorNameList *list, const char *n) {
    if (!n) return;
    for (int j = 0; j < list->count; j++)
        if (strcmp(list->names[j], n) == 0) return;
    if (list->count == list->cap) {
        list->cap = list->cap ? 2 * list->cap : 8;
        list->names = realloc(list->names, (size_t)list->cap * sizeof(char *));
    }
    list->names[list->count++] = strdup(n);
}

static int name_in_tnl(const TensorNameList *l, const char *name) {
    for (int i = 0; i < l->count; i++)
        if (strcmp(l->names[i], name) == 0) return 1;
    return 0;
}

/* Read tensors (IR loads, in load order) go to `inputs`; every tensor an
   Einsum stores to goes to `written`. Caller frees both lists. */
static void collect_io(IRProgram *ir, TensorNameList *inputs, TensorNameList *written) {
    inputs->names = NULL;  inputs->count = 0;  inputs->cap = 0;
    written->names = NULL; written->count = 0; written->cap = 0;
    for (int i = 0; i < ir->count; i++) {
        IRInstr *instr = &ir->instrs[i];
        if (instr->type == IR_LOAD) {
            add_input_name(inputs, instr->data.load.tensor_name);
        } else if (instr->type == IR_STORE) {
            add_input_name(written, instr->data.store.tensor_name);
        } else if (instr->type == IR_FUSED) {
            IRFused *fu = &instr->data.fused;
            add_input_name(inputs, fu->inner1.name);
            add_input_name(inputs, fu->inner2.name);
            add_input_name(inputs, fu->outer.name);
            add_input_name(written, fu->out_name);
        }
    }
}

/* Loaded inputs in `init` statement order: the one order shared by the meta
   "file_inputs" list, the -h usage string, and positional argv binding.
   Returns the count; fills names/paths (borrowed) up to the input count. */
static int collect_file_inputs(IRProgram *ir, TensorNameList *inputs,
                               const char **names, const char **paths) {
    int n = 0;
    for (int i = 0; i < ir->count; i++) {
        if (ir->instrs[i].type != IR_TENSOR_INIT || !ir->instrs[i].data.tensor_init.is_load)
            continue;
        for (int j = 0; j < inputs->count; j++) {
            if (strcmp(ir->instrs[i].data.tensor_init.name, inputs->names[j]) == 0) {
                names[n] = inputs->names[j];
                paths[n] = ir->instrs[i].data.tensor_init.path;
                n++;
                break;
            }
        }
    }
    return n;
}

static void free_tensor_name_list(TensorNameList *l) {
    for (int i = 0; i < l->count; i++) free(l->names[i]);
    free(l->names);
    l->names = NULL;
    l->count = 0;
    l->cap = 0;
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
        default:
            /* Unreachable: the parser rejects unknown compute-op names. */
            fprintf(stderr, "edgecc: internal error: unknown compute operator %d\n", (int)op);
            abort();
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

    /* Generational tensors store every generation; index the slice. */
    char a_name[128], b_name[128], z_name[128];
    gen_slice_name(a_name, sizeof(a_name), load1->data.load.tensor_name, load1->data.load.gen_read);
    gen_slice_name(b_name, sizeof(b_name), load2->data.load.tensor_name, load2->data.load.gen_read);
    gen_slice_name(z_name, sizeof(z_name), store_i->data.store.tensor_name, store_i->data.store.gen_write);

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

/* ---- Fused (two-action) skeleton ---- */

/* Resolve a fused operand's generation slice into `out`. */
static void fused_operand_name(const IRFusedOperand *op, char *out, size_t cap) {
    gen_slice_name(out, cap, op->name, op->gen_read);
}

/* Emit the map body for action 1 (both operands are tensor loads).
 * Produces:  <dst>      = <compute(map1)>   (when merge passes)
 *            <dst>_emit = <dst> != eZ
 * else:      <dst>_emit = False; <dst> = eZ
 * The operand value/existence locals are a_val/a_exists/b_val/b_exists. */
static void emit_fused_map_body(CodeBuf *cb,
                                const IRFusedOperand *opA,
                                const IRFusedOperand *opB,
                                OperatorType op, MergeOp merge,
                                const char *eZ, const char *dst) {
    char a_name[160], b_name[160];
    char *eA = format_double_literal(opA->empty);
    char *eB = format_double_literal(opB->empty);
    fused_operand_name(opA, a_name, sizeof(a_name));
    fused_operand_name(opB, b_name, sizeof(b_name));

    cb_indent(cb); cb_append(cb, "a_val = ");
    emit_tensor_access(cb, a_name, opA->coords, opA->ncoords);
    cb_append(cb, "\n");
    cb_indent(cb); cb_append(cb, "b_val = ");
    emit_tensor_access(cb, b_name, opB->coords, opB->ncoords);
    cb_append(cb, "\n");

    cb_indent(cb);
    cb_appendf(cb, "a_exists = a_val %s %s\n", opA->negate ? "==" : "!=", eA);
    cb_indent(cb);
    cb_appendf(cb, "b_exists = b_val %s %s\n", opB->negate ? "==" : "!=", eB);

    cb_indent(cb); cb_append(cb, "if ");
    emit_merge_guard(cb, merge, "a_exists", "b_exists");
    cb_append(cb, ":\n");
    cb->indent++;
    cb_indent(cb);
    cb_appendf(cb, "a_op = %s if a_exists else %s\n", opA->negate ? "1.0" : "a_val", eA);
    cb_indent(cb);
    cb_appendf(cb, "b_op = %s if b_exists else %s\n", opB->negate ? "1.0" : "b_val", eB);
    char buf[256];
    cb_indent(cb); cb_appendf(cb, "%s = ", dst);
    cb_append(cb, compute_op_pyexpr(op, "a_op", "b_op", buf, sizeof(buf)));
    cb_append(cb, "\n");
    cb_indent(cb); cb_appendf(cb, "%s_emit = %s != %s\n", dst, dst, eZ);
    cb->indent--;
    cb_indent(cb); cb_append(cb, "else:\n");
    cb->indent++;
    cb_indent(cb); cb_appendf(cb, "%s_emit = False\n", dst);
    cb_indent(cb); cb_appendf(cb, "%s = %s\n", dst, eZ);
    cb->indent--;

    free(eA); free(eB);
}

/* Emit a reduce step into accumulator `acc` from value `val` (with existence
 * flag `val_emit`). Mirrors emit_general_skeleton's reduce half. */
static void emit_fused_reduce_step(CodeBuf *cb, OperatorType op, MergeOp merge,
                                   const char *eZ, const char *acc,
                                   const char *val, const char *val_emit) {
    cb_indent(cb); cb_appendf(cb, "_s1 = %s != %s\n", acc, eZ);
    cb_indent(cb); cb_appendf(cb, "_s2 = %s\n", val_emit);
    cb_indent(cb); cb_append(cb, "if ");
    emit_merge_guard(cb, merge, "_s1", "_s2");
    cb_append(cb, ":\n");
    cb->indent++;
    cb_indent(cb); cb_appendf(cb, "_vfr = %s if %s else %s\n", val, val_emit, eZ);
    char buf[256];
    cb_indent(cb); cb_appendf(cb, "%s = ", acc);
    cb_append(cb, compute_op_pyexpr(op, acc, "_vfr", buf, sizeof(buf)));
    cb_append(cb, "\n");
    cb->indent--;
    cb_indent(cb); cb_append(cb, "else:\n");
    cb->indent++;
    cb_indent(cb); cb_appendf(cb, "%s = %s\n", acc, eZ);
    cb->indent--;
}

/* Emit action 2: combine the scalar intermediate (_inter, existence in
   inter_flag) with the outer operand. Produces `_m2`. Operand order follows
   fu->inter_is_left2.
   want_emit != 0: also produce `_m2_emit` (True in the merge-pass branch, False
     in the else), closing both branches; caller follows with a reduce step.
   want_emit == 0: emit only the merge-pass branch (i_op/o_op/_m2) and leave the
     caller positioned *inside* that `if` body (indent not decremented, no else);
     the caller emits its store, then dedents and writes its own else. */
static void emit_fused_action2(CodeBuf *cb, IRFused *fu, const char *eZ,
                               const char *inter_flag, int want_emit) {
    char buf[256];
    char *eC = format_double_literal(fu->outer.empty);
    char c_name[160];
    fused_operand_name(&fu->outer, c_name, sizeof(c_name));

    cb_indent(cb); cb_append(cb, "c_val = ");
    emit_tensor_access(cb, c_name, fu->outer.coords, fu->outer.ncoords);
    cb_append(cb, "\n");
    cb_indent(cb);
    cb_appendf(cb, "c_exists = c_val %s %s\n", fu->outer.negate ? "==" : "!=", eC);

    const char *l_exists = fu->inter_is_left2 ? inter_flag : "c_exists";
    const char *r_exists = fu->inter_is_left2 ? "c_exists" : inter_flag;
    cb_indent(cb); cb_append(cb, "if ");
    emit_merge_guard(cb, fu->map2_merge, l_exists, r_exists);
    cb_append(cb, ":\n");
    cb->indent++;
    cb_indent(cb);
    cb_appendf(cb, "i_op = _inter if %s else %s\n", inter_flag, eZ);
    cb_indent(cb);
    cb_appendf(cb, "o_op = %s if c_exists else %s\n",
               fu->outer.negate ? "1.0" : "c_val", eC);
    cb_indent(cb); cb_append(cb, "_m2 = ");
    if (fu->inter_is_left2)
        cb_append(cb, compute_op_pyexpr(fu->map2_op, "i_op", "o_op", buf, sizeof(buf)));
    else
        cb_append(cb, compute_op_pyexpr(fu->map2_op, "o_op", "i_op", buf, sizeof(buf)));
    cb_append(cb, "\n");
    if (want_emit) {
        cb_indent(cb); cb_appendf(cb, "_m2_emit = _m2 != %s\n", eZ);
        cb->indent--;
        cb_indent(cb); cb_append(cb, "else:\n");
        cb->indent++;
        cb_indent(cb); cb_append(cb, "_m2_emit = False\n");
        cb_indent(cb); cb_appendf(cb, "_m2 = %s\n", eZ);
        cb->indent--;
    }
    /* want_emit == 0: leave caller inside the `if` body. */

    free(eC);
}

/* Emit the full fused two-action skeleton (push / pull / neither). */
static void emit_fused_skeleton(CodeBuf *cb, IRFused *fu) {
    char *eZ = format_double_literal(fu->empty_z);

    /* Output (store) tensor name, gen-resolved. */
    char z_name[160];
    gen_slice_name(z_name, sizeof(z_name), fu->out_name, fu->gen_write);

    /* Free-index loops (outermost). */
    for (int i = 0; i < fu->num_free; i++) {
        cb_indent(cb);
        if (fu->free_dimnames[i])
            cb_appendf(cb, "for %s in range(%s):\n", fu->free_names[i], fu->free_dimnames[i]);
        else
            cb_appendf(cb, "for %s in range(%d):\n", fu->free_names[i], fu->free_sizes[i]);
        cb->indent++;
    }

    if (fu->reduce_on_action == 1) {
        /* PUSH: action 1 reduces; action 2 map-only -> store. */
        char *rid = format_double_literal(fu->reduce1_identity);
        cb_indent(cb); cb_appendf(cb, "_acc1 = %s\n", rid);
        free(rid);
        for (int i = 0; i < fu->num_red; i++) {
            cb_indent(cb);
            if (fu->red_dimnames[i])
                cb_appendf(cb, "for %s in range(%s):\n", fu->red_names[i], fu->red_dimnames[i]);
            else
                cb_appendf(cb, "for %s in range(%d):\n", fu->red_names[i], fu->red_sizes[i]);
            cb->indent++;
        }
        emit_fused_map_body(cb, &fu->inner1, &fu->inner2,
                            fu->map1_op, fu->map1_merge, eZ, "_m1");
        emit_fused_reduce_step(cb, fu->reduce1_op, fu->reduce1_merge, eZ,
                               "_acc1", "_m1", "_m1_emit");
        for (int i = 0; i < fu->num_red; i++) cb->indent--;
        cb_indent(cb); cb_append(cb, "_inter = _acc1\n");
        cb_indent(cb); cb_appendf(cb, "_inter_emit = _inter != %s\n", eZ);

        /* action 2 (map-only) -> store */
        emit_fused_action2(cb, fu, eZ, "_inter_emit", 0);
        cb_indent(cb);
        emit_tensor_access(cb, z_name, fu->out_coords, fu->out_ncoords);
        cb_appendf(cb, " = _m2 if _m2 != %s else %s\n", eZ, eZ);
        cb->indent--;
        cb_indent(cb); cb_append(cb, "else:\n");
        cb->indent++;
        cb_indent(cb);
        emit_tensor_access(cb, z_name, fu->out_coords, fu->out_ncoords);
        cb_appendf(cb, " = %s\n", eZ);
        cb->indent--;

    } else if (fu->reduce_on_action == 2) {
        /* PULL: action 1 map-only inside shared red loop; action 2 reduces. */
        char *rid = format_double_literal(fu->reduce2_identity);
        cb_indent(cb); cb_appendf(cb, "_acc2 = %s\n", rid);
        free(rid);
        for (int i = 0; i < fu->num_red; i++) {
            cb_indent(cb);
            if (fu->red_dimnames[i])
                cb_appendf(cb, "for %s in range(%s):\n", fu->red_names[i], fu->red_dimnames[i]);
            else
                cb_appendf(cb, "for %s in range(%d):\n", fu->red_names[i], fu->red_sizes[i]);
            cb->indent++;
        }
        emit_fused_map_body(cb, &fu->inner1, &fu->inner2,
                            fu->map1_op, fu->map1_merge, eZ, "_inter");

        emit_fused_action2(cb, fu, eZ, "_inter_emit", 1);

        emit_fused_reduce_step(cb, fu->reduce2_op, fu->reduce2_merge, eZ,
                               "_acc2", "_m2", "_m2_emit");
        for (int i = 0; i < fu->num_red; i++) cb->indent--;
        cb_indent(cb);
        emit_tensor_access(cb, z_name, fu->out_coords, fu->out_ncoords);
        cb_append(cb, " = _acc2\n");

    } else {
        /* NEITHER: no reduction. action 1 map -> _inter, action 2 map -> store. */
        emit_fused_map_body(cb, &fu->inner1, &fu->inner2,
                            fu->map1_op, fu->map1_merge, eZ, "_inter");

        emit_fused_action2(cb, fu, eZ, "_inter_emit", 0);
        cb_indent(cb);
        emit_tensor_access(cb, z_name, fu->out_coords, fu->out_ncoords);
        cb_appendf(cb, " = _m2 if _m2 != %s else %s\n", eZ, eZ);
        cb->indent--;
        cb_indent(cb); cb_append(cb, "else:\n");
        cb->indent++;
        cb_indent(cb);
        emit_tensor_access(cb, z_name, fu->out_coords, fu->out_ncoords);
        cb_appendf(cb, " = %s\n", eZ);
        cb->indent--;
    }

    for (int i = 0; i < fu->num_free; i++) cb->indent--;

    free(eZ);
}

/* ---- Main codegen ---- */

/* Return 1 if `name` matches any entry in `list[0..n)`. */
static int name_in_list(const char *name, const char **list, int n) {
    for (int i = 0; i < n; i++)
        if (strcmp(name, list[i]) == 0) return 1;
    return 0;
}

/* Runtime Python subscript depth of alloc dim `d`: a generational axis (dim_size
   == -1 sentinel) is not a materialized Python dimension, so skip it. Returns
   the index of `d` among the data axes (0 for the first data axis, and so on). */
static int python_axis_of(const IRTensorAlloc *a, int d) {
    int axis = 0;
    for (int j = 0; j < d; j++)
        if (a->dim_sizes[j] != -1) axis++;
    return axis;
}

/* Python expression for the length of `name` along data axis `axis`:
   `len(X)`, `len(X[0])`, `len(X[0][0])`, ... */
static void len_expr(char *buf, size_t n, const char *name, int axis) {
    size_t used = (size_t)snprintf(buf, n, "len(%s", name);
    for (int k = 0; k < axis && used < n; k++)
        used += (size_t)snprintf(buf + used, n - used, "[0]");
    if (used < n) snprintf(buf + used, n - used, ")");
}

/* Return 1 if `dimname` names an inferred dim (`dim N = *`). */
static int dim_is_inferred(IRProgram *ir, const char *dimname) {
    if (!dimname) return 0;
    for (int i = 0; i < ir->count; i++)
        if (ir->instrs[i].type == IR_DIM_DEF &&
            strcmp(ir->instrs[i].data.dim_def.name, dimname) == 0)
            return ir->instrs[i].data.dim_def.is_inferred;
    return 0;
}

/* Emit the shape _edgecc_load checks a loaded tensor against, as a Python
   tuple literal with one entry per data axis (the generational axis is
   skipped): the size of a fixed dim, the name (a str, not checked) of an
   inferred dim. A one-entry tuple keeps its comma: `(4,)`, `("N",)`. */
static void emit_load_shape(CodeBuf *cb, IRProgram *ir, const char *tensor_name) {
    int entries = 0;
    cb_append(cb, "(");
    for (int i = 0; i < ir->count; i++) {
        if (ir->instrs[i].type != IR_TENSOR_ALLOC) continue;
        IRTensorAlloc *a = &ir->instrs[i].data.tensor_alloc;
        if (strcmp(a->name, tensor_name) != 0) continue;
        for (int d = 0; d < a->ndims; d++) {
            if (a->dim_sizes[d] == -1) continue;   /* generational axis */
            if (entries++) cb_append(cb, ", ");
            if (dim_is_inferred(ir, a->dim_names[d])) cb_appendf(cb, "\"%s\"", a->dim_names[d]);
            else                                      cb_appendf(cb, "%d", a->dim_sizes[d]);
        }
        break;
    }
    cb_append(cb, entries == 1 ? ",)" : ")");
}

/* For an inferred dim `dimname`, find the first INPUT tensor alloc that has it
   on some axis and emit `dimname = len(input)`, `len(input[0])`, ... */
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
                char le[256];
                len_expr(le, sizeof le, a->name, python_axis_of(a, d));
                cb_indent(cb);
                cb_appendf(cb, "%s = %s\n", dimname, le);
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

    char lenexpr[256];
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
                        len_expr(lenexpr, sizeof(lenexpr), pin, python_axis_of(pa, d));
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
        char le[256];
        len_expr(le, sizeof le, a->name, axis);
        cb_indent(cb);
        cb_appendf(cb, "if %s != %s:\n", le, dimname);
        cb->indent++;
        cb_indent(cb);
        cb_appendf(cb,
            "raise ValueError(\"edgecc: input '%s' has size %%d on dim %s "
            "but '%s' pins it to %%d\" %% (%s, %s))\n",
            a->name, dimname, first, le, dimname);
        cb->indent--;
    }
}

char *codegen_python(IRProgram *ir) {
    return codegen_python_ex(ir, 0, NULL);
}

/* Build the per-generation trace from the stored history, after the loop.
   Step g holds each cross-generational tensor at generation g and, for g >= 1,
   each intermediate at generation g-1 (written during iteration g-1) -- the
   same steps edgecc-trace/1 always recorded. Slices are never written after
   their iteration, so no copies are needed. */
static void emit_trace_from_history(CodeBuf *cb,
                                    const char **cross, int nc,
                                    const char **inter, int ni) {
    cb_indent(cb); cb_append(cb, "if _trace is not None:\n");
    cb->indent++;
    cb_indent(cb);
    cb_append(cb, "_trace.append({\"kind\": \"generation\", \"gen\": 0, \"tensors\": {");
    for (int c = 0; c < nc; c++) {
        if (c) cb_append(cb, ", ");
        cb_appendf(cb, "\"%s\": %s[0]", cross[c], cross[c]);
    }
    cb_append(cb, "}})\n");
    cb_indent(cb); cb_append(cb, "for _g in range(1, _i + 2):\n");
    cb->indent++;
    cb_indent(cb);
    cb_append(cb, "_trace.append({\"kind\": \"generation\", \"gen\": _g, \"tensors\": {");
    int first = 1;
    for (int c = 0; c < nc; c++) {
        if (!first) cb_append(cb, ", ");
        cb_appendf(cb, "\"%s\": %s[_g]", cross[c], cross[c]);
        first = 0;
    }
    for (int m = 0; m < ni; m++) {
        if (!first) cb_append(cb, ", ");
        cb_appendf(cb, "\"%s\": %s[_g - 1]", inter[m], inter[m]);
        first = 0;
    }
    cb_append(cb, "}})\n");
    cb->indent -= 2;
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
            "def _edgecc_load(path, name, shape):\n"
            "    # shape: one entry per data axis, an int for a fixed dim (checked) or\n"
            "    # an inferred dim's name (checked inside run()).\n"
            "    if path.endswith(\".mtx\"):\n"
            "        try:\n"
            "            from scipy.io import mmread\n"
            "        except ImportError:\n"
            "            raise ImportError(\"edgecc load/save requires scipy: pip install scipy\")\n"
            "        m = mmread(path)\n"
            "        a = m.toarray() if hasattr(m, \"toarray\") else m\n"
            "    elif path.endswith(\".npy\"):\n"
            "        try:\n"
            "            import numpy as np\n"
            "        except ImportError:\n"
            "            raise ImportError(\"edgecc .npy load/save requires numpy: pip install numpy\")\n"
            "        a = np.load(path, allow_pickle=False)\n"
            "    else:\n"
            "        raise ValueError(\"edgecc load: unsupported file extension for %r (supported: .mtx, .npy)\" % path)\n"
            "    if a.dtype.kind not in (\"b\", \"i\", \"u\", \"f\"):\n"
            "        raise ValueError(\"edgecc load: \\\"%s\\\" for '%s' has dtype %s; edgecc tensors are real \"\n"
            "                         \"(split Re/Im into separate tensors or a stacked axis)\" % (path, name, a.dtype))\n"
            "    if a.ndim != len(shape) or any(isinstance(want, int) and want != got\n"
            "                                   for want, got in zip(shape, a.shape)):\n"
            "        raise ValueError(\"edgecc load: \\\"%s\\\" for '%s' has shape %s, expected %s\" % (\n"
            "            path, name, \"(\" + \", \".join(str(x) for x in a.shape) + \")\",\n"
            "            \"(\" + \", \".join(str(x) for x in shape) + \")\"))\n"
            "    return a.astype(float).tolist()\n"
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
            "    if path.endswith(\".npy\"):\n"
            "        try:\n"
            "            import numpy as np\n"
            "        except ImportError:\n"
            "            raise ImportError(\"edgecc .npy load/save requires numpy: pip install numpy\")\n"
            "        np.save(path, np.array(tensor, dtype=float))   # full shape, float64\n"
            "        return\n"
            "    raise ValueError(\"edgecc save: unsupported file extension for %r (supported: .mtx, .npy)\" % path)\n"
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

    /* Discover inputs and written tensors */
    TensorNameList inputs, written;
    collect_io(ir, &inputs, &written);

    /* run()'s outputs, in return order (names borrowed from the IR). */
    int num_outs = 0;
    for (int i = 0; i < ir->count; i++)
        if (ir->instrs[i].type == IR_OUTPUT) num_outs++;
    const char **outs = calloc(num_outs ? (size_t)num_outs : 1, sizeof(const char *));
    num_outs = 0;
    for (int i = 0; i < ir->count; i++)
        if (ir->instrs[i].type == IR_OUTPUT)
            outs[num_outs++] = ir->instrs[i].data.output.tensor_name;

    /* Detect generational mode and capture the gen-loop metadata. */
    int iterative = 0;
    const char *cross_gens[MAX_GEN_TENSORS]; int num_cross_gens = 0;
    double cross_gen_empty[MAX_GEN_TENSORS];
    const char *watch = NULL;
    int stop_kind = STOP_EQUAL, stop_gen_a = 1, stop_gen_b = 0;
    double watch_empty = 0.0;
    const char *intermediates[MAX_GEN_TENSORS];
    double intermediate_empty[MAX_GEN_TENSORS];
    int num_intermediates = 0;
    for (int i = 0; i < ir->count; i++) {
        if (ir->instrs[i].type == IR_GEN_LOOP_BEGIN) {
            IRGenLoop *gl = &ir->instrs[i].data.gen_loop;
            iterative = 1;
            num_cross_gens = gl->num_cross_gens;
            for (int j = 0; j < num_cross_gens; j++) {
                cross_gens[j] = gl->cross_gens[j].name;
                cross_gen_empty[j] = gl->cross_gens[j].empty_val;
            }
            watch = gl->watch;
            stop_kind   = gl->stop_kind;
            stop_gen_a  = gl->stop_gen_a;
            stop_gen_b  = gl->stop_gen_b;
            watch_empty = gl->watch_empty;
            num_intermediates = gl->num_intermediates;
            for (int j = 0; j < num_intermediates; j++) {
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

    /* Occupancy helper for `stop: ||X[g]|| == 0`; emitted only when used. */
    if (iterative && stop_kind == STOP_OCCUPANCY) {
        cb_append(&cb,
            "def _edgecc_occupancy(x, empty):\n"
            "    if isinstance(x, list):\n"
            "        return sum(_edgecc_occupancy(v, empty) for v in x)\n"
            "    return 0 if x == empty else 1\n"
            "\n");
    }

    /* Cascade I/O: intermediates are never parameters. */
    if (iterative) {
        int kept = 0;
        for (int i = 0; i < inputs.count; i++) {
            if (name_in_list(inputs.names[i], intermediates, num_intermediates)) {
                free(inputs.names[i]); continue;
            }
            inputs.names[kept++] = inputs.names[i];
        }
        inputs.count = kept;
    } else {
        /* Non-generational: a tensor written by an Einsum starts empty inside
           run() (paper: output tensors are initially empty), so it is never a
           parameter, even when it is also read. Semantic analysis rejects an
           init on it and a read before its first write. */
        int kept = 0;
        for (int i = 0; i < inputs.count; i++) {
            if (name_in_tnl(&written, inputs.names[i])) {
                free(inputs.names[i]); continue;
            }
            inputs.names[kept++] = inputs.names[i];
        }
        inputs.count = kept;
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
    if (num_outs == 1)
        cb_appendf(&cb, "    \"output\": \"%s\",\n", outs[0]);
    cb_append(&cb, "    \"outputs\": [");
    for (int k = 0; k < num_outs; k++) {
        if (k > 0) cb_append(&cb, ", ");
        cb_appendf(&cb, "\"%s\"", outs[k]);
    }
    cb_append(&cb, "],\n");
    size_t fi_cap = inputs.count ? (size_t)inputs.count : 1;
    const char **file_in_names = calloc(fi_cap, sizeof(const char *));
    const char **file_in_paths = calloc(fi_cap, sizeof(const char *));
    int num_file_ins = collect_file_inputs(ir, &inputs, file_in_names, file_in_paths);
    cb_append(&cb, "    \"file_inputs\": [");
    for (int i = 0; i < num_file_ins; i++) {
        if (i > 0) cb_append(&cb, ", ");
        cb_appendf(&cb, "\"%s\"", file_in_names[i]);
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
                /* Every tensor an Einsum writes is allocated (empty) inside
                   run(); a generational tensor is instead built by the gen loop
                   (from the caller's generation 0, or empty for an
                   intermediate). */
                if (!iterative && name_in_tnl(&written, instr->data.tensor_alloc.name)) {
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
                /* Every generational tensor is a list indexed by generation.
                   The caller supplies generation 0 of each cross-generational
                   tensor; an intermediate starts with no generations. */
                for (int c = 0; c < gloop->num_cross_gens; c++) {
                    cb_indent(&cb);
                    cb_appendf(&cb, "%s = [%s]\n",
                               gloop->cross_gens[c].name, gloop->cross_gens[c].name);
                }
                for (int m = 0; m < gloop->num_intermediates; m++) {
                    cb_indent(&cb);
                    cb_appendf(&cb, "%s = []\n", gloop->intermediates[m].name);
                }
                cb_indent(&cb); cb_append(&cb, "_i = 0\n");
                cb_indent(&cb); cb_append(&cb, "while True:\n");
                cb.indent++;
                /* Each iteration first appends one empty slice per tensor it
                   writes: cross-gens at i+1, intermediates at i. */
                IRGenAlloc *allocs[16]; int na = 0;
                for (int c = 0; c < gloop->num_cross_gens; c++) allocs[na++] = &gloop->cross_gens[c];
                for (int m = 0; m < gloop->num_intermediates; m++) allocs[na++] = &gloop->intermediates[m];
                for (int k = 0; k < na; k++) {
                    IRTensorAlloc tmp; memset(&tmp, 0, sizeof(tmp));
                    tmp.ndims = allocs[k]->spatial_ndims;
                    tmp.empty_val = allocs[k]->empty_val;
                    for (int j = 0; j < tmp.ndims; j++) {
                        tmp.dim_sizes[j] = allocs[k]->spatial_dims[j];
                        tmp.dim_names[j] = allocs[k]->spatial_dim_names[j];
                    }
                    cb_indent(&cb);
                    cb_appendf(&cb, "%s.append(", allocs[k]->name);
                    emit_alloc(&cb, &tmp, ir, i);
                    cb_append(&cb, ")\n");
                }
                break;
            }
            case IR_GEN_LOOP_END: {
                /* Compute-then-check: the generation that satisfies the stop
                   predicate is kept, then the loop ends. */
                char pa[160], pb[160];
                gen_slice_name(pa, sizeof(pa), watch, stop_gen_a + 1);
                cb_indent(&cb);
                if (stop_kind == STOP_OCCUPANCY) {
                    char *ev = format_double_literal(watch_empty);
                    cb_appendf(&cb, "if _edgecc_occupancy(%s, %s) == 0:\n", pa, ev);
                    free(ev);
                } else {
                    gen_slice_name(pb, sizeof(pb), watch, stop_gen_b + 1);
                    cb_appendf(&cb, "if %s == %s:\n", pa, pb);
                }
                cb.indent++;
                cb_indent(&cb); cb_append(&cb, "break\n");
                cb.indent--;
                cb_indent(&cb); cb_append(&cb, "_i += 1\n");
                cb_indent(&cb); cb_append(&cb, "if _i >= __EDGECC_MAX_GEN__:\n");
                cb.indent++;
                cb_indent(&cb);
                cb_append(&cb, "raise RuntimeError(\"generational loop did not stop\")\n");
                cb.indent--;
                cb.indent--;   /* leave the while loop */
                if (emit_trace)
                    emit_trace_from_history(&cb, cross_gens, num_cross_gens,
                                            intermediates, num_intermediates);
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
            case IR_OUTPUT:
                /* Consumed for meta, return and __main__. */
                break;
            case IR_FUSED:
                if (!produced_loops) {
                    cb_append(&cb, "\n");
                    produced_loops = 1;
                }
                emit_fused_skeleton(&cb, &instr->data.fused);
                break;
        }
    }

    /* A generational output is returned with every generation. */
    if (num_outs > 0) {
        cb_indent(&cb);
        if (num_outs == 1) {
            cb_appendf(&cb, "return %s\n", outs[0]);
        } else {
            cb_append(&cb, "return (");
            for (int k = 0; k < num_outs; k++) {
                if (k > 0) cb_append(&cb, ", ");
                cb_append(&cb, outs[k]);
            }
            cb_append(&cb, ")\n");
        }
    }
    cb.indent = 0;
    cb_append(&cb, "\n");

    /* __main__ block: re-emit allocs+inits+call */
    cb_append(&cb, "if __name__ == \"__main__\":\n");
    cb.indent = 1;

    /* Map input index -> load path (NULL if literal). */
    const char **load_path_of = calloc(inputs.count ? (size_t)inputs.count : 1,
                                       sizeof(const char *));
    for (int i = 0; i < ir->count; i++) {
        if (ir->instrs[i].type == IR_TENSOR_INIT &&
            ir->instrs[i].data.tensor_init.is_load) {
            for (int j = 0; j < inputs.count; j++)
                if (strcmp(ir->instrs[i].data.tensor_init.name, inputs.names[j]) == 0)
                    load_path_of[j] = ir->instrs[i].data.tensor_init.path;
        }
    }

    /* Emit allocs for INPUT tensors only (the outputs get recomputed by run).
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
        for (int k = 0; k < num_file_ins; k++) {
            /* semantic.c guarantees the default path ends in .mtx or .npy */
            const char *ext = strrchr(file_in_paths[k], '.');
            cb_appendf(&cb, "%s%s%s", k > 0 ? " " : "", file_in_names[k], ext ? ext : "");
        }
        cb_append(&cb, "]\"); sys.exit(0)\n");
        cb.indent--;
        {
            int k = num_file_ins;
            for (int j = 0; j < num_file_ins; j++) {
                cb_indent(&cb);
                cb_appendf(&cb,
                    "%s = _edgecc_load(_argv[%d] if len(_argv) > %d else \"%s\", \"%s\", ",
                    file_in_names[j], j, j, file_in_paths[j], file_in_names[j]);
                emit_load_shape(&cb, ir, file_in_names[j]);
                cb_append(&cb, ")\n");
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

    /* Emit init statements (inputs only — outputs are initialized inside run).
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
    if (num_outs > 0) {
        if (emit_trace) { cb_indent(&cb); cb_append(&cb, "_tr = []\n"); }
        cb_indent(&cb);
        for (int k = 0; k < num_outs; k++) {
            if (k > 0) cb_append(&cb, ", ");
            cb_append(&cb, outs[k]);
        }
        cb_append(&cb, " = run(");
        for (int i = 0; i < inputs.count; i++) {
            if (i > 0) cb_append(&cb, ", ");
            cb_append(&cb, inputs.names[i]);
        }
        if (emit_trace) cb_append(&cb, ", _trace=_tr");
        cb_append(&cb, ")\n");
        for (int k = 0; k < num_outs; k++) {
            cb_indent(&cb);
            cb_appendf(&cb, "print(%s)\n", outs[k]);
        }
        if (emit_trace) {
            cb_indent(&cb); cb_append(&cb, "import json\n");
            cb_indent(&cb);
            cb_appendf(&cb,
                "json.dump({\"schema\": \"edgecc-trace/1\", \"kernel\": \"%s\", "
                "\"dims\": {\"N\": ", kernel_name);
            /* N is the watched tensor's length. Returned, it holds every
               generation; otherwise __main__ never sees it, so read it from
               the trace's last step, which records every generational tensor. */
            if (name_in_list(watch, outs, num_outs))
                cb_appendf(&cb, "len(%s[0])", watch);
            else
                cb_appendf(&cb, "len(_tr[-1][\"tensors\"][\"%s\"])", watch);
            cb_append(&cb, "}, \"gen_tensors\": [");
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

    free(load_path_of);
    free(file_in_names);
    free(file_in_paths);
    free(outs);
    free_tensor_name_list(&inputs);
    free_tensor_name_list(&written);
    return cb.buf;
}
