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
        dim_names[d] = NULL;
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
    if (load1->data.load.gen_read)
        snprintf(a_name, sizeof(a_name), "%s_prev", load1->data.load.tensor_name);
    else
        snprintf(a_name, sizeof(a_name), "%s", load1->data.load.tensor_name);
    if (load2->data.load.gen_read)
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

    /* MAP existence */
    cb_indent(cb); cb_appendf(cb, "a_exists = a_val != %s\n", eA);
    cb_indent(cb); cb_appendf(cb, "b_exists = b_val != %s\n", eB);

    cb_indent(cb); cb_append(cb, "if ");
    emit_merge_guard(cb, map_i->data.map.merge_op, "a_exists", "b_exists");
    cb_append(cb, ":\n");
    cb->indent++;
    cb_indent(cb); cb_appendf(cb, "a_op = a_val if a_exists else %s\n", eA);
    cb_indent(cb); cb_appendf(cb, "b_op = b_val if b_exists else %s\n", eB);

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

char *codegen_python(IRProgram *ir) {
    if (!ir) return NULL;
    CodeBuf cb;
    cb_init(&cb);

    cb_append(&cb, "# Generated by edgecc\n");
    cb_append(&cb, "import math\n\n");

    /* Module-level dim constants */
    for (int i = 0; i < ir->count; i++) {
        if (ir->instrs[i].type == IR_DIM_DEF) {
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
    const char *gen_tensor = NULL;
    const char *intermediates[8];
    int num_intermediates = 0;
    for (int i = 0; i < ir->count; i++) {
        if (ir->instrs[i].type == IR_GEN_LOOP_BEGIN) {
            iterative = 1;
            gen_tensor = ir->instrs[i].data.gen_loop.cross_gens[0].name;
            num_intermediates = ir->instrs[i].data.gen_loop.num_intermediates;
            for (int j = 0; j < num_intermediates && j < 8; j++)
                intermediates[j] = ir->instrs[i].data.gen_loop.intermediates[j].name;
            break;
        }
    }

    /* Cascade I/O: intermediates are neither parameters nor the return value;
       the returned tensor is the cross-generational one. */
    if (iterative) {
        TensorNameList filtered;
        filtered.count = 0;
        for (int i = 0; i < inputs.count; i++) {
            int is_inter = 0;
            for (int j = 0; j < num_intermediates; j++)
                if (strcmp(inputs.names[i], intermediates[j]) == 0) is_inter = 1;
            if (is_inter) { free(inputs.names[i]); continue; }
            filtered.names[filtered.count++] = inputs.names[i];
        }
        inputs = filtered;
        if (gen_tensor) {
            if (output) free(output);
            output = strdup(gen_tensor);
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
    cb_append(&cb, "}\n\n");

    /* def run(inputs) */
    cb_append(&cb, "def run(");
    for (int i = 0; i < inputs.count; i++) {
        if (i > 0) cb_append(&cb, ", ");
        cb_append(&cb, inputs.names[i]);
    }
    cb_append(&cb, "):\n");
    cb.indent = 1;

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
                    !(iterative && gen_tensor &&
                      strcmp(instr->data.tensor_alloc.name, gen_tensor) == 0)) {
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
                const char *dim_name = find_dim_name(ir, i, range);
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
                cb_indent(&cb);
                cb_appendf(&cb, "%s_prev = %s\n", gloop->cross_gens[0].name, gloop->cross_gens[0].name);
                cb_indent(&cb);
                cb_append(&cb, "_it = 0\n");
                cb_indent(&cb);
                cb_append(&cb, "while True:\n");
                cb.indent++;
                /* cross-generational next slice (fresh each generation) */
                {
                    IRTensorAlloc tmp;
                    memset(&tmp, 0, sizeof(tmp));
                    tmp.ndims = gloop->cross_gens[0].spatial_ndims;
                    tmp.empty_val = gloop->cross_gens[0].empty_val;
                    for (int k = 0; k < tmp.ndims; k++)
                        tmp.dim_sizes[k] = gloop->cross_gens[0].spatial_dims[k];
                    cb_indent(&cb);
                    cb_appendf(&cb, "%s_next = ", gloop->cross_gens[0].name);
                    emit_alloc(&cb, &tmp, ir, i);
                    cb_append(&cb, "\n");
                }
                /* each intra-generation intermediate (fresh each generation) */
                for (int m = 0; m < gloop->num_intermediates; m++) {
                    IRTensorAlloc tmp;
                    memset(&tmp, 0, sizeof(tmp));
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
                const char *t = gen_tensor;
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
                cb.indent--;   /* close the while body */
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
        }
    }

    if (output) {
        cb_indent(&cb);
        if (iterative && gen_tensor) {
            cb_appendf(&cb, "return %s_prev\n", gen_tensor);
        } else {
            cb_appendf(&cb, "return %s\n", output);
        }
    }
    cb.indent = 0;
    cb_append(&cb, "\n");

    /* __main__ block: re-emit allocs+inits+call */
    cb_append(&cb, "if __name__ == \"__main__\":\n");
    cb.indent = 1;
    /* Emit allocs for INPUT tensors only (the output gets recomputed by run) */
    for (int i = 0; i < ir->count; i++) {
        IRInstr *instr = &ir->instrs[i];
        if (instr->type != IR_TENSOR_ALLOC) continue;
        int is_input = 0;
        for (int j = 0; j < inputs.count; j++) {
            if (strcmp(instr->data.tensor_alloc.name, inputs.names[j]) == 0) { is_input = 1; break; }
        }
        if (!is_input) continue;
        if (iterative && gen_tensor &&
            strcmp(instr->data.tensor_alloc.name, gen_tensor) == 0) continue;
        cb_indent(&cb);
        cb_appendf(&cb, "%s = ", instr->data.tensor_alloc.name);
        emit_alloc(&cb, &instr->data.tensor_alloc, ir, i);
        cb_append(&cb, "\n");
    }
    /* Emit init statements (inputs only — output is initialized inside run) */
    for (int i = 0; i < ir->count; i++) {
        IRInstr *instr = &ir->instrs[i];
        if (instr->type != IR_TENSOR_INIT) continue;
        int is_input = 0;
        for (int j = 0; j < inputs.count; j++) {
            if (strcmp(instr->data.tensor_init.name, inputs.names[j]) == 0) { is_input = 1; break; }
        }
        if (!is_input) continue;
        cb_indent(&cb);
        cb_appendf(&cb, "%s = ", instr->data.tensor_init.name);
        emit_literal(&cb, instr->data.tensor_init.data);
        cb_append(&cb, "\n");
    }

    /* Call run and print */
    if (output) {
        cb_indent(&cb);
        cb_appendf(&cb, "%s = run(", output);
        for (int i = 0; i < inputs.count; i++) {
            if (i > 0) cb_append(&cb, ", ");
            cb_append(&cb, inputs.names[i]);
        }
        cb_append(&cb, ")\n");
        cb_indent(&cb);
        cb_appendf(&cb, "print(%s)\n", output);
    }

    free(output);
    free_tensor_name_list(&inputs);
    return cb.buf;
}
