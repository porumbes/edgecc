#include "codegen.h"
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
                double v = data->data.scalar.float_val;
                if (v == (int)v) {
                    cb_appendf(cb, "%g", v);
                } else {
                    cb_appendf(cb, "%g", v);
                }
                break;
            }
            case TYPE_BOOLEAN:
                cb_append(cb, data->data.scalar.bool_val ? "True" : "False");
                break;
        }
    }
}

/* Get Python infix operator string, or NULL if it's a function-style op */
static const char *op_infix(OperatorType op) {
    switch (op) {
        case OP_ADD: return " + ";
        case OP_SUB: return " - ";
        case OP_MUL: return " * ";
        default: return NULL;
    }
}

/* Get Python function-style op name for min/max */
static const char *op_func(OperatorType op) {
    switch (op) {
        case OP_MIN: return "min";
        case OP_MAX: return "max";
        default: return NULL;
    }
}

/* Check if op is boolean */
static int op_is_bool(OperatorType op) {
    return op == OP_AND || op == OP_OR;
}

static const char *op_bool_str(OperatorType op) {
    switch (op) {
        case OP_AND: return " and ";
        case OP_OR:  return " or ";
        default: return NULL;
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

/* Emit a tensor alloc as nested list comprehension.
   e.g. for dims [2,3] with empty_val 0.0:
   [[0.0] * 3 for _ in range(2)]
   For 1D: [0.0] * N
*/
static void emit_alloc(CodeBuf *cb, IRTensorAlloc *alloc, IRProgram *ir, int instr_pos) {
    /* Format the empty value */
    char valstr[64];
    double v = alloc->empty_val;
    if (v == (int)v && v >= -1e9 && v <= 1e9) {
        snprintf(valstr, sizeof(valstr), "%.1f", v);
    } else {
        snprintf(valstr, sizeof(valstr), "%g", v);
    }

    if (alloc->ndims == 0) {
        cb_append(cb, valstr);
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
}

/* ---- Main codegen ---- */

char *codegen_python(IRProgram *ir) {
    if (!ir) return NULL;

    CodeBuf cb;
    cb_init(&cb);

    cb_append(&cb, "# Generated by edgecc\n\n");

    const char *last_store_tensor = NULL;

    for (int i = 0; i < ir->count; i++) {
        IRInstr *instr = &ir->instrs[i];

        switch (instr->type) {
            case IR_DIM_DEF:
                cb_appendf(&cb, "%s = %d\n",
                    instr->data.dim_def.name,
                    instr->data.dim_def.size);
                break;

            case IR_TENSOR_ALLOC:
                cb_appendf(&cb, "%s = ", instr->data.tensor_alloc.name);
                emit_alloc(&cb, &instr->data.tensor_alloc, ir, i);
                cb_append(&cb, "\n");
                break;

            case IR_TENSOR_INIT:
                cb_appendf(&cb, "%s = ", instr->data.tensor_init.name);
                emit_literal(&cb, instr->data.tensor_init.data);
                cb_append(&cb, "\n");
                break;

            case IR_LOOP_BEGIN: {
                /* Emit blank line before the outermost loop */
                if (cb.indent == 0) {
                    cb_append(&cb, "\n");
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

            case IR_LOAD:
            case IR_MAP:
            case IR_REDUCE:
                /* These are combined into STORE */
                break;

            case IR_STORE: {
                last_store_tensor = instr->data.store.tensor_name;

                /* Look back for LOAD, LOAD, MAP, REDUCE */
                if (i < 4) break;
                IRInstr *load1 = &ir->instrs[i - 4];
                IRInstr *load2 = &ir->instrs[i - 3];
                IRInstr *map   = &ir->instrs[i - 2];
                IRInstr *red   = &ir->instrs[i - 1];

                if (load1->type != IR_LOAD || load2->type != IR_LOAD ||
                    map->type != IR_MAP || red->type != IR_REDUCE) break;

                cb_indent(&cb);

                /* output[coords] = reduce_op(output[coords], map_op(input1[coords], input2[coords])) */
                OperatorType map_op = map->data.map.op;
                OperatorType red_op = red->data.reduce.op;

                /* LHS: Z[m][n] = */
                emit_tensor_access(&cb,
                    instr->data.store.tensor_name,
                    instr->data.store.coords,
                    instr->data.store.ncoords);
                cb_append(&cb, " = ");

                /* Build the map expression string */
                /* map_expr = "input1[coords] op input2[coords]" */

                /* Build reduce expression:
                   infix reduce:  output[coords] + map_expr
                   func reduce:   min(output[coords], map_expr)
                   bool reduce:   output[coords] or (map_expr)
                */

                const char *red_infix = op_infix(red_op);
                const char *red_func = op_func(red_op);
                int red_bool = op_is_bool(red_op);

                if (red_func) {
                    /* min(output[coords], map_expr) */
                    cb_appendf(&cb, "%s(", red_func);
                    emit_tensor_access(&cb,
                        instr->data.store.tensor_name,
                        instr->data.store.coords,
                        instr->data.store.ncoords);
                    cb_append(&cb, ", ");
                } else if (red_bool) {
                    /* output[coords] or ( */
                    emit_tensor_access(&cb,
                        instr->data.store.tensor_name,
                        instr->data.store.coords,
                        instr->data.store.ncoords);
                    cb_append(&cb, op_bool_str(red_op));
                    cb_append(&cb, "(");
                } else {
                    /* output[coords] + */
                    emit_tensor_access(&cb,
                        instr->data.store.tensor_name,
                        instr->data.store.coords,
                        instr->data.store.ncoords);
                    cb_append(&cb, red_infix);
                }

                /* Now emit the map expression */
                const char *map_infix = op_infix(map_op);
                const char *map_func_name = op_func(map_op);
                int map_bool = op_is_bool(map_op);

                if (map_func_name) {
                    cb_appendf(&cb, "%s(", map_func_name);
                    emit_tensor_access(&cb,
                        load1->data.load.tensor_name,
                        load1->data.load.coords,
                        load1->data.load.ncoords);
                    cb_append(&cb, ", ");
                    emit_tensor_access(&cb,
                        load2->data.load.tensor_name,
                        load2->data.load.coords,
                        load2->data.load.ncoords);
                    cb_append(&cb, ")");
                } else if (map_bool) {
                    emit_tensor_access(&cb,
                        load1->data.load.tensor_name,
                        load1->data.load.coords,
                        load1->data.load.ncoords);
                    cb_append(&cb, op_bool_str(map_op));
                    emit_tensor_access(&cb,
                        load2->data.load.tensor_name,
                        load2->data.load.coords,
                        load2->data.load.ncoords);
                } else {
                    emit_tensor_access(&cb,
                        load1->data.load.tensor_name,
                        load1->data.load.coords,
                        load1->data.load.ncoords);
                    cb_append(&cb, map_infix);
                    emit_tensor_access(&cb,
                        load2->data.load.tensor_name,
                        load2->data.load.coords,
                        load2->data.load.ncoords);
                }

                /* Close parens for func/bool reduce */
                if (red_func || red_bool) {
                    cb_append(&cb, ")");
                }

                cb_append(&cb, "\n");
                break;
            }
        }
    }

    /* Emit trailing blank line and print */
    if (last_store_tensor) {
        cb_appendf(&cb, "\nprint(%s)\n", last_store_tensor);
    }

    return cb.buf;
}
