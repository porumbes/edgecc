#include "ir.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

/* ---- Internal helpers ---- */

static void ir_grow(IRProgram *ir) {
    if (ir->count >= ir->capacity) {
        ir->capacity = ir->capacity ? ir->capacity * 2 : 16;
        ir->instrs = realloc(ir->instrs, ir->capacity * sizeof(IRInstr));
    }
}

static IRInstr *ir_emit(IRProgram *ir, IRType type) {
    ir_grow(ir);
    IRInstr *instr = &ir->instrs[ir->count++];
    memset(instr, 0, sizeof(IRInstr));
    instr->type = type;
    return instr;
}

static char *dup_str(const char *s) {
    if (!s) return NULL;
    size_t len = strlen(s);
    char *d = malloc(len + 1);
    memcpy(d, s, len + 1);
    return d;
}

static char **dup_str_array(char **src, int count) {
    if (!src || count == 0) return NULL;
    char **dst = malloc(count * sizeof(char *));
    for (int i = 0; i < count; i++) {
        dst[i] = dup_str(src[i]);
    }
    return dst;
}

/* Duplicate coords, omitting index `strip_axis` (pass -1 to keep all). */
static char **dup_coords_strip(char **src, int count, int strip_axis, int *out_count) {
    int n = 0;
    for (int i = 0; i < count; i++) if (i != strip_axis) n++;
    *out_count = n;
    if (n == 0) return NULL;
    char **dst = malloc(n * sizeof(char *));
    int j = 0;
    for (int i = 0; i < count; i++) {
        if (i == strip_axis) continue;
        dst[j++] = dup_str(src[i]);
    }
    return dst;
}

/* ---- Emit phases ---- */

static void emit_dim_defs(IRProgram *ir, SymbolTable *symtab) {
    for (int i = 0; i < symtab->num_dims; i++) {
        IRInstr *instr = ir_emit(ir, IR_DIM_DEF);
        instr->data.dim_def.name = dup_str(symtab->dims[i].name);
        instr->data.dim_def.size = symtab->dims[i].size;
    }
}

static void emit_tensor_allocs(IRProgram *ir, AstNode *program, SymbolTable *symtab) {
    for (int i = 0; i < program->data.program.count; i++) {
        AstNode *stmt = program->data.program.statements[i];
        if (stmt->type != NODE_TENSOR_DECL) continue;

        const char *tname = stmt->data.tensor_decl.name;

        /* Find matching tensor entry in symtab */
        TensorEntry *te = NULL;
        for (int j = 0; j < symtab->num_tensors; j++) {
            if (strcmp(symtab->tensors[j].name, tname) == 0) {
                te = &symtab->tensors[j];
                break;
            }
        }

        IRInstr *instr = ir_emit(ir, IR_TENSOR_ALLOC);
        instr->data.tensor_alloc.name = dup_str(tname);
        instr->data.tensor_alloc.ndims = stmt->data.tensor_decl.num_ranks;
        instr->data.tensor_alloc.empty_val = stmt->data.tensor_decl.empty_val;
        if (te) {
            for (int d = 0; d < te->num_ranks && d < 8; d++) {
                instr->data.tensor_alloc.dim_sizes[d] = te->dim_sizes[d];
            }
        }
    }
}

static void emit_tensor_inits(IRProgram *ir, AstNode *program) {
    for (int i = 0; i < program->data.program.count; i++) {
        AstNode *stmt = program->data.program.statements[i];
        if (stmt->type != NODE_INIT) continue;

        IRInstr *instr = ir_emit(ir, IR_TENSOR_INIT);
        instr->data.tensor_init.name = dup_str(stmt->data.init.tensor_name);
        instr->data.tensor_init.data = stmt->data.init.data;  /* borrowed */
    }
}

static void emit_einsum(IRProgram *ir, AstNode *einsum_node, SymbolTable *symtab) {
    EinsumData *e = &einsum_node->data.einsum;

    /* Resolve the generational tensor's axis per operand (-1 if not the gen tensor). */
    int out_gen_axis = -1, in1_gen_axis = -1, in2_gen_axis = -1;
    if (symtab->is_iterative && symtab->gen_tensor) {
        TensorEntry *gte = symtab_find_tensor(symtab, symtab->gen_tensor);
        int gax = gte ? gte->gen_axis : -1;
        if (strcmp(e->output.tensor_name, symtab->gen_tensor) == 0) out_gen_axis = gax;
        if (strcmp(e->input1.tensor_name, symtab->gen_tensor) == 0) in1_gen_axis = gax;
        if (strcmp(e->input2.tensor_name, symtab->gen_tensor) == 0) in2_gen_axis = gax;

        IRInstr *gl = ir_emit(ir, IR_GEN_LOOP_BEGIN);
        gl->data.gen_loop.tensor_name = dup_str(symtab->gen_tensor);
        gl->data.gen_loop.empty_val = gte ? gte->empty_val : 0.0;
        gl->data.gen_loop.max_gen = 1000;
        int sn = 0;
        for (int i = 0; gte && i < gte->num_ranks; i++) {
            if (i == gte->gen_axis) continue;
            gl->data.gen_loop.spatial_dims[sn++] = gte->dim_sizes[i];
        }
        gl->data.gen_loop.spatial_ndims = sn;
    }

    /* Spatial loops: free indices, then reduction indices (gen rank excluded
       by semantic analysis). */
    for (int i = 0; i < symtab->num_free_indices; i++) {
        IRInstr *instr = ir_emit(ir, IR_LOOP_BEGIN);
        instr->data.loop_begin.index_var = dup_str(symtab->free_indices[i].name);
        instr->data.loop_begin.range = symtab->free_indices[i].size;
    }
    for (int i = 0; i < symtab->num_reduction_indices; i++) {
        IRInstr *instr = ir_emit(ir, IR_LOOP_BEGIN);
        instr->data.loop_begin.index_var = dup_str(symtab->reduction_indices[i].name);
        instr->data.loop_begin.range = symtab->reduction_indices[i].size;
    }

    /* LOAD input1 -> reg 0 */
    {
        IRInstr *instr = ir_emit(ir, IR_LOAD);
        instr->data.load.reg = 0;
        instr->data.load.tensor_name = dup_str(e->input1.tensor_name);
        instr->data.load.coords = dup_coords_strip(e->input1.rank_vars,
            e->input1.num_vars, in1_gen_axis, &instr->data.load.ncoords);
        instr->data.load.gen_read = (in1_gen_axis >= 0);
    }

    /* LOAD input2 -> reg 1 */
    {
        IRInstr *instr = ir_emit(ir, IR_LOAD);
        instr->data.load.reg = 1;
        instr->data.load.tensor_name = dup_str(e->input2.tensor_name);
        instr->data.load.coords = dup_coords_strip(e->input2.rank_vars,
            e->input2.num_vars, in2_gen_axis, &instr->data.load.ncoords);
        instr->data.load.gen_read = (in2_gen_axis >= 0);
    }

    /* MAP: reg 2 = map_op(reg 0, reg 1) */
    {
        IRInstr *instr = ir_emit(ir, IR_MAP);
        instr->data.map.dst_reg = 2;
        instr->data.map.op = e->map_op;
        instr->data.map.merge_op = e->map_merge_op;
        instr->data.map.src_reg1 = 0;
        instr->data.map.src_reg2 = 1;
        instr->data.map.empty_a = symtab_find_tensor(symtab, e->input1.tensor_name)->empty_val;
        instr->data.map.empty_b = symtab_find_tensor(symtab, e->input2.tensor_name)->empty_val;
        instr->data.map.empty_z = symtab_find_tensor(symtab, e->output.tensor_name)->empty_val;
    }

    /* REDUCE: reg 3 = reduce_op(reg 2, reg 3) */
    {
        TensorEntry *zent = symtab_find_tensor(symtab, e->output.tensor_name);
        IRInstr *instr = ir_emit(ir, IR_REDUCE);
        instr->data.reduce.dst_reg = 3;
        instr->data.reduce.op = e->reduce_op;
        instr->data.reduce.merge_op = e->reduce_merge_op;
        instr->data.reduce.src_reg = 2;
        instr->data.reduce.acc_reg = 3;
        instr->data.reduce.identity = compute_op_identity(e->reduce_op, zent->datatype);
        instr->data.reduce.empty_z  = zent->empty_val;
    }

    /* STORE: output[coords] = reg 3 */
    {
        IRInstr *instr = ir_emit(ir, IR_STORE);
        instr->data.store.tensor_name = dup_str(e->output.tensor_name);
        instr->data.store.coords = dup_coords_strip(e->output.rank_vars,
            e->output.num_vars, out_gen_axis, &instr->data.store.ncoords);
        instr->data.store.src_reg = 3;
        instr->data.store.gen_write = (out_gen_axis >= 0);
    }

    /* LOOP_END in reverse order: reduction first, then free */
    for (int i = symtab->num_reduction_indices - 1; i >= 0; i--) {
        ir_emit(ir, IR_LOOP_END);
    }
    for (int i = symtab->num_free_indices - 1; i >= 0; i--) {
        ir_emit(ir, IR_LOOP_END);
    }

    if (symtab->is_iterative) {
        ir_emit(ir, IR_GEN_LOOP_END);
    }
}

/* ---- Public API ---- */

IRProgram *ir_lower(AstNode *program, SymbolTable *symtab) {
    if (!program || program->type != NODE_PROGRAM) return NULL;

    IRProgram *ir = calloc(1, sizeof(IRProgram));

    emit_dim_defs(ir, symtab);
    emit_tensor_allocs(ir, program, symtab);
    emit_tensor_inits(ir, program);

    for (int i = 0; i < program->data.program.count; i++) {
        AstNode *stmt = program->data.program.statements[i];
        if (stmt->type == NODE_EINSUM) {
            emit_einsum(ir, stmt, symtab);
        }
    }

    return ir;
}

void ir_free(IRProgram *ir) {
    if (!ir) return;

    for (int i = 0; i < ir->count; i++) {
        IRInstr *instr = &ir->instrs[i];
        switch (instr->type) {
            case IR_DIM_DEF:
                free(instr->data.dim_def.name);
                break;
            case IR_TENSOR_ALLOC:
                free(instr->data.tensor_alloc.name);
                break;
            case IR_TENSOR_INIT:
                free(instr->data.tensor_init.name);
                /* data is borrowed, not freed */
                break;
            case IR_LOOP_BEGIN:
                free(instr->data.loop_begin.index_var);
                break;
            case IR_LOOP_END:
                break;
            case IR_GEN_LOOP_BEGIN:
                free(instr->data.gen_loop.tensor_name);
                break;
            case IR_GEN_LOOP_END:
                break;
            case IR_LOAD:
                free(instr->data.load.tensor_name);
                for (int j = 0; j < instr->data.load.ncoords; j++) {
                    free(instr->data.load.coords[j]);
                }
                free(instr->data.load.coords);
                break;
            case IR_MAP:
                break;
            case IR_REDUCE:
                break;
            case IR_STORE:
                free(instr->data.store.tensor_name);
                for (int j = 0; j < instr->data.store.ncoords; j++) {
                    free(instr->data.store.coords[j]);
                }
                free(instr->data.store.coords);
                break;
        }
    }

    free(ir->instrs);
    free(ir);
}

static void dump_coords(FILE *out, char **coords, int n) {
    fprintf(out, "[");
    for (int i = 0; i < n; i++) {
        if (i > 0) fprintf(out, ", ");
        fprintf(out, "%s", coords[i]);
    }
    fprintf(out, "]");
}

void ir_dump(IRProgram *ir, FILE *out) {
    for (int i = 0; i < ir->count; i++) {
        IRInstr *instr = &ir->instrs[i];
        switch (instr->type) {
            case IR_DIM_DEF:
                fprintf(out, "DIM_DEF(%s, %d)\n",
                        instr->data.dim_def.name, instr->data.dim_def.size);
                break;
            case IR_TENSOR_ALLOC:
                fprintf(out, "TENSOR_ALLOC(%s, dims=[", instr->data.tensor_alloc.name);
                for (int j = 0; j < instr->data.tensor_alloc.ndims; j++) {
                    if (j > 0) fprintf(out, ",");
                    fprintf(out, "%d", instr->data.tensor_alloc.dim_sizes[j]);
                }
                fprintf(out, "], empty=%g)\n", instr->data.tensor_alloc.empty_val);
                break;
            case IR_TENSOR_INIT:
                fprintf(out, "TENSOR_INIT(%s, ...)\n", instr->data.tensor_init.name);
                break;
            case IR_LOOP_BEGIN:
                fprintf(out, "LOOP_BEGIN(%s, %d)\n",
                        instr->data.loop_begin.index_var, instr->data.loop_begin.range);
                break;
            case IR_LOOP_END:
                fprintf(out, "LOOP_END\n");
                break;
            case IR_LOAD:
                fprintf(out, "LOAD(r%d, %s%s, ", instr->data.load.reg,
                        instr->data.load.tensor_name,
                        instr->data.load.gen_read ? " [gen_read]" : "");
                dump_coords(out, instr->data.load.coords, instr->data.load.ncoords);
                fprintf(out, ")\n");
                break;
            case IR_MAP:
                fprintf(out, "  MAP %s %s (eA=%g, eB=%g, eZ=%g) r%d, r%d -> r%d\n",
                        operator_to_str(instr->data.map.op),
                        merge_op_to_str(instr->data.map.merge_op),
                        instr->data.map.empty_a,
                        instr->data.map.empty_b,
                        instr->data.map.empty_z,
                        instr->data.map.src_reg1,
                        instr->data.map.src_reg2,
                        instr->data.map.dst_reg);
                break;
            case IR_REDUCE:
                fprintf(out, "  REDUCE %s %s (id=%g, eZ=%g) r%d -> r%d\n",
                        operator_to_str(instr->data.reduce.op),
                        merge_op_to_str(instr->data.reduce.merge_op),
                        instr->data.reduce.identity,
                        instr->data.reduce.empty_z,
                        instr->data.reduce.src_reg,
                        instr->data.reduce.dst_reg);
                break;
            case IR_STORE:
                fprintf(out, "STORE(%s%s, ", instr->data.store.tensor_name,
                        instr->data.store.gen_write ? " [gen_write]" : "");
                dump_coords(out, instr->data.store.coords, instr->data.store.ncoords);
                fprintf(out, ", r%d)\n", instr->data.store.src_reg);
                break;
            case IR_GEN_LOOP_BEGIN:
                fprintf(out, "GEN_LOOP_BEGIN(%s, spatial=[",
                        instr->data.gen_loop.tensor_name);
                for (int j = 0; j < instr->data.gen_loop.spatial_ndims; j++) {
                    if (j > 0) fprintf(out, ",");
                    fprintf(out, "%d", instr->data.gen_loop.spatial_dims[j]);
                }
                fprintf(out, "], empty=%g, max_gen=%d)\n",
                        instr->data.gen_loop.empty_val,
                        instr->data.gen_loop.max_gen);
                break;
            case IR_GEN_LOOP_END:
                fprintf(out, "GEN_LOOP_END\n");
                break;
        }
    }
}
