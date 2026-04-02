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

    /* Emit LOOP_BEGIN for free indices first, then reduction indices */
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
        instr->data.load.ncoords = e->input1.num_vars;
        instr->data.load.coords = dup_str_array(e->input1.rank_vars, e->input1.num_vars);
    }

    /* LOAD input2 -> reg 1 */
    {
        IRInstr *instr = ir_emit(ir, IR_LOAD);
        instr->data.load.reg = 1;
        instr->data.load.tensor_name = dup_str(e->input2.tensor_name);
        instr->data.load.ncoords = e->input2.num_vars;
        instr->data.load.coords = dup_str_array(e->input2.rank_vars, e->input2.num_vars);
    }

    /* MAP: reg 2 = map_op(reg 0, reg 1) */
    {
        IRInstr *instr = ir_emit(ir, IR_MAP);
        instr->data.map.dst_reg = 2;
        instr->data.map.op = e->map_op;
        instr->data.map.src_reg1 = 0;
        instr->data.map.src_reg2 = 1;
    }

    /* REDUCE: reg 3 = reduce_op(reg 2, reg 3) */
    {
        IRInstr *instr = ir_emit(ir, IR_REDUCE);
        instr->data.reduce.dst_reg = 3;
        instr->data.reduce.op = e->reduce_op;
        instr->data.reduce.src_reg = 2;
        instr->data.reduce.acc_reg = 3;
    }

    /* STORE: output[coords] = reg 3 */
    {
        IRInstr *instr = ir_emit(ir, IR_STORE);
        instr->data.store.tensor_name = dup_str(e->output.tensor_name);
        instr->data.store.ncoords = e->output.num_vars;
        instr->data.store.coords = dup_str_array(e->output.rank_vars, e->output.num_vars);
        instr->data.store.src_reg = 3;
    }

    /* LOOP_END in reverse order: reduction first, then free */
    for (int i = symtab->num_reduction_indices - 1; i >= 0; i--) {
        ir_emit(ir, IR_LOOP_END);
    }
    for (int i = symtab->num_free_indices - 1; i >= 0; i--) {
        ir_emit(ir, IR_LOOP_END);
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
                fprintf(out, "LOAD(r%d, %s, ", instr->data.load.reg,
                        instr->data.load.tensor_name);
                dump_coords(out, instr->data.load.coords, instr->data.load.ncoords);
                fprintf(out, ")\n");
                break;
            case IR_MAP:
                fprintf(out, "MAP(r%d, %s, r%d, r%d)\n",
                        instr->data.map.dst_reg, operator_to_str(instr->data.map.op),
                        instr->data.map.src_reg1, instr->data.map.src_reg2);
                break;
            case IR_REDUCE:
                fprintf(out, "REDUCE(r%d, %s, r%d, r%d)\n",
                        instr->data.reduce.dst_reg, operator_to_str(instr->data.reduce.op),
                        instr->data.reduce.src_reg, instr->data.reduce.acc_reg);
                break;
            case IR_STORE:
                fprintf(out, "STORE(%s, ", instr->data.store.tensor_name);
                dump_coords(out, instr->data.store.coords, instr->data.store.ncoords);
                fprintf(out, ", r%d)\n", instr->data.store.src_reg);
                break;
        }
    }
}
