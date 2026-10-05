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
        instr->data.dim_def.is_inferred = symtab->dims[i].is_inferred;
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
                const char *dn = te->ranks[d].dim_name;
                instr->data.tensor_alloc.dim_names[d] = dn ? dup_str(dn) : NULL;
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
        instr->data.tensor_init.data = stmt->data.init.data;  /* borrowed (NULL when load) */
        instr->data.tensor_init.is_load = stmt->data.init.is_load;
        instr->data.tensor_init.path = stmt->data.init.is_load
                                     ? dup_str(stmt->data.init.path) : NULL;
        instr->data.tensor_init.is_seed = stmt->data.init.is_seed;
        instr->data.tensor_init.seed_source = stmt->data.init.seed_source;
        instr->data.tensor_init.seed_value = stmt->data.init.seed_value;
    }
}

static int is_cross_gen_name(SymbolTable *st, const char *name) {
    for (int m = 0; m < st->num_gen_tensors; m++)
        if (strcmp(st->gen_tensors[m], name) == 0) return 1;
    return 0;
}

static int is_intermediate_name(SymbolTable *st, const char *name) {
    for (int m = 0; m < st->num_intermediates; m++)
        if (strcmp(st->intermediates[m], name) == 0) return 1;
    return 0;
}

/* Generation kind of a reference: 0 not generational; 1 generation i;
   2 generation i+1. Applies to cross-generational tensors and intermediates
   alike -- every generational tensor stores all of its generations. */
static int gen_kind(SymbolTable *st, TensorRef *ref, TensorEntry *te) {
    if (!te || te->gen_axis < 0) return 0;
    if (!is_cross_gen_name(st, te->name) && !is_intermediate_name(st, te->name)) return 0;
    int eff = (ref->gen_axis == te->gen_axis) ? ref->gen_offset : 0;
    return (eff == 1) ? 2 : 1;
}

/* Find the dim name bounding a loop index by locating that rank variable in one
   of the Einsum's tensor refs and reading the corresponding axis's dim name.
   Rank vars align positionally with the tensor's declared ranks; the
   generational axis has a NULL dim name, so we keep searching past it. */
static const char *dim_name_for_index(EinsumData *e,
                                      TensorEntry *out_te,
                                      TensorEntry *in1_te,
                                      TensorEntry *in2_te,
                                      const char *idx) {
    TensorRef *refs[3]  = { &e->output, &e->input1, &e->input2 };
    TensorEntry *tes[3] = { out_te, in1_te, in2_te };
    for (int c = 0; c < 3; c++) {
        TensorRef *r = refs[c];
        TensorEntry *te = tes[c];
        if (!te) continue;
        for (int a = 0; a < r->num_vars && a < te->num_ranks; a++) {
            if (strcmp(r->rank_vars[a], idx) == 0 && te->ranks[a].dim_name)
                return te->ranks[a].dim_name;
        }
    }
    return NULL;
}

/* Like dim_name_for_index, but scans the four fused tensor refs. */
static const char *dim_name_for_index_fused(EinsumFusedData *f,
                                            TensorEntry *out_te,
                                            TensorEntry *i1_te,
                                            TensorEntry *i2_te,
                                            TensorEntry *ou_te,
                                            const char *idx) {
    TensorRef *refs[4]  = { &f->output, &f->inner1, &f->inner2, &f->outer };
    TensorEntry *tes[4] = { out_te, i1_te, i2_te, ou_te };
    for (int c = 0; c < 4; c++) {
        TensorRef *r = refs[c];
        TensorEntry *te = tes[c];
        if (!te) continue;
        for (int a = 0; a < r->num_vars && a < te->num_ranks; a++) {
            if (strcmp(r->rank_vars[a], idx) == 0 && te->ranks[a].dim_name)
                return te->ranks[a].dim_name;
        }
    }
    return NULL;
}

static void fill_fused_operand(IRFusedOperand *o, TensorRef *r, TensorEntry *te, SymbolTable *st) {
    o->name = dup_str(r->tensor_name);
    o->coords = dup_coords_strip(r->rank_vars, r->num_vars, te->gen_axis, &o->ncoords);
    o->empty = te->empty_val;
    o->negate = r->negate;
    o->gen_read = gen_kind(st, r, te);
}

static void emit_einsum_fused(IRProgram *ir, AstNode *node, SymbolTable *st, int eidx) {
    EinsumFusedData *f = &node->data.einsum_fused;
    EinsumClass *ec = &st->einsum_class[eidx];
    TensorEntry *out_te = symtab_find_tensor(st, f->output.tensor_name);
    TensorEntry *i1 = symtab_find_tensor(st, f->inner1.tensor_name);
    TensorEntry *i2 = symtab_find_tensor(st, f->inner2.tensor_name);
    TensorEntry *ou = symtab_find_tensor(st, f->outer.tensor_name);

    IRInstr *instr = ir_emit(ir, IR_FUSED);
    IRFused *fu = &instr->data.fused;
    /* ir_emit already memset the instr to 0 */
    fu->out_name = dup_str(f->output.tensor_name);
    fu->out_coords = dup_coords_strip(f->output.rank_vars, f->output.num_vars, out_te->gen_axis, &fu->out_ncoords);
    fu->empty_z = out_te->empty_val;
    fu->gen_write = gen_kind(st, &f->output, out_te);
    fill_fused_operand(&fu->inner1, &f->inner1, i1, st);
    fill_fused_operand(&fu->inner2, &f->inner2, i2, st);
    fill_fused_operand(&fu->outer,  &f->outer,  ou, st);
    fu->inter_is_left2 = f->inter_is_left2;
    fu->map1_op = f->map1_op; fu->map1_merge = f->map1_merge;
    fu->a1_has_reduce = f->a1_has_reduce; fu->reduce1_op = f->reduce1_op; fu->reduce1_merge = f->reduce1_merge;
    fu->reduce1_identity = compute_op_identity(f->reduce1_op, out_te->datatype);
    fu->map2_op = f->map2_op; fu->map2_merge = f->map2_merge;
    fu->a2_has_reduce = f->a2_has_reduce; fu->reduce2_op = f->reduce2_op; fu->reduce2_merge = f->reduce2_merge;
    fu->reduce2_identity = compute_op_identity(f->reduce2_op, out_te->datatype);

    for (int i = 0; i < ec->num_free_indices; i++) {
        fu->free_names[fu->num_free] = dup_str(ec->free_indices[i].name);
        fu->free_sizes[fu->num_free] = ec->free_indices[i].size;
        const char *dn = dim_name_for_index_fused(f, out_te, i1, i2, ou, ec->free_indices[i].name);
        fu->free_dimnames[fu->num_free] = dn ? dup_str(dn) : NULL;
        fu->num_free++;
    }
    IndexEntry *red = NULL; int nred = 0;
    if (ec->num_a1_reduce > 0) { red = ec->a1_reduce; nred = ec->num_a1_reduce; fu->reduce_on_action = 1; }
    else if (ec->num_a2_reduce > 0) { red = ec->a2_reduce; nred = ec->num_a2_reduce; fu->reduce_on_action = 2; }
    else { fu->reduce_on_action = 0; }
    for (int i = 0; i < nred; i++) {
        fu->red_names[fu->num_red] = dup_str(red[i].name);
        fu->red_sizes[fu->num_red] = red[i].size;
        const char *dn = dim_name_for_index_fused(f, out_te, i1, i2, ou, red[i].name);
        fu->red_dimnames[fu->num_red] = dn ? dup_str(dn) : NULL;
        fu->num_red++;
    }
}

static void emit_einsum(IRProgram *ir, AstNode *einsum_node, SymbolTable *symtab, int eidx) {
    EinsumData *e = &einsum_node->data.einsum;
    EinsumClass *cls = &symtab->einsum_class[eidx];

    TensorEntry *out_te = symtab_find_tensor(symtab, e->output.tensor_name);
    TensorEntry *in1_te = symtab_find_tensor(symtab, e->input1.tensor_name);
    TensorEntry *in2_te = symtab_find_tensor(symtab, e->input2.tensor_name);

    /* Spatial loops: this Einsum's free indices, then its reduction indices. */
    for (int i = 0; i < cls->num_free_indices; i++) {
        IRInstr *instr = ir_emit(ir, IR_LOOP_BEGIN);
        instr->data.loop_begin.index_var = dup_str(cls->free_indices[i].name);
        instr->data.loop_begin.range = cls->free_indices[i].size;
        const char *dn = dim_name_for_index(e, out_te, in1_te, in2_te,
                                            cls->free_indices[i].name);
        instr->data.loop_begin.range_name = dn ? dup_str(dn) : NULL;
    }
    for (int i = 0; i < cls->num_reduction_indices; i++) {
        IRInstr *instr = ir_emit(ir, IR_LOOP_BEGIN);
        instr->data.loop_begin.index_var = dup_str(cls->reduction_indices[i].name);
        instr->data.loop_begin.range = cls->reduction_indices[i].size;
        const char *dn = dim_name_for_index(e, out_te, in1_te, in2_te,
                                            cls->reduction_indices[i].name);
        instr->data.loop_begin.range_name = dn ? dup_str(dn) : NULL;
    }

    /* LOAD input1 -> reg 0 */
    {
        IRInstr *instr = ir_emit(ir, IR_LOAD);
        instr->data.load.reg = 0;
        instr->data.load.tensor_name = dup_str(e->input1.tensor_name);
        instr->data.load.coords = dup_coords_strip(e->input1.rank_vars,
            e->input1.num_vars, in1_te->gen_axis, &instr->data.load.ncoords);
        instr->data.load.gen_read = gen_kind(symtab, &e->input1, in1_te);
        instr->data.load.negate = e->input1.negate;
    }
    /* LOAD input2 -> reg 1 */
    {
        IRInstr *instr = ir_emit(ir, IR_LOAD);
        instr->data.load.reg = 1;
        instr->data.load.tensor_name = dup_str(e->input2.tensor_name);
        instr->data.load.coords = dup_coords_strip(e->input2.rank_vars,
            e->input2.num_vars, in2_te->gen_axis, &instr->data.load.ncoords);
        instr->data.load.gen_read = gen_kind(symtab, &e->input2, in2_te);
        instr->data.load.negate = e->input2.negate;
    }

    /* MAP: reg 2 = map_op(reg 0, reg 1) */
    {
        IRInstr *instr = ir_emit(ir, IR_MAP);
        instr->data.map.dst_reg = 2;
        instr->data.map.op = e->map_op;
        instr->data.map.merge_op = e->map_merge_op;
        instr->data.map.src_reg1 = 0;
        instr->data.map.src_reg2 = 1;
        instr->data.map.empty_a = in1_te->empty_val;
        instr->data.map.empty_b = in2_te->empty_val;
        instr->data.map.empty_z = out_te->empty_val;
    }

    /* REDUCE: reg 3 = reduce_op(reg 2, reg 3) */
    {
        IRInstr *instr = ir_emit(ir, IR_REDUCE);
        instr->data.reduce.dst_reg = 3;
        instr->data.reduce.op = e->reduce_op;
        instr->data.reduce.merge_op = e->reduce_merge_op;
        instr->data.reduce.src_reg = 2;
        instr->data.reduce.acc_reg = 3;
        instr->data.reduce.identity = compute_op_identity(e->reduce_op, out_te->datatype);
        instr->data.reduce.empty_z  = out_te->empty_val;
    }

    /* STORE: output[coords] = reg 3 */
    {
        IRInstr *instr = ir_emit(ir, IR_STORE);
        instr->data.store.tensor_name = dup_str(e->output.tensor_name);
        instr->data.store.coords = dup_coords_strip(e->output.rank_vars,
            e->output.num_vars, out_te->gen_axis, &instr->data.store.ncoords);
        instr->data.store.src_reg = 3;
        instr->data.store.gen_write = gen_kind(symtab, &e->output, out_te);
    }

    /* LOOP_END in reverse order: reduction first, then free. */
    for (int i = cls->num_reduction_indices - 1; i >= 0; i--) ir_emit(ir, IR_LOOP_END);
    for (int i = cls->num_free_indices - 1; i >= 0; i--) ir_emit(ir, IR_LOOP_END);
}

/* Fill an IRGenAlloc with a tensor's spatial shape (dims excluding gen axis). */
static void fill_gen_alloc(IRGenAlloc *a, TensorEntry *te) {
    a->name = dup_str(te->name);
    a->empty_val = te->empty_val;
    int sn = 0;
    for (int i = 0; i < te->num_ranks; i++) {
        if (i == te->gen_axis) continue;
        const char *dn = te->ranks[i].dim_name;
        a->spatial_dim_names[sn] = dn ? dup_str(dn) : NULL;
        a->spatial_dims[sn++] = te->dim_sizes[i];
    }
    a->spatial_ndims = sn;
}

/* ---- Public API ---- */

IRProgram *ir_lower(AstNode *program, SymbolTable *symtab) {
    if (!program || program->type != NODE_PROGRAM) return NULL;

    IRProgram *ir = calloc(1, sizeof(IRProgram));

    emit_dim_defs(ir, symtab);
    emit_tensor_allocs(ir, program, symtab);
    emit_tensor_inits(ir, program);

    if (symtab->is_iterative) {
        /* One generational loop wraps the whole cascade. */
        IRInstr *gl = ir_emit(ir, IR_GEN_LOOP_BEGIN);
        gl->data.gen_loop.max_gen = 1000;
        int nc = 0;
        for (int m = 0; m < symtab->num_gen_tensors && nc < MAX_GEN_TENSORS; m++) {
            TensorEntry *te = symtab_find_tensor(symtab, symtab->gen_tensors[m]);
            if (te) fill_gen_alloc(&gl->data.gen_loop.cross_gens[nc++], te);
        }
        gl->data.gen_loop.num_cross_gens = nc;
        gl->data.gen_loop.watch = symtab->stop_watch ? dup_str(symtab->stop_watch) : NULL;
        gl->data.gen_loop.stop_kind  = symtab->stop_kind;
        gl->data.gen_loop.stop_gen_a = symtab->stop_gen_a;
        gl->data.gen_loop.stop_gen_b = symtab->stop_gen_b;
        TensorEntry *wte = symtab->stop_watch ? symtab_find_tensor(symtab, symtab->stop_watch) : NULL;
        gl->data.gen_loop.watch_empty = wte ? wte->empty_val : 0.0;
        int ni = 0;
        for (int m = 0; m < symtab->num_intermediates && ni < MAX_GEN_TENSORS; m++) {
            TensorEntry *te = symtab_find_tensor(symtab, symtab->intermediates[m]);
            if (te) fill_gen_alloc(&gl->data.gen_loop.intermediates[ni++], te);
        }
        gl->data.gen_loop.num_intermediates = ni;

        int eidx = 0;
        for (int i = 0; i < program->data.program.count; i++) {
            AstNode *stmt = program->data.program.statements[i];
            if (stmt->type == NODE_EINSUM)
                emit_einsum(ir, stmt, symtab, eidx++);
            else if (stmt->type == NODE_EINSUM_FUSED)
                emit_einsum_fused(ir, stmt, symtab, eidx++);
        }

        ir_emit(ir, IR_GEN_LOOP_END);
    } else {
        int eidx = 0;
        for (int i = 0; i < program->data.program.count; i++) {
            AstNode *stmt = program->data.program.statements[i];
            if (stmt->type == NODE_EINSUM)
                emit_einsum(ir, stmt, symtab, eidx++);
            else if (stmt->type == NODE_EINSUM_FUSED)
                emit_einsum_fused(ir, stmt, symtab, eidx++);
        }
    }

    if (symtab->save_tensor) {
        IRInstr *s = ir_emit(ir, IR_SAVE);
        s->data.save.tensor_name = dup_str(symtab->save_tensor);
        s->data.save.path = dup_str(symtab->save_path);
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
                for (int d = 0; d < instr->data.tensor_alloc.ndims && d < 8; d++)
                    free(instr->data.tensor_alloc.dim_names[d]);
                break;
            case IR_TENSOR_INIT:
                free(instr->data.tensor_init.name);
                free(instr->data.tensor_init.path);  /* NULL-safe; borrowed .data not freed */
                break;
            case IR_LOOP_BEGIN:
                free(instr->data.loop_begin.index_var);
                free(instr->data.loop_begin.range_name);
                break;
            case IR_LOOP_END:
                break;
            case IR_GEN_LOOP_BEGIN:
                for (int c = 0; c < instr->data.gen_loop.num_cross_gens; c++) {
                    free(instr->data.gen_loop.cross_gens[c].name);
                    for (int k = 0; k < instr->data.gen_loop.cross_gens[c].spatial_ndims; k++)
                        free(instr->data.gen_loop.cross_gens[c].spatial_dim_names[k]);
                }
                free(instr->data.gen_loop.watch);
                for (int j = 0; j < instr->data.gen_loop.num_intermediates; j++) {
                    free(instr->data.gen_loop.intermediates[j].name);
                    for (int k = 0; k < instr->data.gen_loop.intermediates[j].spatial_ndims; k++)
                        free(instr->data.gen_loop.intermediates[j].spatial_dim_names[k]);
                }
                break;
            case IR_GEN_LOOP_END:
                break;
            case IR_SAVE:
                free(instr->data.save.tensor_name);
                free(instr->data.save.path);
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
            case IR_FUSED: {
                IRFused *fu = &instr->data.fused;
                free(fu->out_name);
                for (int j = 0; j < fu->out_ncoords; j++) free(fu->out_coords[j]);
                free(fu->out_coords);
                IRFusedOperand *ops[3] = { &fu->inner1, &fu->inner2, &fu->outer };
                for (int o = 0; o < 3; o++) {
                    free(ops[o]->name);
                    for (int j = 0; j < ops[o]->ncoords; j++) free(ops[o]->coords[j]);
                    free(ops[o]->coords);
                }
                for (int j = 0; j < fu->num_free; j++) {
                    free(fu->free_names[j]);
                    free(fu->free_dimnames[j]);
                }
                for (int j = 0; j < fu->num_red; j++) {
                    free(fu->red_names[j]);
                    free(fu->red_dimnames[j]);
                }
                break;
            }
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
                if (instr->data.tensor_init.is_load)
                    fprintf(out, "TENSOR_INIT(%s, load=%s)\n",
                            instr->data.tensor_init.name, instr->data.tensor_init.path);
                else
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
                fprintf(out, "LOAD(r%d, %s%s%s, ", instr->data.load.reg,
                        instr->data.load.tensor_name,
                        instr->data.load.negate ? " [neg]" : "",
                        instr->data.load.gen_read == 2 ? " [gen i+1]" : instr->data.load.gen_read == 1 ? " [gen i]" : "");
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
                        instr->data.store.gen_write == 2 ? " [gen i+1]" : instr->data.store.gen_write == 1 ? " [gen i]" : "");
                dump_coords(out, instr->data.store.coords, instr->data.store.ncoords);
                fprintf(out, ", r%d)\n", instr->data.store.src_reg);
                break;
            case IR_GEN_LOOP_BEGIN:
                fprintf(out, "GEN_LOOP_BEGIN(");
                for (int c = 0; c < instr->data.gen_loop.num_cross_gens; c++) {
                    if (c > 0) fprintf(out, ", ");
                    fprintf(out, "cross=%s, spatial=[",
                            instr->data.gen_loop.cross_gens[c].name);
                    for (int j = 0; j < instr->data.gen_loop.cross_gens[c].spatial_ndims; j++) {
                        if (j > 0) fprintf(out, ",");
                        fprintf(out, "%d", instr->data.gen_loop.cross_gens[c].spatial_dims[j]);
                    }
                    fprintf(out, "]");
                }
                fprintf(out, ", intermediates=[");
                for (int j = 0; j < instr->data.gen_loop.num_intermediates; j++) {
                    if (j > 0) fprintf(out, ",");
                    fprintf(out, "%s", instr->data.gen_loop.intermediates[j].name);
                }
                {
                    IRGenLoop *g = &instr->data.gen_loop;
                    const char *ga = g->stop_gen_a ? "i+1" : "i";
                    if (g->stop_kind == STOP_OCCUPANCY)
                        fprintf(out, "], stop=||%s[%s]||==0, max_gen=%d)\n",
                                g->watch ? g->watch : "?", ga, g->max_gen);
                    else
                        fprintf(out, "], stop=%s[%s]==%s[%s], max_gen=%d)\n",
                                g->watch ? g->watch : "?", ga,
                                g->watch ? g->watch : "?", g->stop_gen_b ? "i+1" : "i",
                                g->max_gen);
                }
                break;
            case IR_GEN_LOOP_END:
                fprintf(out, "GEN_LOOP_END\n");
                break;
            case IR_SAVE:
                fprintf(out, "SAVE(%s -> %s)\n",
                        instr->data.save.tensor_name, instr->data.save.path);
                break;
            case IR_FUSED: {
                IRFused *fu = &instr->data.fused;
                fprintf(out, "FUSED(%s", fu->out_name);
                dump_coords(out, fu->out_coords, fu->out_ncoords);
                fprintf(out, " = (%s %s %s) %s %s | map1=%s/%s reduce1=%s/%s map2=%s/%s reduce2=%s/%s"
                             " reduce_on=%d free=%d red=%d)\n",
                        fu->inner1.name, operator_to_str(fu->map1_op), fu->inner2.name,
                        operator_to_str(fu->map2_op), fu->outer.name,
                        operator_to_str(fu->map1_op), merge_op_to_str(fu->map1_merge),
                        fu->a1_has_reduce ? operator_to_str(fu->reduce1_op) : "-",
                        fu->a1_has_reduce ? merge_op_to_str(fu->reduce1_merge) : "-",
                        operator_to_str(fu->map2_op), merge_op_to_str(fu->map2_merge),
                        fu->a2_has_reduce ? operator_to_str(fu->reduce2_op) : "-",
                        fu->a2_has_reduce ? merge_op_to_str(fu->reduce2_merge) : "-",
                        fu->reduce_on_action, fu->num_free, fu->num_red);
                break;
            }
        }
    }
}
