%{
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "ast.h"

extern int yylex(void);
void yyerror(const char *s);

AstNode *parse_result = NULL;

/* Report an unknown compute-operator name. Returns 1 (after yyerror) if
   `name` is not a built-in compute op, else 0. */
static int unknown_compute_op(const char *name) {
    if ((int)operator_from_str(name) != -1) return 0;
    char buf[256];
    snprintf(buf, sizeof(buf),
             "unknown compute operator '%s' (expected one of: "
             "add, sub, mul, min, max, and, or, take_left)",
             name);
    yyerror(buf);
    return 1;
}

/* Helper structs for list-building in semantic actions */

typedef struct {
    char **items;
    int   *offsets;
    int count;
    int capacity;
} StrList;

static StrList *strlist_new(void) {
    StrList *sl = malloc(sizeof(StrList));
    sl->capacity = 4;
    sl->items = malloc(sl->capacity * sizeof(char *));
    sl->offsets = malloc(sl->capacity * sizeof(int));
    sl->count = 0;
    return sl;
}

static void strlist_add(StrList *sl, char *s, int offset) {
    if (sl->count >= sl->capacity) {
        sl->capacity *= 2;
        sl->items = realloc(sl->items, sl->capacity * sizeof(char *));
        sl->offsets = realloc(sl->offsets, sl->capacity * sizeof(int));
    }
    sl->items[sl->count] = s;
    sl->offsets[sl->count] = offset;
    sl->count++;
}

typedef struct {
    RankPair *items;
    int count;
    int capacity;
} RankPairList;

static RankPairList *rankpairlist_new(void) {
    RankPairList *rpl = malloc(sizeof(RankPairList));
    rpl->capacity = 4;
    rpl->items = malloc(rpl->capacity * sizeof(RankPair));
    rpl->count = 0;
    return rpl;
}

static void rankpairlist_add(RankPairList *rpl, RankPair rp) {
    if (rpl->count >= rpl->capacity) {
        rpl->capacity *= 2;
        rpl->items = realloc(rpl->items, rpl->capacity * sizeof(RankPair));
    }
    rpl->items[rpl->count++] = rp;
}

typedef struct {
    LiteralData **items;
    int count;
    int capacity;
} LitList;

static LitList *litlist_new(void) {
    LitList *ll = malloc(sizeof(LitList));
    ll->capacity = 4;
    ll->items = malloc(ll->capacity * sizeof(LiteralData *));
    ll->count = 0;
    return ll;
}

static void litlist_add(LitList *ll, LiteralData *lit) {
    if (ll->count >= ll->capacity) {
        ll->capacity *= 2;
        ll->items = realloc(ll->items, ll->capacity * sizeof(LiteralData *));
    }
    ll->items[ll->count++] = lit;
}

/* Fused-einsum action-spec accumulator (map.N / reduce.N entries). */
typedef struct { int kind;   /* 0=map, 1=reduce */
                 int label;  /* 1 or 2 */
                 char *op;   /* compute op name (owned) */
                 int merge;  /* MergeOp */
                 char *over; /* rank name or NULL (owned) */ } FAct;
typedef struct { FAct items[8]; int n; int overflow; } FActList;
static FActList *factlist_new(void){ FActList *l=malloc(sizeof(FActList)); l->n=0; l->overflow=0; return l; }
static void factlist_add(FActList *l, FAct a){ if(l->n<8) l->items[l->n++]=a; else l->overflow=1; }

/* reduce(...) trailing args: optional merge keyword and optional `over=IDENT`. */
typedef struct { int merge; char *over; } ReduceArgs;

%}

%error-verbose

%union {
    int int_val;
    double float_val;
    char *str_val;
    void *node;        /* AstNode* */
    void *ptr;         /* generic pointer for helper structs */
    int datatype;      /* DataType enum */
    int op;            /* OperatorType enum */
    int mergeop;       /* MergeOp enum */
}

%token TOK_DIM TOK_TENSOR TOK_INIT TOK_MAP TOK_REDUCE TOK_EMPTY
%token TOK_GEN TOK_STOP TOK_STABLE TOK_OROR TOK_EQEQ
%token TOK_TYPE_INTEGER TOK_TYPE_FLOAT TOK_TYPE_BOOLEAN
%token TOK_DOUBLE_COLON
%token <int_val> TOK_INT_LIT TOK_BOOL_LIT
%token <float_val> TOK_FLOAT_LIT
%token <str_val> TOK_IDENT
%token TOK_LOAD TOK_SAVE TOK_ARROW TOK_SEED TOK_OVER TOK_OUTPUT
%token <str_val> TOK_STRING_LIT

%type <node> program stmt_list stmt dim_decl tensor_decl init_stmt einsum_stmt stop_stmt save_stmt output_stmt
%type <ptr> rank_pair_list ident_list output_names nested_literal lit_list tensor_ref map_operand
%type <datatype> datatype
%type <float_val> literal_value seed_value
%type <mergeop> merge_arg
%type <ptr> reduce_args fused_rhs fused_action_list fused_action

%%

program:
    stmt_list
    {
        parse_result = (AstNode *)$1;
    }
    ;

stmt_list:
    /* empty */
    {
        $$ = (void *)ast_program_new();
    }
    | stmt_list stmt
    {
        ast_program_add((AstNode *)$1, (AstNode *)$2);
        $$ = $1;
    }
    ;

stmt:
    dim_decl
    | tensor_decl
    | init_stmt
    | einsum_stmt
    | stop_stmt
    | save_stmt
    | output_stmt
    ;

stop_stmt:
    TOK_STOP ':' TOK_STABLE '(' TOK_IDENT ')' ';'
    {
        $$ = (void *)ast_stop_stable_new($5);
    }
    | TOK_STOP ':' TOK_IDENT '(' TOK_IDENT ')' ';'
    {
        char buf[256];
        snprintf(buf, sizeof(buf), "unknown stop predicate '%s' (expected 'stable')", $3);
        yyerror(buf);
        free($3); free($5);
        YYERROR;
    }
    | TOK_STOP ':' TOK_OROR tensor_ref TOK_OROR TOK_EQEQ TOK_INT_LIT ';'
    {
        TensorRef *r = (TensorRef *)$4;
        $$ = (void *)ast_stop_occupancy_new(*r, $7);
        free(r);
    }
    | TOK_STOP ':' tensor_ref TOK_EQEQ tensor_ref ';'
    {
        TensorRef *a = (TensorRef *)$3;
        TensorRef *b = (TensorRef *)$5;
        $$ = (void *)ast_stop_equal_new(*a, *b);
        free(a); free(b);
    }
    ;

save_stmt:
    TOK_SAVE ':' TOK_IDENT TOK_ARROW TOK_STRING_LIT ';'
    {
        $$ = (void *)ast_save_new($3, $5);
    }
    ;

output_stmt:
    TOK_OUTPUT ':' output_names ';'
    {
        StrList *sl = (StrList *)$3;
        $$ = (void *)ast_output_new(sl->items, sl->count);
        free(sl->offsets);
        free(sl);
    }
    ;

output_names:
    TOK_IDENT
    {
        StrList *sl = strlist_new();
        strlist_add(sl, $1, 0);
        $$ = (void *)sl;
    }
    | output_names ',' TOK_IDENT
    {
        StrList *sl = (StrList *)$1;
        strlist_add(sl, $3, 0);
        $$ = (void *)sl;
    }
    ;

dim_decl:
    TOK_DIM TOK_IDENT '=' TOK_INT_LIT ';'
    {
        $$ = (void *)ast_dim_decl_new($2, $4);
    }
    | TOK_DIM TOK_IDENT '=' '*' ';'
    {
        $$ = (void *)ast_dim_decl_inferred_new($2);
    }
    ;

tensor_decl:
    TOK_TENSOR TOK_IDENT '[' rank_pair_list ']' ':' datatype ',' TOK_EMPTY '=' literal_value ';'
    {
        RankPairList *rpl = (RankPairList *)$4;
        $$ = (void *)ast_tensor_decl_new($2, rpl->items, rpl->count, (DataType)$7, $11);
        free(rpl);  /* free the list struct but not items (owned by AST now) */
    }
    ;

rank_pair_list:
    TOK_IDENT '=' TOK_IDENT
    {
        RankPairList *rpl = rankpairlist_new();
        RankPair rp;
        rp.rank_name = $1;
        rp.dim_name = $3;
        rp.is_generational = 0;
        rankpairlist_add(rpl, rp);
        $$ = (void *)rpl;
    }
    | TOK_GEN TOK_IDENT
    {
        RankPairList *rpl = rankpairlist_new();
        RankPair rp;
        rp.rank_name = $2;
        rp.dim_name = NULL;
        rp.is_generational = 1;
        rankpairlist_add(rpl, rp);
        $$ = (void *)rpl;
    }
    | rank_pair_list ',' TOK_IDENT '=' TOK_IDENT
    {
        RankPairList *rpl = (RankPairList *)$1;
        RankPair rp;
        rp.rank_name = $3;
        rp.dim_name = $5;
        rp.is_generational = 0;
        rankpairlist_add(rpl, rp);
        $$ = (void *)rpl;
    }
    | rank_pair_list ',' TOK_GEN TOK_IDENT
    {
        RankPairList *rpl = (RankPairList *)$1;
        RankPair rp;
        rp.rank_name = $4;
        rp.dim_name = NULL;
        rp.is_generational = 1;
        rankpairlist_add(rpl, rp);
        $$ = (void *)rpl;
    }
    ;

datatype:
    TOK_TYPE_INTEGER { $$ = TYPE_INTEGER; }
    | TOK_TYPE_FLOAT { $$ = TYPE_FLOAT; }
    | TOK_TYPE_BOOLEAN { $$ = TYPE_BOOLEAN; }
    ;

literal_value:
    TOK_INT_LIT   { $$ = (double)$1; }
    | TOK_FLOAT_LIT { $$ = $1; }
    | TOK_BOOL_LIT  { $$ = (double)$1; }
    ;

init_stmt:
    TOK_INIT TOK_IDENT '=' nested_literal ';'
    {
        $$ = (void *)ast_init_new($2, (LiteralData *)$4);
    }
    | TOK_INIT TOK_IDENT '=' TOK_LOAD '(' TOK_STRING_LIT ')' ';'
    {
        $$ = (void *)ast_init_load_new($2, $6);
    }
    | TOK_INIT TOK_IDENT '=' TOK_SEED '(' TOK_IDENT '=' TOK_INT_LIT ',' TOK_IDENT '=' seed_value ')' ';'
    {
        if (strcmp($6, "source") != 0 || strcmp($10, "value") != 0) {
            yyerror("seed expects seed(source=<int>, value=<number>)");
            free($6); free($10); free($2);
            YYERROR;
        }
        free($6); free($10);
        $$ = (void *)ast_init_seed_new($2, $8, $12);
    }
    ;

seed_value:
      TOK_FLOAT_LIT { $$ = $1; }
    | TOK_INT_LIT   { $$ = (double)$1; }
    ;

nested_literal:
    TOK_INT_LIT
    {
        $$ = (void *)literal_scalar_int($1);
    }
    | TOK_FLOAT_LIT
    {
        $$ = (void *)literal_scalar_float($1);
    }
    | TOK_BOOL_LIT
    {
        $$ = (void *)literal_scalar_bool($1);
    }
    | '[' lit_list ']'
    {
        LitList *ll = (LitList *)$2;
        $$ = (void *)literal_list_new(ll->items, ll->count);
        free(ll);  /* free list struct but not items */
    }
    ;

lit_list:
    nested_literal
    {
        LitList *ll = litlist_new();
        litlist_add(ll, (LiteralData *)$1);
        $$ = (void *)ll;
    }
    | lit_list ',' nested_literal
    {
        LitList *ll = (LitList *)$1;
        litlist_add(ll, (LiteralData *)$3);
        $$ = (void *)ll;
    }
    ;

einsum_stmt:
    tensor_ref '=' map_operand '.' map_operand TOK_DOUBLE_COLON
        TOK_MAP '(' TOK_IDENT merge_arg ')'
        TOK_REDUCE '(' TOK_IDENT reduce_args ')' ';'
    {
        TensorRef *out = (TensorRef *)$1;
        TensorRef *in1 = (TensorRef *)$3;
        TensorRef *in2 = (TensorRef *)$5;
        ReduceArgs *ra = (ReduceArgs *)$15;
        if (unknown_compute_op($9) || unknown_compute_op($14)) {
            free(ra->over); free(ra); free($9); free($14);
            free(out); free(in1); free(in2);
            YYERROR;
        }
        OperatorType map_op    = operator_from_str($9);
        MergeOp     map_merge  = (MergeOp)$10;
        OperatorType reduce_op = operator_from_str($14);
        MergeOp     red_merge  = (MergeOp)ra->merge;
        AstNode *en = ast_einsum_new(*out, *in1, *in2,
                                     map_op, map_merge, reduce_op, red_merge);
        en->data.einsum.reduce_over = ra->over;   /* NULL when absent */
        $$ = (void *)en;
        free(ra);
        free($9);
        free($14);
        free(out);
        free(in1);
        free(in2);
    }
    | tensor_ref '=' fused_rhs TOK_DOUBLE_COLON fused_action_list ';'
    {
        EinsumFusedData *f = (EinsumFusedData *)$3;
        FActList *acts = (FActList *)$5;
        f->output = *(TensorRef *)$1;
        f->a1_has_reduce = 0; f->a2_has_reduce = 0;
        f->over1 = NULL; f->over2 = NULL;
        int seen_map1=0, seen_map2=0, seen_r1=0, seen_r2=0;
        if (acts->overflow) { yyerror("fused einsum: too many action specs"); free($1); free($3); free(acts); YYERROR; }
        for (int i = 0; i < acts->n; i++) {
            FAct *a = &acts->items[i];
            if (a->label != 1 && a->label != 2) { yyerror("fused einsum: action label must be .1 or .2"); free(a->op); free(a->over); free($1); free($3); free(acts); YYERROR; }
            if (unknown_compute_op(a->op)) { free(a->op); free(a->over); free($1); free($3); free(acts); YYERROR; }
            OperatorType op = operator_from_str(a->op);
            if (a->kind == 0 && a->label == 1) { if (seen_map1) { yyerror("fused einsum: duplicate map.1"); free(a->op); free(a->over); free($1); free($3); free(acts); YYERROR; } seen_map1=1; f->map1_op=op; f->map1_merge=a->merge; }
            else if (a->kind == 0 && a->label == 2) { if (seen_map2) { yyerror("fused einsum: duplicate map.2"); free(a->op); free(a->over); free($1); free($3); free(acts); YYERROR; } seen_map2=1; f->map2_op=op; f->map2_merge=a->merge; }
            else if (a->kind == 1 && a->label == 1) { if (seen_r1) { yyerror("fused einsum: duplicate reduce.1"); free(a->op); free(a->over); free($1); free($3); free(acts); YYERROR; } seen_r1=1; f->reduce1_op=op; f->reduce1_merge=a->merge; f->a1_has_reduce=1; f->over1=a->over; }
            else { if (seen_r2) { yyerror("fused einsum: duplicate reduce.2"); free(a->op); free(a->over); free($1); free($3); free(acts); YYERROR; } seen_r2=1; f->reduce2_op=op; f->reduce2_merge=a->merge; f->a2_has_reduce=1; f->over2=a->over; }
            free(a->op);
        }
        if (!seen_map1 || !seen_map2) { yyerror("fused einsum requires map.1 and map.2"); free($1); free($3); free(acts); YYERROR; }
        AstNode *node = ast_einsum_fused_new(*f);
        free($1); free($3); free(acts);
        $$ = (void *)node;
    }
    ;

fused_rhs:
    '(' map_operand '.' TOK_INT_LIT map_operand ')' '.' TOK_INT_LIT map_operand
    {
        if ($4 != 1 || $8 != 2) { yyerror("fused operators must be labeled .1 (inner) and .2 (outer)"); YYERROR; }
        EinsumFusedData *f = calloc(1, sizeof(EinsumFusedData));
        f->inner1 = *(TensorRef *)$2;
        f->inner2 = *(TensorRef *)$5;
        f->outer  = *(TensorRef *)$9;
        f->inter_is_left2 = 1;
        free($2); free($5); free($9);
        $$ = (void *)f;
    }
    | map_operand '.' TOK_INT_LIT '(' map_operand '.' TOK_INT_LIT map_operand ')'
    {
        if ($3 != 2 || $7 != 1) { yyerror("fused operators must be labeled .1 (inner) and .2 (outer)"); YYERROR; }
        EinsumFusedData *f = calloc(1, sizeof(EinsumFusedData));
        f->outer  = *(TensorRef *)$1;
        f->inner1 = *(TensorRef *)$5;
        f->inner2 = *(TensorRef *)$8;
        f->inter_is_left2 = 0;
        free($1); free($5); free($8);
        $$ = (void *)f;
    }
    ;

fused_action_list:
    fused_action                     { FActList *l = factlist_new(); factlist_add(l, *(FAct *)$1); free($1); $$ = (void *)l; }
    | fused_action_list fused_action { FActList *l = (FActList *)$1; factlist_add(l, *(FAct *)$2); free($2); $$ = (void *)l; }
    ;

fused_action:
    TOK_MAP '.' TOK_INT_LIT '(' TOK_IDENT merge_arg ')'
    {
        FAct *a = malloc(sizeof(FAct));
        a->kind = 0; a->label = $3; a->op = $5; a->merge = $6; a->over = NULL;
        $$ = (void *)a;
    }
    | TOK_REDUCE '.' TOK_INT_LIT '(' TOK_IDENT reduce_args ')'
    {
        ReduceArgs *ra = (ReduceArgs *)$6;
        FAct *a = malloc(sizeof(FAct));
        a->kind = 1; a->label = $3; a->op = $5; a->merge = ra->merge; a->over = ra->over;
        free(ra);
        $$ = (void *)a;
    }
    ;

/* A map operand is a tensor ref, optionally prefixed with `!` (unary
   complement). `!` is only valid here, not on the einsum output (LHS). */
map_operand:
    tensor_ref
    {
        $$ = $1;
    }
    | '!' tensor_ref
    {
        TensorRef *r = (TensorRef *)$2;
        r->negate = 1;
        $$ = (void *)r;
    }
    ;

/* Optional merge-operator arg shared by map(...) and reduce(...).
   Empty production yields MERGE_PASSTHROUGH; explicit form is ', IDENT'
   where IDENT must resolve to one of the 9 known merge keywords. */
merge_arg:
    /* empty */                { $$ = MERGE_PASSTHROUGH; }
    | ',' TOK_IDENT
    {
        MergeOp m = merge_op_from_str($2);
        if ((int)m == -1) {
            char buf[256];
            snprintf(buf, sizeof(buf),
                     "unknown merge operator '%s' (expected one of: "
                     "passthrough, nopass, intersect, take_left, "
                     "take_left_only, take_right, take_right_only, xor, union)",
                     $2);
            yyerror(buf);
            YYERROR;
        }
        free($2);
        $$ = m;
    }
    ;

/* reduce(...) trailing args: optional merge keyword then optional `over=IDENT`.
   Folded into one rule so the comma-prefixed forms stay LALR(1)-unambiguous
   (after the first ',' the token TOK_IDENT vs TOK_OVER disambiguates). */
reduce_args:
    /* empty */
    {
        ReduceArgs *r = malloc(sizeof(ReduceArgs));
        r->merge = MERGE_PASSTHROUGH; r->over = NULL;
        $$ = (void *)r;
    }
    | ',' TOK_IDENT
    {
        MergeOp m = merge_op_from_str($2);
        if ((int)m == -1) {
            char buf[256];
            snprintf(buf, sizeof(buf),
                     "unknown merge operator '%s' (expected one of: "
                     "passthrough, nopass, intersect, take_left, "
                     "take_left_only, take_right, take_right_only, xor, union)",
                     $2);
            yyerror(buf);
            YYERROR;
        }
        free($2);
        ReduceArgs *r = malloc(sizeof(ReduceArgs));
        r->merge = m; r->over = NULL;
        $$ = (void *)r;
    }
    | ',' TOK_IDENT ',' TOK_OVER '=' TOK_IDENT
    {
        MergeOp m = merge_op_from_str($2);
        if ((int)m == -1) {
            char buf[256];
            snprintf(buf, sizeof(buf),
                     "unknown merge operator '%s' (expected one of: "
                     "passthrough, nopass, intersect, take_left, "
                     "take_left_only, take_right, take_right_only, xor, union)",
                     $2);
            yyerror(buf);
            YYERROR;
        }
        free($2);
        ReduceArgs *r = malloc(sizeof(ReduceArgs));
        r->merge = m; r->over = $6;
        $$ = (void *)r;
    }
    | ',' TOK_OVER '=' TOK_IDENT
    {
        ReduceArgs *r = malloc(sizeof(ReduceArgs));
        r->merge = MERGE_PASSTHROUGH; r->over = $4;
        $$ = (void *)r;
    }
    ;

tensor_ref:
    TOK_IDENT '[' ident_list ']'
    {
        StrList *sl = (StrList *)$3;
        int noffsets = 0;
        for (int i = 0; i < sl->count; i++) if (sl->offsets[i] != 0) noffsets++;
        if (noffsets > 1) {
            yyerror("at most one rank offset per tensor reference");
            for (int i = 0; i < sl->count; i++) free(sl->items[i]);
            free(sl->items); free(sl->offsets); free(sl); free($1);
            YYERROR;
        }
        TensorRef *ref = malloc(sizeof(TensorRef));
        *ref = tensor_ref_new($1, sl->items, sl->count);
        for (int i = 0; i < sl->count; i++) {
            if (sl->offsets[i] != 0) {
                ref->gen_axis = i;
                ref->gen_offset = sl->offsets[i];
            }
        }
        free(sl->offsets);
        free(sl);  /* free list struct but not items */
        $$ = (void *)ref;
    }
    ;

ident_list:
    TOK_IDENT
    {
        StrList *sl = strlist_new();
        strlist_add(sl, $1, 0);
        $$ = (void *)sl;
    }
    | TOK_IDENT '+' TOK_INT_LIT
    {
        StrList *sl = strlist_new();
        strlist_add(sl, $1, $3);
        $$ = (void *)sl;
    }
    | ident_list ',' TOK_IDENT
    {
        StrList *sl = (StrList *)$1;
        strlist_add(sl, $3, 0);
        $$ = (void *)sl;
    }
    | ident_list ',' TOK_IDENT '+' TOK_INT_LIT
    {
        StrList *sl = (StrList *)$1;
        strlist_add(sl, $3, $5);
        $$ = (void *)sl;
    }
    ;

%%

void yyerror(const char *s) {
    fprintf(stderr, "parse error: %s\n", s);
}
