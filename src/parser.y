%{
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "ast.h"

extern int yylex(void);
void yyerror(const char *s);

AstNode *parse_result = NULL;

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
%token TOK_GEN TOK_STOP TOK_STABLE
%token TOK_TYPE_INTEGER TOK_TYPE_FLOAT TOK_TYPE_BOOLEAN
%token TOK_DOUBLE_COLON
%token <int_val> TOK_INT_LIT TOK_BOOL_LIT
%token <float_val> TOK_FLOAT_LIT
%token <str_val> TOK_IDENT
%token TOK_LOAD TOK_SAVE TOK_ARROW TOK_SEED
%token <str_val> TOK_STRING_LIT

%type <node> program stmt_list stmt dim_decl tensor_decl init_stmt einsum_stmt stop_stmt save_stmt
%type <ptr> rank_pair_list ident_list nested_literal lit_list tensor_ref map_operand
%type <datatype> datatype
%type <float_val> literal_value seed_value
%type <mergeop> merge_arg

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
    ;

stop_stmt:
    TOK_STOP ':' TOK_STABLE '(' TOK_IDENT ')' ';'
    {
        $$ = (void *)ast_stop_new($5);
    }
    ;

save_stmt:
    TOK_SAVE ':' TOK_IDENT TOK_ARROW TOK_STRING_LIT ';'
    {
        $$ = (void *)ast_save_new($3, $5);
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
        TOK_REDUCE '(' TOK_IDENT merge_arg ')' ';'
    {
        TensorRef *out = (TensorRef *)$1;
        TensorRef *in1 = (TensorRef *)$3;
        TensorRef *in2 = (TensorRef *)$5;
        OperatorType map_op    = operator_from_str($9);
        MergeOp     map_merge  = (MergeOp)$10;
        OperatorType reduce_op = operator_from_str($14);
        MergeOp     red_merge  = (MergeOp)$15;
        $$ = (void *)ast_einsum_new(*out, *in1, *in2,
                                    map_op, map_merge,
                                    reduce_op, red_merge);
        free($9);
        free($14);
        free(out);
        free(in1);
        free(in2);
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

tensor_ref:
    TOK_IDENT '[' ident_list ']'
    {
        StrList *sl = (StrList *)$3;
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
