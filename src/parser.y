%{
#include <stdio.h>
#include <stdlib.h>
#include "ast.h"

extern int yylex(void);
void yyerror(const char *s);

AstNode *parse_result = NULL;
%}

%union {
    int int_val;
    double float_val;
    char *str_val;
    void *node;
    void *tensor_ref;
    void *rank_pair;
    void *rank_pair_list;
    void *literal;
    int datatype;
    int op;
}

%token TOK_DIM TOK_TENSOR TOK_INIT TOK_MAP TOK_REDUCE TOK_EMPTY
%token TOK_TYPE_INTEGER TOK_TYPE_FLOAT TOK_TYPE_BOOLEAN
%token TOK_DOUBLE_COLON
%token <int_val> TOK_INT_LIT TOK_BOOL_LIT
%token <float_val> TOK_FLOAT_LIT
%token <str_val> TOK_IDENT

%%

/* Stub grammar — will be completed in Task 4 */
program: /* empty */ ;

%%

void yyerror(const char *s) {
    fprintf(stderr, "parse error: %s\n", s);
}
