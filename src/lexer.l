%{
#include "ast.h"
#include "parser.tab.h"
#include <string.h>
%}

%option noyywrap
%option nounput
%option noinput

%%

"//"[^\n]*          { /* skip line comments */ }
[ \t\r\n]+          { /* skip whitespace */ }

"dim"               { return TOK_DIM; }
"tensor"            { return TOK_TENSOR; }
"init"              { return TOK_INIT; }
"map"               { return TOK_MAP; }
"reduce"            { return TOK_REDUCE; }
"gen"               { return TOK_GEN; }
"stop"              { return TOK_STOP; }
"stable"            { return TOK_STABLE; }
"empty"             { return TOK_EMPTY; }
"load"              { return TOK_LOAD; }
"seed"              { return TOK_SEED; }
"over"              { return TOK_OVER; }
"save"              { return TOK_SAVE; }
"integer"           { return TOK_TYPE_INTEGER; }
"float"             { return TOK_TYPE_FLOAT; }
"boolean"           { return TOK_TYPE_BOOLEAN; }
"true"              { yylval.int_val = 1; return TOK_BOOL_LIT; }
"false"             { yylval.int_val = 0; return TOK_BOOL_LIT; }
"inf"               { yylval.float_val = 1.0/0.0; return TOK_FLOAT_LIT; }

"->"                { return TOK_ARROW; }
"::"                { return TOK_DOUBLE_COLON; }
"||"                { return TOK_OROR; }
"=="                { return TOK_EQEQ; }

[0-9]+\.[0-9]+      { yylval.float_val = atof(yytext); return TOK_FLOAT_LIT; }
[0-9]+              { yylval.int_val = atoi(yytext); return TOK_INT_LIT; }

\"[^\"\n]*\"        {
                      int n = (int)strlen(yytext);
                      yylval.str_val = malloc(n - 1);     /* drop 2 quotes, add NUL */
                      memcpy(yylval.str_val, yytext + 1, n - 2);
                      yylval.str_val[n - 2] = '\0';
                      return TOK_STRING_LIT;
                    }

[a-zA-Z_][a-zA-Z0-9_]* { yylval.str_val = strdup(yytext); return TOK_IDENT; }

[=\.\,\;\:\[\]\(\)+*!]  { return yytext[0]; }

.                   { fprintf(stderr, "lexer error: unexpected character '%c'\n", yytext[0]); }

%%
