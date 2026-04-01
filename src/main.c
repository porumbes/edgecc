#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "ast.h"
#include "semantic.h"
#include "ir.h"
#include "codegen.h"

extern int yyparse(void);
extern AstNode *parse_result;
extern FILE *yyin;

int main(int argc, char *argv[]) {
    if (argc < 2) {
        fprintf(stderr, "Usage: edgecc <input.edge> [-o output.py]\n");
        return 1;
    }

    const char *input_file = argv[1];
    const char *output_file = "output.py";

    for (int i = 2; i < argc; i++) {
        if (strcmp(argv[i], "-o") == 0 && i + 1 < argc) {
            output_file = argv[++i];
        }
    }

    FILE *in = fopen(input_file, "r");
    if (!in) {
        fprintf(stderr, "error: cannot open '%s'\n", input_file);
        return 1;
    }
    yyin = in;

    if (yyparse() != 0) {
        fprintf(stderr, "error: parsing failed\n");
        fclose(in);
        return 1;
    }
    fclose(in);

    if (!parse_result) {
        fprintf(stderr, "error: no AST produced\n");
        return 1;
    }

    SemanticResult sem = semantic_analyze(parse_result);
    if (sem.error_count > 0) {
        for (int i = 0; i < sem.error_count; i++) {
            fprintf(stderr, "error: %s\n", sem.errors[i]);
        }
        semantic_result_free(&sem);
        ast_free(parse_result);
        return 1;
    }

    IRProgram *ir = ir_lower(parse_result, sem.symtab);
    char *code = codegen_python(ir);

    FILE *out = fopen(output_file, "w");
    if (!out) {
        fprintf(stderr, "error: cannot open '%s' for writing\n", output_file);
        free(code);
        ir_free(ir);
        semantic_result_free(&sem);
        ast_free(parse_result);
        return 1;
    }
    fputs(code, out);
    fclose(out);

    fprintf(stderr, "edgecc: %s -> %s\n", input_file, output_file);

    free(code);
    ir_free(ir);
    semantic_result_free(&sem);
    ast_free(parse_result);
    return 0;
}
