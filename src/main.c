#include <stdio.h>
#include <stdlib.h>
#include <string.h>

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

    printf("edgecc: %s -> %s\n", input_file, output_file);
    return 0;
}
