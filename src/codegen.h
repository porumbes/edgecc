#ifndef CODEGEN_H
#define CODEGEN_H

#include "ir.h"

/* Generate Python source code from IR. Returns a malloc'd string. */
char *codegen_python(IRProgram *ir);

/* Extended form: emit_trace=1 adds per-generation trace capture for
   generational kernels (no-op for non-generational). kernel_name labels the
   trace and its output file (NULL -> "kernel"). */
char *codegen_python_ex(IRProgram *ir, int emit_trace, const char *kernel_name);

#endif /* CODEGEN_H */
