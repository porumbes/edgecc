#ifndef CODEGEN_H
#define CODEGEN_H

#include "ir.h"

/* Generate Python source code from IR. Returns a malloc'd string. */
char *codegen_python(IRProgram *ir);

#endif /* CODEGEN_H */
