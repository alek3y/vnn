#ifndef VNN_H
#define VNN_H

#include <assert.h>
#include <string.h>

#if !defined(VNN_MALLOC) && !defined(VNN_FREE)
#include <stdlib.h>
#define VNN_MALLOC malloc
#define VNN_FREE free
#endif

// TODO: Replace SUB with NEG and DIV with INV?

#ifndef VNN_DTYPE
#define VNN_DTYPE float
#endif
#ifndef VNN_DTYPE_ADD
#define VNN_DTYPE_ADD(a, b) ((a) + (b))
#endif
#ifndef VNN_DTYPE_SUB
#define VNN_DTYPE_SUB(a, b) ((a) - (b))
#endif
#ifndef VNN_DTYPE_MUL
#define VNN_DTYPE_MUL(a, b) ((a) * (b))
#endif
#ifndef VNN_DTYPE_DIV
#define VNN_DTYPE_DIV(a, b) ((a) / (b))
#endif

typedef struct {
	VNN_DTYPE *data;
	size_t cols, rows;
} Matrix;

typedef struct {
	Matrix *weights;
	size_t layers;
} Network;

#endif
