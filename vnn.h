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

#ifdef VNN_EXTERN
#define VNNDEF extern
#else
#define VNNDEF static
#endif

typedef struct {
	VNN_DTYPE *data;
	size_t rows, cols;
} Matrix;

VNNDEF Matrix matrix_new(VNN_DTYPE *data, size_t rows, size_t cols);
VNNDEF void matrix_free(Matrix *dest);
VNNDEF Matrix matrix_copy(Matrix src);
VNNDEF Matrix matrix_transpose(Matrix src);

#define MATRIX_AT(src, i, j) (src).data[(i)*(src).cols + (j)]

typedef struct {
	Matrix *weights;
	size_t layers;
} Network;

VNNDEF Matrix matrix_new(VNN_DTYPE *data, size_t rows, size_t cols) {
	Matrix dest = {data, rows, cols};
	return matrix_copy(dest);
}

VNNDEF void matrix_free(Matrix *dest) {
	assert(dest->data != NULL);
	VNN_FREE(dest->data);
	dest->data = NULL;
}

VNNDEF Matrix matrix_copy(Matrix src) {
	size_t size = src.cols*src.rows * sizeof(VNN_DTYPE);
	Matrix dest = {VNN_MALLOC(size), src.rows, src.cols};
	memcpy(dest.data, src.data, size);
	return dest;
}

VNNDEF Matrix matrix_transpose(Matrix src) {
	Matrix dest = matrix_copy(src);
	dest.rows = src.cols;
	dest.cols = src.rows;

	for (size_t i = 0; i < src.rows; i++) {
		for (size_t j = 0; j < src.cols; j++) {
			MATRIX_AT(dest, j, i) = MATRIX_AT(src, i, j);
		}
	}

	return dest;
}

#endif
