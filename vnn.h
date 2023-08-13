#ifndef VNN_H
#define VNN_H

// TODO: One might not have the standard library available
#include <stdio.h>
#include <assert.h>
#include <string.h>

#if !defined(VNN_MALLOC) && !defined(VNN_FREE)
#include <stdlib.h>
#define VNN_MALLOC malloc
#define VNN_FREE free
#endif

#ifndef VNN_DTYPE
#define VNN_DTYPE float
#endif
#ifndef VNN_DTYPE_ADD
#define VNN_DTYPE_ADD(a, b) ((a) + (b))
#endif
#ifndef VNN_DTYPE_MUL
#define VNN_DTYPE_MUL(a, b) ((a) * (b))
#endif
#ifndef VNN_DTYPE_NEG
#define VNN_DTYPE_NEG(a) (-(a))
#endif

// TODO: Train network with DIV/MUL+INV and see if weights differ (a*1/b loses precision?)
#ifndef VNN_DTYPE_DIV
#define VNN_DTYPE_DIV(a, b) ((a) / (b))
#endif

#define VNN_DTYPE_SUB(a, b) VNN_DTYPE_ADD((a), VNN_DTYPE_NEG(b))
#define VNN_DTYPE_ZERO VNN_DTYPE_SUB((VNN_DTYPE) {0}, (VNN_DTYPE) {0})

#ifdef VNN_EXTERN
#define VNNDEF extern
#else
#define VNNDEF static
#endif

typedef struct {
	VNN_DTYPE *data;
	size_t rows, cols;
} Matrix;

VNNDEF Matrix matrix_empty(size_t rows, size_t cols);
VNNDEF Matrix matrix_zeros(size_t rows, size_t cols);
VNNDEF Matrix matrix_copy(Matrix src);
VNNDEF Matrix matrix_from(VNN_DTYPE *data, size_t rows, size_t cols);
VNNDEF void matrix_free(Matrix *dest);
VNNDEF Matrix matrix_transpose(Matrix src);	// TODO: Rewrite in-place?
VNNDEF Matrix matrix_multiply(Matrix a, Matrix b);
VNNDEF void matrix_print(Matrix src);

#define MATRIX_AT(src, i, j) (src).data[(i)*(src).cols + (j)]

typedef struct {
	Matrix *weights;
	size_t layers;
} Network;

VNNDEF Matrix matrix_empty(size_t rows, size_t cols) {
	return (Matrix) {
		.data = VNN_MALLOC(rows*cols * sizeof(VNN_DTYPE)),
		.rows = rows,
		.cols = cols
	};
}

VNNDEF Matrix matrix_zeros(size_t rows, size_t cols) {
	Matrix dest = matrix_empty(rows, cols);
	for (size_t i = 0; i < rows*cols; i++) {
		dest.data[i] = VNN_DTYPE_ZERO;
	}
	return dest;
}

VNNDEF Matrix matrix_copy(Matrix src) {
	Matrix dest = matrix_empty(src.rows, src.cols);
	memcpy(dest.data, src.data, src.rows*src.cols * sizeof(VNN_DTYPE));
	return dest;
}

VNNDEF Matrix matrix_from(VNN_DTYPE *data, size_t rows, size_t cols) {
	Matrix dest = {.data = data, .rows = rows, .cols = cols};
	return matrix_copy(dest);
}

VNNDEF void matrix_free(Matrix *dest) {
	assert(dest->data != NULL);
	VNN_FREE(dest->data);
	dest->data = NULL;
}

VNNDEF Matrix matrix_transpose(Matrix src) {
	Matrix dest = matrix_empty(src.cols, src.rows);
	for (size_t i = 0; i < src.rows; i++) {
		for (size_t j = 0; j < src.cols; j++) {
			MATRIX_AT(dest, j, i) = MATRIX_AT(src, i, j);
		}
	}
	return dest;
}

VNNDEF Matrix matrix_multiply(Matrix a, Matrix b) {
	assert(a.cols == b.rows);

	Matrix dest = matrix_empty(a.rows, b.cols);
	for (size_t i = 0; i < a.rows; i++) {
		for (size_t j = 0; j < b.cols; j++) {
			VNN_DTYPE sum = VNN_DTYPE_ZERO;
			for (size_t k = 0; k < a.cols; k++) {
				VNN_DTYPE mul = VNN_DTYPE_MUL(MATRIX_AT(a, i, k), MATRIX_AT(b, k, j));
				sum = VNN_DTYPE_ADD(sum, mul);
			}
			MATRIX_AT(dest, i, j) = sum;
		}
	}
	return dest;
}

VNNDEF void matrix_print(Matrix src) {
	printf("{");
	for (size_t i = 0; i < src.rows; i++) {
		printf("{");
		for (size_t j = 0; j < src.cols; j++) {
			int longest = 0;
			for (size_t k = 0; k < src.rows; k++) {
				int len = snprintf(NULL, 0, "%lg", ((long) (MATRIX_AT(src, k, j) * 1000.0)) / 1000.0);
				if (len > longest) {
					longest = len;
				}
			}

			printf("%*lg", longest, ((long) (MATRIX_AT(src, i, j) * 1000.0)) / 1000.0);
			if (j < src.cols-1) {
				printf(" ");
			}
		}
		printf("}");
		if (i < src.rows-1) {
			printf(",\n ");
		}
	}
	printf("}\n");
}

#endif
