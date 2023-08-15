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

#define VNN_CALLOC(s) (memset(VNN_MALLOC(s), 0, (s)))

#ifndef VNN_DTYPE
#define VNN_DTYPE float
#endif
#ifndef VNN_DTYPE_ONE
#define VNN_DTYPE_ONE 1.0
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
#define VNN_DTYPE_ZERO VNN_DTYPE_SUB(VNN_DTYPE_ONE, VNN_DTYPE_ONE)

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
VNNDEF Matrix matrix_rand(size_t rows, size_t cols, VNN_DTYPE (*rand)());
VNNDEF Matrix matrix_copy(Matrix src);
VNNDEF Matrix matrix_from(VNN_DTYPE *data, size_t rows, size_t cols);
VNNDEF void matrix_free(Matrix *dest);
VNNDEF void matrix_apply(Matrix dest, VNN_DTYPE (*func)(VNN_DTYPE));
VNNDEF Matrix matrix_transpose(Matrix src);
VNNDEF Matrix matrix_add(Matrix a, Matrix b);
VNNDEF Matrix matrix_multiply(Matrix a, Matrix b);
VNNDEF void matrix_add_scalar(Matrix dest, VNN_DTYPE scalar);
VNNDEF void matrix_multiply_scalar(Matrix dest, VNN_DTYPE scalar);
VNNDEF void matrix_negate(Matrix dest);
VNNDEF void matrix_print(Matrix src);

#define MATRIX_AT(src, i, j) (src).data[(i)*(src).cols + (j)]
#define MATRIX_FREED(src) ((src).data == NULL)

typedef struct {
	size_t layers;
	VNN_DTYPE rate, (**s)(VNN_DTYPE), (**ds)(VNN_DTYPE);
	Matrix *weights;

	// Epoch relative data
	Matrix *diags, *outputs;
} Network;

VNNDEF Network network_new(
	size_t *shape, size_t layers, VNN_DTYPE rate,
	VNN_DTYPE (**activations)(VNN_DTYPE),
	VNN_DTYPE (**derivatives)(VNN_DTYPE),
	VNN_DTYPE (*rand)()
);
VNNDEF void network_free(Network *dest);
VNNDEF VNN_DTYPE network_error(Network src, Matrix target);

#define NETWORK_FREED(src) ((src).layers == 0)

VNNDEF Matrix matrix_empty(size_t rows, size_t cols) {
	assert(rows > 0 && cols > 0);

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

VNNDEF Matrix matrix_rand(size_t rows, size_t cols, VNN_DTYPE (*rand)()) {
	assert(rand != NULL);
	Matrix dest = matrix_empty(rows, cols);
	for (size_t i = 0; i < rows*cols; i++) {
		dest.data[i] = rand();
	}
	return dest;
}

VNNDEF Matrix matrix_copy(Matrix src) {
	Matrix dest = matrix_empty(src.rows, src.cols);
	memcpy(dest.data, src.data, src.rows*src.cols * sizeof(VNN_DTYPE));
	return dest;
}

VNNDEF Matrix matrix_from(VNN_DTYPE *data, size_t rows, size_t cols) {
	assert(data != NULL);
	Matrix dest = {.data = data, .rows = rows, .cols = cols};
	return matrix_copy(dest);
}

VNNDEF void matrix_free(Matrix *dest) {
	assert(!MATRIX_FREED(*dest));
	VNN_FREE(dest->data);
	memset(dest, 0, sizeof(Matrix));
}

VNNDEF void matrix_apply(Matrix dest, VNN_DTYPE (*func)(VNN_DTYPE)) {
	assert(!MATRIX_FREED(dest) && func != NULL);
	for (size_t i = 0; i < dest.rows*dest.cols; i++) {
		dest.data[i] = func(dest.data[i]);
	}
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

VNNDEF Matrix matrix_add(Matrix a, Matrix b) {
	assert(a.rows == b.rows && a.cols == b.cols);

	Matrix dest = matrix_empty(a.rows, a.cols);
	for (size_t i = 0; i < a.rows*a.cols; i++) {
		dest.data[i] = VNN_DTYPE_ADD(a.data[i], b.data[i]);
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

VNNDEF void matrix_add_scalar(Matrix dest, VNN_DTYPE scalar) {
	assert(!MATRIX_FREED(dest));
	for (size_t i = 0; i < dest.rows*dest.cols; i++) {
		dest.data[i] = VNN_DTYPE_ADD(dest.data[i], scalar);
	}
}

VNNDEF void matrix_multiply_scalar(Matrix dest, VNN_DTYPE scalar) {
	assert(!MATRIX_FREED(dest));
	for (size_t i = 0; i < dest.rows*dest.cols; i++) {
		dest.data[i] = VNN_DTYPE_MUL(dest.data[i], scalar);
	}
}

VNNDEF void matrix_negate(Matrix dest) {
	assert(!MATRIX_FREED(dest));
	for (size_t i = 0; i < dest.rows*dest.cols; i++) {
		dest.data[i] = VNN_DTYPE_NEG(dest.data[i]);
	}
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

VNNDEF Network network_new(
	size_t *shape, size_t layers, VNN_DTYPE rate,
	VNN_DTYPE (**activations)(VNN_DTYPE),
	VNN_DTYPE (**derivatives)(VNN_DTYPE),
	VNN_DTYPE (*rand)()
) {
	assert(shape != NULL && shape[0] > 0 && layers >= 2);
	assert(activations != NULL && derivatives != NULL && rand != NULL);

	Network dest = {
		.layers = layers, .rate = rate,
		.s = activations, .ds = derivatives,
		.weights = VNN_MALLOC((layers-1) * sizeof(Matrix)),

		.diags = VNN_CALLOC((layers-1) * sizeof(Matrix)),	// At `diags[0]` are the diagonalized derivatives of the 2nd layer
		.outputs = VNN_CALLOC(layers * sizeof(Matrix))
	};

	for (size_t i = 0; i < layers-1; i++) {
		assert(activations[i] != NULL && derivatives[i] != NULL);

		// Matrices are shaped $(n+1) \times k$ where $n$ is the number of the previous layer units,
		// extended to include the biases and $k$ is the number of the next layer units.
		// Each weight $w_{ij}$ at the $i$-th row and $j$-th column is the connection between the $i$-th
		// unit of the previous layer and the $j$-th unit of the next one (see Section 7.3.1, p. 165)
		dest.weights[i] = matrix_rand(shape[i]+1, shape[i+1], rand);
	}

	return dest;
}

VNNDEF VNN_DTYPE network_error(Network src, Matrix target) {
	assert(!NETWORK_FREED(src) && !MATRIX_FREED(src.diags[0]));

	matrix_negate(target);
	Matrix diff = matrix_add(src.outputs[src.layers-1], target);
	matrix_negate(target);

	VNN_DTYPE squares = VNN_DTYPE_ZERO;	// Stores the norm of `diff` squared
	for (size_t i = 0; i < diff.rows*diff.cols; i++) {
		squares = VNN_DTYPE_ADD(squares, VNN_DTYPE_MUL(diff.data[i], diff.data[i]));
	}

	VNN_DTYPE error = VNN_DTYPE_DIV(squares, VNN_DTYPE_ADD(VNN_DTYPE_ONE, VNN_DTYPE_ONE));	// Derivative cancels 2 out

	matrix_free(&diff);
	return error;	// Mean squared error (see Section 7.2.1, p. 156)
}

VNNDEF void network_free(Network *dest) {
	assert(!NETWORK_FREED(*dest));

	if (!MATRIX_FREED(dest->outputs[0])) {
		matrix_free(&dest->outputs[0]);
	}

	for (size_t i = 1; i < dest->layers; i++) {
		matrix_free(&dest->weights[i-1]);

		if (!MATRIX_FREED(dest->diags[i-1])) {
			matrix_free(&dest->diags[i-1]);
		}
		if (!MATRIX_FREED(dest->outputs[i])) {
			matrix_free(&dest->outputs[i]);
		}
	}

	VNN_FREE(dest->weights);
	VNN_FREE(dest->diags);
	VNN_FREE(dest->outputs);
	memset(dest, 0, sizeof(Network));
}

#endif
