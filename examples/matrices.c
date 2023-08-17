#include <stdio.h>
#include "vnn.h"

int main(void) {
	Matrix a = matrix_from((float[]) {1, 2, 3, 4, 5, 6}, 2, 3);
	Matrix b = matrix_from((float[]) {7, 8, 9}, 3, 1);

	matrix_transpose(&a);
	matrix_print(a);
	matrix_transpose(&a);

	printf("\n");
	matrix_print(a);
	printf(".\n");
	matrix_print(b);
	printf("=\n");

	Matrix ab = matrix_multiply(a, b);	// 2x3 . 3x1 ~> 2x1
	assert(ab.rows == 2 && ab.cols == 1);
	matrix_print(ab);

	printf("\n");
	matrix_multiply_scalar(ab, -1);
	matrix_print(ab);

	printf("\n");
	Matrix c = matrix_add(ab, ab);
	matrix_print(c);

	printf("\n");
	matrix_add_scalar(c, 100);
	matrix_print(c);

	printf("\n");
	matrix_multiply_scalar(ab, 2);
	matrix_print(ab);

	printf("\n");
	matrix_transpose(&ab);
	matrix_print(ab);

	printf("\n");
	Matrix d = matrix_from((float *) "\x00\x00\x00\x00\x00\x00\x80\x40", 1, 2);	// {0, 4} in LE IEE754
	d = matrix_clone(d);
	matrix_add_scalar(d, 2);	// Would segfault without cloning `d`
	matrix_print(d);

	matrix_free(&d);
	matrix_free(&c);
	matrix_free(&ab);
}
