#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include "vnn.h"

float randw() {
	return (float) rand() / RAND_MAX;
}

float activation(float excitation) {
	return 1.0/(1.0 + exp(-excitation));	// See Section 7.1.1, p. 152
}

float derivative(float excitation) {
	return exp(-excitation)/pow(1.0 + exp(-excitation), 2);
}

int main(void) {
	srand(time(NULL));
	Network n = network_new(
		(size_t[]) {2, 3, 2}, 3, 1,
		(VNN_DTYPE (*[])(VNN_DTYPE)) {activation, activation},
		(VNN_DTYPE (*[])(VNN_DTYPE)) {derivative, derivative},
		randw
	);
	for (size_t i = 0; i < n.layers-1; i++) {
		matrix_multiply_scalar(n.weights[i], 0);
		matrix_add_scalar(n.weights[i], 1);
		matrix_print(n.weights[i]);
	}
	network_free(&n);
}
