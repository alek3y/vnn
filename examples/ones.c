#include <stdio.h>
#include <math.h>
#include "vnn.h"

float weights(void) {
	return 1.0;
}

float activation(float excitation) {
	return 1.0/(1.0 + exp(-excitation));	// Sigmoid (see Section 7.1.1, p. 152)
}

float derivative(float excitation) {
	return exp(-excitation)/pow(1.0 + exp(-excitation), 2);
}

int main(void) {
	Network nn = network_new(
		(size_t[]) {2, 3, 2}, 3, 1,
		(float (*[])(float)) {activation, activation},
		(float (*[])(float)) {derivative, derivative},
		weights
	);

	printf("Weights:\n");
	for (size_t i = 0; i < nn.layers-1; i++) {
		matrix_print(nn.weights[i]);
	}

	Matrix input = matrix_from((float[]) {1, 2}, 1, 2);
	printf("Input:\n");
	matrix_print(input);

	Matrix target = matrix_from((float[]) {1, 0}, 1, 2);
	printf("Target:\n");
	matrix_print(target);

	Matrix output = network_feed(nn, input);
	printf("Output:\n");
	matrix_print(output);

	printf("Error:\n%g\n", network_error(nn, target));

	network_adjust(nn, target);
	printf("Deltas:\n");
	for (size_t i = 0; i < nn.layers-1; i++) {
		matrix_print(nn.deltas[i]);
	}
	printf("Weights:\n");
	for (size_t i = 0; i < nn.layers-1; i++) {
		matrix_print(nn.weights[i]);
	}

	network_free(&nn);
}
