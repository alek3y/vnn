#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include "vnn.h"

float weights(void) {
	return (float) rand() / (float) RAND_MAX;
}

float activation(float excitation) {
	return 1.0/(1.0 + exp(-excitation));
}

float derivative(float excitation) {
	return exp(-excitation)/pow(1.0 + exp(-excitation), 2);
}

int main(void) {
	srand(time(NULL));

	const size_t epochs = 10000;	// Small epochs show slow learning for certain starting weights

	const Matrix inputs[] = {
		matrix_from((float[]) {0, 0}, 1, 2),
		matrix_from((float[]) {0, 1}, 1, 2),
		matrix_from((float[]) {1, 0}, 1, 2),
		matrix_from((float[]) {1, 1}, 1, 2)
	};

	const Matrix targets[] = {
		matrix_from((float[]) {0}, 1, 1),
		matrix_from((float[]) {1}, 1, 1),
		matrix_from((float[]) {1}, 1, 1),
		matrix_from((float[]) {0}, 1, 1)
	};

	Network nn = network_new(
		(size_t[]) {2, 2, 1}, 3, 1,
		(float (*[])(float)) {activation, activation},
		(float (*[])(float)) {derivative, derivative},
		weights
	);

	size_t samples = sizeof(inputs)/sizeof(inputs[0]);
	for (size_t e = 1; e <= epochs; e++) {
		size_t sample = rand() % samples;
		network_feed(nn, inputs[sample]);

		float error = network_error(nn, targets[sample]);
		printf("Epoch %lu/%lu, Error: %g\r", e, epochs, error);

		network_adjust(nn, targets[sample]);
	}
	printf("\n");

	for (size_t i = 0; i < samples; i++) {
		Matrix input = inputs[i];
		printf("Input {%g, %g}", MATRIX_AT(input, 0, 0), MATRIX_AT(input, 0, 1));
		Matrix output = network_feed(nn, input);
		printf(", Output {%g, %g}\n", MATRIX_AT(output, 0, 0), MATRIX_AT(output, 0, 1));
	}

	network_free(&nn);
}
