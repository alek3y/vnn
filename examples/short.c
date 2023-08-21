#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <assert.h>

#define VNN_DTYPE int16_t
#define VNN_DTYPE_TO_FLOAT(a) (((float) (a)) / 1000)
#define VNN_DTYPE_FROM_FLOAT(a) ((int16_t) ((a) * 1000))

#include "vnn.h"

float weights(void) {
	return ((float) rand() / (float) RAND_MAX)*2 - 1;
}

float activation(float excitation) {
	return 1.0/(1.0 + exp(-excitation));
}

float derivative(float excitation) {
	return activation(excitation) * (1.0 - activation(excitation));
}

float fix(float input) {
	return input*VNN_DTYPE_FROM_FLOAT(1);
}

const size_t samples = 10;

int16_t serif[][5] = {
	{0,1,1,1,0},	// 0
	{1,0,0,0,1},
	{1,0,0,0,1},
	{1,0,0,0,1},
	{0,1,1,1,0},

	{0,1,1,0,0},	// 1
	{0,0,1,0,0},
	{0,0,1,0,0},
	{0,0,1,0,0},
	{0,1,1,1,0},

	{0,1,1,1,0},	// 2
	{1,0,0,0,1},
	{0,0,1,1,0},
	{0,1,0,0,0},
	{1,1,1,1,1},

	{0,1,1,1,0},	// 3
	{1,0,0,0,1},
	{0,0,1,1,0},
	{1,0,0,0,1},
	{0,1,1,1,0},

	{0,0,0,1,0},	// 4
	{0,0,1,1,0},
	{0,1,0,1,0},
	{1,1,1,1,1},
	{0,0,0,1,0},

	{1,1,1,1,1},	// 5
	{1,0,0,0,0},
	{0,1,1,1,0},
	{0,0,0,0,1},
	{1,1,1,1,0},

	{0,1,1,1,1},	// 6
	{1,0,0,0,0},
	{1,1,1,1,0},
	{1,0,0,0,1},
	{0,1,1,1,0},

	{0,1,1,1,1},	// 7
	{1,0,0,0,1},
	{0,0,0,1,0},
	{0,0,1,0,0},
	{0,1,0,0,0},

	{0,1,1,1,0},	// 8
	{1,0,0,0,1},
	{0,1,1,1,0},
	{1,0,0,0,1},
	{0,1,1,1,0},

	{0,1,1,1,0},	// 9
	{1,0,0,0,1},
	{0,1,1,1,1},
	{0,0,0,0,1},
	{1,1,1,1,0}
};

int16_t segments[][5] = {
	{0,1,1,1,0},	// 0
	{0,1,0,1,0},
	{0,1,0,1,0},
	{0,1,0,1,0},
	{0,1,1,1,0},

	{0,0,0,1,0},	// 1
	{0,0,0,1,0},
	{0,0,0,1,0},
	{0,0,0,1,0},
	{0,0,0,1,0},

	{0,1,1,1,0},	// 2
	{0,0,0,1,0},
	{0,1,1,1,0},
	{0,1,0,0,0},
	{0,1,1,1,0},

	{0,1,1,1,0},	// 3
	{0,0,0,1,0},
	{0,1,1,1,0},
	{0,0,0,1,0},
	{0,1,1,1,0},

	{0,1,0,1,0},	// 4
	{0,1,0,1,0},
	{0,1,1,1,0},
	{0,0,0,1,0},
	{0,0,0,1,0},

	{0,1,1,1,0},	// 5
	{0,1,0,0,0},
	{0,1,1,1,0},
	{0,0,0,1,0},
	{0,1,1,1,0},

	{0,1,1,1,0},	// 6
	{0,1,0,0,0},
	{0,1,1,1,0},
	{0,1,0,1,0},
	{0,1,1,1,0},

	{0,1,1,1,0},	// 7
	{0,0,0,1,0},
	{0,0,0,1,0},
	{0,0,0,1,0},
	{0,0,0,1,0},

	{0,1,1,1,0},	// 8
	{0,1,0,1,0},
	{0,1,1,1,0},
	{0,1,0,1,0},
	{0,1,1,1,0},

	{0,1,1,1,0},	// 9
	{0,1,0,1,0},
	{0,1,1,1,0},
	{0,0,0,1,0},
	{0,1,1,1,0}
};

// NOTE: Training is not diverse enough for the network to perform well on test data
int16_t blocky[][5] = {
	{1,1,1,1,1},	// 0
	{1,0,0,0,1},
	{1,0,0,0,1},
	{1,0,0,0,1},
	{1,1,1,1,1},

	{0,0,1,0,0},	// 1
	{0,0,1,0,0},
	{0,0,1,0,0},
	{0,0,1,0,0},
	{0,0,1,0,0},

	{1,1,1,1,1},	// 2
	{0,0,0,0,1},
	{1,1,1,1,1},
	{1,0,0,0,0},
	{1,1,1,1,1},

	{1,1,1,1,1},	// 3
	{0,0,0,0,1},
	{0,1,1,1,1},
	{0,0,0,0,1},
	{1,1,1,1,1},

	{0,0,1,1,0},	// 4
	{0,1,0,1,0},
	{1,0,0,1,0},
	{1,1,1,1,1},
	{0,0,0,1,0},

	{1,1,1,1,1},	// 5
	{1,0,0,0,0},
	{1,1,1,1,1},
	{0,0,0,0,1},
	{1,1,1,1,1},

	{1,1,1,1,1},	// 6
	{1,0,0,0,0},
	{1,1,1,1,1},
	{1,0,0,0,1},
	{1,1,1,1,1},

	{1,1,1,1,1},	// 7
	{0,0,0,1,0},
	{0,0,1,0,0},
	{0,1,0,0,0},
	{1,0,0,0,0},

	{1,1,1,1,1},	// 8
	{1,0,0,0,1},
	{1,1,1,1,1},
	{1,0,0,0,1},
	{1,1,1,1,1},

	{1,1,1,1,1},	// 9
	{1,0,0,0,1},
	{1,1,1,1,1},
	{0,0,0,0,1},
	{1,1,1,1,1}
};

int main(void) {
	srand(time(NULL));

	const size_t epochs = 2500;

	Matrix inputs[samples*3], targets[samples];
	for (size_t i = 0; i < samples; i++) {
		inputs[samples*0 + i] = matrix_from(serif[i*5], 1, 5*5);
		inputs[samples*1 + i] = matrix_from(segments[i*5], 1, 5*5);
		inputs[samples*2 + i] = matrix_from(blocky[i*5], 1, 5*5);
		matrix_apply(inputs[samples*0 + i], fix);
		matrix_apply(inputs[samples*1 + i], fix);
		matrix_apply(inputs[samples*2 + i], fix);

		targets[i] = matrix_zeros(1, 10);
		MATRIX_AT(targets[i], 0, i) = VNN_DTYPE_FROM_FLOAT(1);
	}

	Network nn = network_new(
		(size_t[]) {5*5, 100, 20, 10}, 4, 3,
		(float (*[])(float)) {activation, activation, activation},
		(float (*[])(float)) {derivative, derivative, derivative},
		weights
	);

	for (size_t e = 1; e <= epochs; e++) {
		float error = 0;
		for (size_t i = 0; i < samples*2; i++) {	// TIP: Try to change `*2` to `*3` for control to be recognized
			network_feed(nn, inputs[i]);
			error += network_error(nn, targets[i % samples]);
			network_adjust(nn, targets[i % samples]);
		}
		error /= samples*2;

		printf("Epoch %lu/%lu, Error: %.10f\r", e, epochs, error);
	}
	printf("\n");

	for (size_t i = 0; i < samples; i++) {
		size_t set = rand() % 3;	// Choose between `serif`, `segments` and `blocky`
		Matrix input = inputs[samples*set + i];
		Matrix output = network_feed(nn, input);

		printf("Input");
		if (set == 2) {
			printf(" (control)");
		}
		printf(":\n");

		input.rows = 5;
		input.cols = 5;
		matrix_print(input);

		printf("Expected output:\n");
		matrix_print(targets[i]);

		printf("Output:\n");
		matrix_print(output);
	}

	network_free(&nn);
	for (size_t i = 0; i < samples; i++) {
		matrix_free(&targets[i]);
	}
}
