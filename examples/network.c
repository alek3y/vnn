#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "vnn.h"

float weights_rand() {
	return (float) rand() / RAND_MAX;
}

int main(void) {
	srand(time(NULL));
	Network n = network_new((size_t[]) {2, 3, 2}, 3, weights_rand);
	for (size_t i = 0; i < n.layers-1; i++) {
		matrix_multiply_scalar(n.weights[i], 0);
		matrix_add_scalar(n.weights[i], 1);
		matrix_print(n.weights[i]);
	}
	network_free(&n);
}
