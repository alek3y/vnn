#include <stdio.h>
#include "vnn.h"

int main(void) {
	Network n = network_new((size_t[]) {2, 3, 2}, 3);
	for (size_t i = 0; i < n.layers-1; i++) {
		matrix_print(n.weights[i]);
	}
	network_free(&n);
}
