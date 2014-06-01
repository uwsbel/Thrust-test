#include <iostream>
#include <cmath>

#include <thrust/scan.h>
#include <thrust/functional.h>
#include <thrust/sequence.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/gather.h>
#include <thrust/logical.h>
#include <thrust/host_vector.h>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/binary_search.h>
#include <thrust/execution_policy.h>
#include <thrust/inner_product.h>
#include <thrust/system/cuda/execution_policy.h>

using std::cout;
using std::cerr;
using std::endl;
using std::flush;

void inner_product_test() {
	cout << "Inner product test ... " << flush;
	const int ARRAY_SIZE = 1000;

	double *hA, *dA, *dB, *hB;

	hA = (double *)malloc(sizeof(double) * ARRAY_SIZE);
	hB = (double *)malloc(sizeof(double) * ARRAY_SIZE);
	cudaMalloc(&dA, sizeof(double) * ARRAY_SIZE);
	cudaMalloc(&dB, sizeof(double) * ARRAY_SIZE);

	for (int i = 0; i < ARRAY_SIZE; i++) {
		hA[i] = 1.0 * (i+1);
		hB[i] = 1.0 * (ARRAY_SIZE - i);
	}

	cudaMemcpy(dA, hA, sizeof(double) * ARRAY_SIZE, cudaMemcpyHostToDevice);
	cudaMemcpy(dB, hB, sizeof(double) * ARRAY_SIZE, cudaMemcpyHostToDevice);

	//// double inner_product = thrust::inner_product(thrust::cuda::par, dA, dA + ARRAY_SIZE, dB, 0.0, thrust::plus<double>(), thrust::multiplies<double>());
	double inner_product;
	{
		thrust::device_ptr<double> A_begin(dA), A_end(dA + ARRAY_SIZE), B_begin(dB);
		inner_product = thrust::inner_product(thrust::cuda::par, A_begin, A_end, B_begin, 0.0, thrust::plus<double>(), thrust::multiplies<double>());
	}
	double ref_inner_product = 0.0;

	for (int i = 0; i < ARRAY_SIZE; i++)
		ref_inner_product += hA[i] * hB[i];

	bool correct = (fabs(inner_product - ref_inner_product) / fabs(ref_inner_product) < 1e-10);

	cudaFree(dA);
	cudaFree(dB);
	free(hA);
	free(hB);

	if (correct)
		cout << "OK" << endl;
	else
		cout << "Failed" << endl;
}

int main(int argc, char **argv) {
	inner_product_test();
	return 0;
}
