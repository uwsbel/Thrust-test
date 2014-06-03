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

const int ARRAY_SIZE = 1000;

enum Method {
	RAW,
	WRAPPED
};

bool inner_product_test(Method method)
{
	double* hA;
	double* hB;
	hA = (double *) malloc(sizeof(double) * ARRAY_SIZE);
	hB = (double *) malloc(sizeof(double) * ARRAY_SIZE);
	for (int i = 0; i < ARRAY_SIZE; i++) {
		hA[i] = 1.0 * (i+1);
		hB[i] = 1.0 * (ARRAY_SIZE - i);
	}

	double* dA;
	double* dB;
	cudaMalloc((void **) &dA, sizeof(double) * ARRAY_SIZE);
	cudaMalloc((void **) &dB, sizeof(double) * ARRAY_SIZE);
	cudaMemcpy(dA, hA, sizeof(double) * ARRAY_SIZE, cudaMemcpyHostToDevice);
	cudaMemcpy(dB, hB, sizeof(double) * ARRAY_SIZE, cudaMemcpyHostToDevice);

	double inner_product;
	switch (method) {
	case RAW:
		{
			inner_product = thrust::inner_product(thrust::cuda::par, dA, dA + ARRAY_SIZE, dB, 0.0, thrust::plus<double>(), thrust::multiplies<double>());
			break;
		}
	case WRAPPED:
		{
			thrust::device_ptr<double> wdA = thrust::device_pointer_cast(dA);
			thrust::device_ptr<double> wdB = thrust::device_pointer_cast(dB);
			inner_product = thrust::inner_product(wdA, wdA + ARRAY_SIZE, wdB, 0.0, thrust::plus<double>(), thrust::multiplies<double>());
			break;
		}
	}

	double ref_inner_product = 0.0;
	for (int i = 0; i < ARRAY_SIZE; i++)
		ref_inner_product += hA[i] * hB[i];

	bool result = (fabs(inner_product - ref_inner_product) / fabs(ref_inner_product) < 1e-10);

	cudaFree(dA);
	cudaFree(dB);
	free(hA);
	free(hB);

	return result;
}

int main(int argc, char **argv)
{
	std::cout << "Inner_product DR ... " << std::flush << inner_product_test(RAW) << std::endl;
	std::cout << "Inner_product DW ... " << std::flush << inner_product_test(WRAPPED) << std::endl;

	return 0;
}
