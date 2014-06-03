#include <iostream>
#include <cmath>
#include <stdlib.h>

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
#include <thrust/system/cuda/execution_policy.h>

const int ARRAY_SIZE = 1000;

enum Method {
	RAW,
	WRAPPED
};

bool reduce_test(Method method)
{
	double* hA;
	hA = (double *) malloc(ARRAY_SIZE * sizeof(double));
	for (int i = 0; i < ARRAY_SIZE; i++)
		hA[i] = 1.0 * (i + 1);

	double* dA;
	cudaMalloc((void **) &dA, ARRAY_SIZE * sizeof(double));
	cudaMemcpy(dA, hA, sizeof(double) * ARRAY_SIZE, cudaMemcpyHostToDevice);

	double maximum;
	switch (method) {
	case RAW:
		{
			maximum = thrust::reduce(thrust::cuda::par, dA, dA + ARRAY_SIZE, 0.0, thrust::maximum<double>());
			break;
		}
	case WRAPPED:
		{
			thrust::device_ptr<double> wdA = thrust::device_pointer_cast(dA);
			maximum = thrust::reduce(wdA, wdA + ARRAY_SIZE, 0.0, thrust::maximum<double>());
			break;
		}
	}

	bool result = (fabs(maximum - ARRAY_SIZE) < 1e-10);

	cudaFree(dA);
	free(hA);

	return result;
}

// ------------------------------------------------------------------------------------

int main(int argc, char **argv) 
{
	std::cout << "Reduce DR ... " << std::flush << reduce_test(RAW) << std::endl;
	std::cout << "Reduce DW ... " << std::flush << reduce_test(WRAPPED) << std::endl;

	return 0;
}
