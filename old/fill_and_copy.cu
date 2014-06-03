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
#include <thrust/system/cuda/execution_policy.h>

const int ARRAY_SIZE = 1000;

enum Method {
	RAW,
	WRAPPED
};

// ------------------------------------------------------------------------------------

bool check_fill(double* hA)
{
	for (int i = 0; i < ARRAY_SIZE; i++) {
		if (hA[i] != 9.0)
			return false;
	}

	return true;
}

bool fill_test(Method method)
{
	double* hA;
	hA = (double *) malloc(sizeof(double) * ARRAY_SIZE);
	for (int i = 0; i < ARRAY_SIZE; i++)
		hA[i] = 0.0;

	double* dA;
	cudaMalloc((void **) &dA, sizeof(double) * ARRAY_SIZE);
	cudaMemcpy(dA, hA, sizeof(double) * ARRAY_SIZE, cudaMemcpyHostToDevice);

	switch (method) {
	case RAW:
		{
			thrust::fill(thrust::cuda::par, dA, dA + ARRAY_SIZE, 9.0);
			break;
		}
	case WRAPPED:
		{
			thrust::device_ptr<double> wdA = thrust::device_pointer_cast(dA);
			thrust::fill(wdA, wdA + ARRAY_SIZE, 9.0);
			break;
		}
	}

	cudaMemcpy(hA, dA, sizeof(double) * ARRAY_SIZE, cudaMemcpyDeviceToHost);
	bool result = check_fill(hA);

	free(hA);
	cudaFree(dA);

	return result;
}

// ------------------------------------------------------------------------------------

bool check_copy(double* hB)
{
	for (int i = 0; i < ARRAY_SIZE; i++) {
		if (hB[i] != 1.0 * (i + 1))
			return false;
	}

	return true;
}

bool copy_test(Method method)
{
	double* hA;
	double* hB;
	hA = (double *) malloc(sizeof(double) * ARRAY_SIZE);
	hB = (double *) malloc(sizeof(double) * ARRAY_SIZE);
	for (int i = 0; i < ARRAY_SIZE; i++) {
		hA[i] = 1.0 * (i+1);
		hB[i] = 0.0;
	}

	double* dA;
	double* dB;
	cudaMalloc((void **) &dA, sizeof(double) * ARRAY_SIZE);
	cudaMalloc((void **) &dB, sizeof(double) * ARRAY_SIZE);
	cudaMemcpy(dA, hA, sizeof(double) * ARRAY_SIZE, cudaMemcpyHostToDevice);
	cudaMemcpy(dB, hB, sizeof(double) * ARRAY_SIZE, cudaMemcpyHostToDevice);

	switch (method) {
	case RAW:
		{
			thrust::copy(thrust::cuda::par, dA, dA + ARRAY_SIZE, dB);
			break;
		}
	case WRAPPED:
		{
			thrust::device_ptr<double> wdA = thrust::device_pointer_cast(dA);
			thrust::device_ptr<double> wdB = thrust::device_pointer_cast(dB);
			thrust::copy(wdA, wdA + ARRAY_SIZE, wdB);
		}
	}

	cudaMemcpy(hB, dB, sizeof(double) * ARRAY_SIZE, cudaMemcpyDeviceToHost);
	bool result = check_copy(hB);

	free(hA);
	free(hB);
	cudaFree(dA);
	cudaFree(dB);

	return result;
}

// ------------------------------------------------------------------------------------

int main(int argc, char **argv)
{
	std::cout << "Fill DR ... " << std::flush << fill_test(RAW) << std::endl;
	std::cout << "Fill DW ... " << std::flush << fill_test(WRAPPED) << std::endl;

	std::cout << "Copy DR ... " << std::flush << copy_test(RAW) << std::endl;
	std::cout << "Copy DW ... " << std::flush << copy_test(WRAPPED) << std::endl;

	return 0;
}
