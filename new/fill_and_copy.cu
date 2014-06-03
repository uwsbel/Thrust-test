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

bool check_fill(double* mA)
{
	for (int i = 0; i < ARRAY_SIZE; i++) {
		if (mA[i] != 9.0)
			return false;
	}

	return true;
}

bool fill_test(Method method) {
	double *mA;

	cudaMallocManaged(&mA, sizeof(double) * ARRAY_SIZE);

	for (int i = 0; i < ARRAY_SIZE; i++)
		mA[i] = 0.0;

	switch (method)
	{
	case RAW:
		thrust::fill(thrust::cuda::par, mA, mA + ARRAY_SIZE, 9.0);
		break;
	case WRAPPED:
		{
			thrust::device_ptr<double> wmA(mA);
			thrust::fill(thrust::cuda::par, wmA, wmA + ARRAY_SIZE, 9.0);
			break;
		}
	default: break;
	}
	cudaDeviceSynchronize();

	bool result = check_fill(mA);
	cudaFree(mA);

	return result;
}

// ------------------------------------------------------------------------------------

bool check_copy(double* mB)
{
	for (int i = 0; i < ARRAY_SIZE; i++) {
		if (mB[i] != 1.0 * (i + 1))
			return false;
	}

	return true;
}

bool copy_test(Method method) {
	double *mA, *mB;

	cudaMallocManaged(&mA, sizeof(double) * ARRAY_SIZE);
	cudaMallocManaged(&mB, sizeof(double) * ARRAY_SIZE);

	for (int i = 0; i < ARRAY_SIZE; i++)
	{
		mA[i] = 1.0 * (i+1);
		mB[i] = 0.0;
	}

	switch (method) {
	case RAW:
		thrust::copy(thrust::cuda::par, mA, mA + ARRAY_SIZE, mB);
		break;
	case WRAPPED:
		{
			thrust::device_ptr<double> wmA(mA), wmB(mB);
			thrust::copy(thrust::cuda::par, wmA, wmA + ARRAY_SIZE, wmB);
			break;
		}
	default: break;
	}
	cudaDeviceSynchronize();

	bool result = check_copy(mB);

	cudaFree(mA);
	cudaFree(mB);

	return result;
}

// ------------------------------------------------------------------------------------

int main(int argc, char **argv)
{
	std::cout << "Fill DMR ... " << std::flush << fill_test(RAW) << std::endl;
	std::cout << "Fill DMW ... " << std::flush << fill_test(WRAPPED) << std::endl;

	std::cout << "Copy DMR ... " << std::flush << copy_test(RAW) << std::endl;
	std::cout << "Copy DMW ... " << std::flush << copy_test(WRAPPED) << std::endl;

	return 0;
}
