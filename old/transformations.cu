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

bool check_transform(double* hA)
{
	for (int i = 0; i < ARRAY_SIZE; i++) {
		if (hA[i] != - 1.0 * (i + 1))
			return false;
	}

	return true;
}

bool transform_test(Method method)
{
	double *hA;
	hA = (double *) malloc(ARRAY_SIZE * sizeof(double));
	for (int i = 0; i < ARRAY_SIZE; i++)
		hA[i] = 1.0 * (i + 1);

	double* dA;
	cudaMalloc(&dA, ARRAY_SIZE * sizeof(double));
	cudaMemcpy(dA, hA, sizeof(double) * ARRAY_SIZE, cudaMemcpyHostToDevice);

	switch (method) {
	case RAW:
		{
			thrust::transform(thrust::cuda::par, dA, dA + ARRAY_SIZE, dA, thrust::negate<double>());
			break;
		}
	case WRAPPED:
		{
			thrust::device_ptr<double> A_begin(dA);
			thrust::device_ptr<double> A_end(dA + ARRAY_SIZE);
			thrust::transform(thrust::cuda::par, A_begin, A_end, A_begin, thrust::negate<double>());
			break;
		}
	}

	cudaMemcpy(hA, dA, sizeof(double) * ARRAY_SIZE, cudaMemcpyDeviceToHost);
	bool result = check_transform(hA);

	cudaFree(dA);
	free(hA);

	return result;
}

// ------------------------------------------------------------------------------------

bool check_transform_if(double* hA)
{
	for (int i = 0; i < (ARRAY_SIZE >> 1); i++) {
		if (hA[i] != 2.0 * (i + 1))
			return false;
	}

	for (int i = (ARRAY_SIZE >> 1); i < ARRAY_SIZE; i++) {
		if (hA[i] != 1.0 * (i + 1))
			return false;
	}

	return true;
}


bool transform_if_test(Method method)
{
	double* hA;
	double* hB;
	int*    h_stencil;
	hA = (double *) malloc(ARRAY_SIZE * sizeof(double));
	hB = (double *) malloc(ARRAY_SIZE * sizeof(double));
	h_stencil = (int *) malloc(ARRAY_SIZE * sizeof(int));
	for (int i = 0; i < ARRAY_SIZE; i++)
		hB[i] = hA[i] = 1.0 * (i + 1);

	for (int i = 0; i < ARRAY_SIZE; i++) {
		if (i < (ARRAY_SIZE >> 1))
			h_stencil[i] = 1;
		else
			h_stencil[i] = 0;
	}

	double* dA;
	double* dB;
	int*    d_stencil;
	cudaMalloc(&dA, ARRAY_SIZE * sizeof(double));
	cudaMalloc(&dB, ARRAY_SIZE * sizeof(double));
	cudaMalloc(&d_stencil, ARRAY_SIZE * sizeof(int));
	cudaMemcpy(dA, hA, sizeof(double) * ARRAY_SIZE, cudaMemcpyHostToDevice);
	cudaMemcpy(dB, hB, sizeof(double) * ARRAY_SIZE, cudaMemcpyHostToDevice);
	cudaMemcpy(d_stencil, h_stencil, sizeof(int) * ARRAY_SIZE, cudaMemcpyHostToDevice);

	switch (method) {
	case RAW:
		{
			thrust::transform_if(thrust::cuda::par, dA, dA + ARRAY_SIZE, dB, d_stencil, dA, thrust::plus<double>(), thrust::identity<int>());
			break;
		}
	case WRAPPED:
		{
			thrust::device_ptr<double> A_begin(dA);
			thrust::device_ptr<double> A_end(dA + ARRAY_SIZE);
			thrust::device_ptr<double> B_begin(dB);
			thrust::device_ptr<int>    stencil_begin(d_stencil);
			thrust::transform_if(thrust::cuda::par, A_begin, A_end, B_begin, stencil_begin, A_begin, thrust::plus<double>(), thrust::identity<int>());
			break;
		}
	}

	cudaMemcpy(hA, dA, sizeof(double) * ARRAY_SIZE, cudaMemcpyDeviceToHost);
	bool result = check_transform_if(hA);

	cudaFree(dA);
	cudaFree(dB);
	cudaFree(d_stencil);
	free(hA);
	free(hB);
	free(h_stencil);

	return result;
}

// ------------------------------------------------------------------------------------

int main(int argc, char **argv) 
{
	std::cout << "Transform DR ... " << std::flush << transform_test(RAW) << std::endl;
	std::cout << "Transform DW ... " << std::flush << transform_test(WRAPPED) << std::endl;

	std::cout << "Transform_if DR ... " << std::flush << transform_if_test(RAW) << std::endl;
	std::cout << "Transform_if DW ... " << std::flush << transform_if_test(WRAPPED) << std::endl;

	return 0;
}
