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
	double *mA;
	cudaMallocManaged(&mA, ARRAY_SIZE * sizeof(double));

	for (int i = 0; i < ARRAY_SIZE; i++)
		mA[i] = 1.0 * (i + 1);

	double maximum;

	switch (method) {
	case RAW:
		{
			maximum = thrust::reduce(thrust::cuda::par, mA, mA + ARRAY_SIZE, 0.0, thrust::maximum<double>());
			break;
		}
	case WRAPPED:
		{
			thrust::device_ptr<double> A_begin(mA), A_end(mA + ARRAY_SIZE);
			maximum = thrust::reduce(A_begin, A_end , 0.0, thrust::maximum<double>());
			break;
		}
	default:
		break;
	}
	cudaDeviceSynchronize();

	bool result = (fabs(maximum - ARRAY_SIZE) < 1e-10);

	cudaFree(mA);

	return result;
}

int main(int argc, char **argv) 
{
	std::cout << "Reduce DMR ... " << std::flush << reduce_test(RAW) << std::endl;
	std::cout << "Reduce DMW ... " << std::flush << reduce_test(WRAPPED) << std::endl;
	return 0;
}
