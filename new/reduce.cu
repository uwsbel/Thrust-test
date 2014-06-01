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

using std::cout;
using std::cerr;
using std::endl;
using std::flush;

void reduce_test() {
	cout << "Reduce ... " << flush;

	const int ARRAY_SIZE = 1000;

	double *mA;
	cudaMallocManaged(&mA, ARRAY_SIZE * sizeof(double));

	for (int i = 0; i < ARRAY_SIZE; i++)
		mA[i] = 1.0 * (i + 1);

	double maximum = thrust::reduce(thrust::cuda::par, mA, mA + ARRAY_SIZE, 0.0, thrust::maximum<double>());
	//// double maximum = 0.0;
	{
		// thrust::device_ptr<double> A_begin(mA), A_end(mA + ARRAY_SIZE);
		// maximum = thrust::reduce(A_begin, A_end , 0.0, thrust::maximum<double>());
	}
	cudaDeviceSynchronize();

	bool correct = (fabs(maximum - ARRAY_SIZE) < 1e-10);

	cudaFree(mA);

	if (correct)
		cout << "OK" << endl;
	else
		cout << "Failed" << endl;
}

int main(int argc, char **argv) 
{
	reduce_test();
	return 0;
}
