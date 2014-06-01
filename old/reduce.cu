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

	double *hA, *dA;
	hA = (double *) malloc(ARRAY_SIZE * sizeof(double));
	cudaMalloc(&dA, ARRAY_SIZE * sizeof(double));

	for (int i = 0; i < ARRAY_SIZE; i++)
		hA[i] = 1.0 * (i + 1);

	cudaMemcpy(dA, hA, sizeof(double) * ARRAY_SIZE, cudaMemcpyHostToDevice);

	double maximum = thrust::reduce(thrust::cuda::par, dA, dA + ARRAY_SIZE, 0.0, thrust::maximum<double>());
	//// double maximum;
	{
		//// thrust::device_ptr<double> A_begin(dA), A_end(dA + ARRAY_SIZE);
		//// maximum = thrust::reduce(A_begin, A_end , 0.0, thrust::maximum<double>());
	}

	bool correct = (fabs(maximum - ARRAY_SIZE) < 1e-10);

	cudaFree(dA);
	free(hA);

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
