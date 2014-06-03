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

using std::cout;
using std::cerr;
using std::endl;
using std::flush;

void fill_test() {
	cout << "Fill test ... " << flush;

	const int ARRAY_SIZE = 1000;

	double *mA;

	cudaMallocManaged(&mA, sizeof(double) * ARRAY_SIZE);

	for (int i = 0; i < ARRAY_SIZE; i++)
		mA[i] = 0.0;

	thrust::fill(thrust::cuda::par, mA, mA + ARRAY_SIZE, 9.0);
	cudaDeviceSynchronize();

	bool correct = true;
	for (int i = 0; i < ARRAY_SIZE; i++)
		if (mA[i] != 9.0) {
			correct = false;
			break;
		}

	cudaFree(mA);

	if (correct)
		cout << "OK" << endl;
	else
		cout << "Failed" << endl;
}

void copy_test() {
	cout << "Copy test ... " << flush;

	const int ARRAY_SIZE = 1000;

	double *mA, *mB;

	cudaMallocManaged(&mA, sizeof(double) * ARRAY_SIZE);
	cudaMallocManaged(&mB, sizeof(double) * ARRAY_SIZE);

	for (int i = 0; i < ARRAY_SIZE; i++)
		mA[i] = 1.0 * (i+1);

	thrust::copy(thrust::cuda::par, mA, mA + ARRAY_SIZE, mB);
	cudaDeviceSynchronize();

	bool correct = true;
	for (int i = 0; i < ARRAY_SIZE; i++)
		if (mB[i] != 1.0 * (i + 1)) {
			correct = false;
			break;
		}

	cudaFree(mA);
	cudaFree(mB);

	if (correct)
		cout << "OK" << endl;
	else
		cout << "Failed" << endl;

}

int main(int argc, char **argv) 
{
	fill_test();
	copy_test();
	return 0;
}
