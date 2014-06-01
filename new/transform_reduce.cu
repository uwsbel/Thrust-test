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

	double *mA, *mB;

	cudaMallocManaged(&mA, sizeof(double) * ARRAY_SIZE);
	cudaMallocManaged(&mB, sizeof(double) * ARRAY_SIZE);

	for (int i = 0; i < ARRAY_SIZE; i++) {
		mA[i] = 1.0 * (i+1);
		mB[i] = 1.0 * (ARRAY_SIZE - i);
	}

	//// double inner_product = thrust::inner_product(thrust::cuda::par, mA, mA + ARRAY_SIZE, mB, 0.0, thrust::plus<double>(), thrust::multiplies<double>());
	double inner_product;
	{
		thrust::device_ptr<double> A_begin(mA), A_end(mA + ARRAY_SIZE), B_begin(mB);
		inner_product = thrust::inner_product(thrust::cuda::par, A_begin, A_end , B_begin, 0.0, thrust::plus<double>(), thrust::multiplies<double>());
	}
	cudaDeviceSynchronize();

	double ref_inner_product = 0.0;

	for (int i = 0; i < ARRAY_SIZE; i++)
		ref_inner_product += mA[i] * mB[i];

	bool correct = (fabs(inner_product - ref_inner_product) / fabs(ref_inner_product) < 1e-10);

	cudaFree(mA);
	cudaFree(mB);

	if (correct)
		cout << "OK" << endl;
	else
		cout << "Failed" << endl;
}

int main(int argc, char **argv) {
	inner_product_test();
	return 0;
}
