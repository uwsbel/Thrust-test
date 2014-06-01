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

void transform_test() {
	cout << "Transform test ... " << flush;

	const int ARRAY_SIZE = 1000;

	double *mA;
	cudaMallocManaged(&mA, ARRAY_SIZE * sizeof(double));

	for (int i = 0; i < ARRAY_SIZE; i++)
		mA[i] = 1.0 * (i + 1);

	/* FIXME: it is not correct to use
	thrust::transform(thrust::cuda::par, mA, mA + ARRAY_SIZE, mA, thrust::negate<double>());
	*/
	{
		thrust::device_ptr<double> A_begin(mA);
		thrust::device_ptr<double> A_end(mA + ARRAY_SIZE);
		thrust::transform(thrust::cuda::par, A_begin, A_end, A_begin, thrust::negate<double>());
	}
	cudaDeviceSynchronize();

	bool correct = true;
	for (int i = 0; i < ARRAY_SIZE; i++)
		if (mA[i] != - 1.0 * (i + 1)) {
			correct = false;
			break;
		}

	cudaFree(mA);

	if (correct)
		cout << "OK" << endl;
	else
		cout << "Failed" << endl;
}

void transform_if_test() {
	cout << "Transform if test ... " << flush;

	const int ARRAY_SIZE = 1000;

	double *mA, *mB;
	int *m_stencil;

	cudaMallocManaged(&mA, ARRAY_SIZE * sizeof(double));
	cudaMallocManaged(&mB, ARRAY_SIZE * sizeof(double));
	cudaMallocManaged(&m_stencil, ARRAY_SIZE * sizeof(int));

	for (int i = 0; i < ARRAY_SIZE; i++)
		mB[i] = mA[i] = 1.0 * (i + 1);

	for (int i = 0; i < ARRAY_SIZE; i++) {
		if (i < (ARRAY_SIZE >> 1))
			m_stencil[i] = 1;
		else
			m_stencil[i] = 0;
	}

	/* FIXME: it is not correct to use 
	thrust::transform_if(thrust::cuda::par, mA, mA + ARRAY_SIZE, mB, m_stencil, mA, thrust::plus<double>(), thrust::identity<int>());
	*/

	{ 
		thrust::device_ptr<double> A_begin(mA);
		thrust::device_ptr<double> A_end(mA + ARRAY_SIZE);
		thrust::device_ptr<double> B_begin(mB);
		thrust::device_ptr<int>    stencil_begin(m_stencil);
		thrust::transform_if(thrust::cuda::par, A_begin, A_end, B_begin, stencil_begin, A_begin, thrust::plus<double>(), thrust::identity<int>());
	}
	cudaDeviceSynchronize();

	bool correct = true;
	for (int i = 0; i < (ARRAY_SIZE >> 1); i++)
		if (mA[i] != 2.0 * (i + 1)) {
			correct = false;
			break;
		}

	if (correct) {
		for (int i = (ARRAY_SIZE >> 1); i < ARRAY_SIZE; i++)
			if (mA[i] != 1.0 * (i + 1)) {
				correct = false;
				break;
			}
	}

	cudaFree(mA);
	cudaFree(mB);
	cudaFree(m_stencil);

	if (correct)
		cout << "OK" << endl;
	else
		cout << "Failed" << endl;
}

int main(int argc, char **argv) 
{
	transform_test();
	transform_if_test();
	return 0;
}
