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

void sort_test(bool decrease = false) {
	cout << "Sort ";
	if (decrease)
		cout << "non-increasingly test ... " << flush;
	else
		cout << "non-decreasingly test ... " << flush;

	const int ARRAY_SIZE = 1000;

	double *mA;

	cudaMallocManaged(&mA, sizeof(double) * ARRAY_SIZE);

	for (int i = 0; i < ARRAY_SIZE; i++)
		mA[i] = 1.0 * (rand() % ARRAY_SIZE);

	if (decrease)
		thrust::sort(thrust::cuda::par, mA, mA + ARRAY_SIZE, thrust::greater<double>());
	else
		thrust::sort(thrust::cuda::par, mA, mA + ARRAY_SIZE);

	cudaDeviceSynchronize();

	bool correct = true;
	for (int i = 1; i < ARRAY_SIZE; i++) {
		if (decrease ? (mA[i] > mA[i-1]) : (mA[i] < mA[i-1])) {
			correct = false;
			break;
		}
	}

	cudaFree(mA);

	if (correct)
		cout << "OK" << endl;
	else
		cout << "Failed" << endl;
}

void sort_by_key_test(bool decrease = false) {
	cout << "Sort by key ";
	if (decrease)
		cout << "non-increasingly test ... " << endl;
	else
		cout << "non-decreasingly test ... " << endl;

	const int ARRAY_SIZE = 10;

	int    *m_keys;
	double *m_values;

	cudaMallocManaged(&m_keys, sizeof(int) * ARRAY_SIZE);
	cudaMallocManaged(&m_values, sizeof(double) * ARRAY_SIZE);

	cout << "Before: " << endl;
	for (int i = 0; i < ARRAY_SIZE; i++) {
		m_keys[i] = rand() % (ARRAY_SIZE >> 1);
		m_values[i] = 1.0 * (rand() % ARRAY_SIZE);
		cout << "(" << m_keys[i] << ", " << m_values[i] << ") ";
	}
	cout << endl;

	if (decrease) {
		thrust::device_ptr<int> keys_begin = thrust::device_pointer_cast(m_keys);
		thrust::device_ptr<int> keys_end   = thrust::device_pointer_cast(m_keys + ARRAY_SIZE);
		thrust::device_ptr<double> values_begin = thrust::device_pointer_cast(m_values);
		thrust::sort_by_key(thrust::cuda::par, keys_begin, keys_end, values_begin, thrust::greater<int>());
		/* FIXME: The program gets wrong results if we do:
		   thrust::sort_by_key(thrust::cuda::par, d_keys, d_keys + ARRAY_SIZE, d_values, thrust::greater<int>());
		   which does not seem to make much sense.
		   Also note that the behavior of sort_by_keys seems to be different from that of sort.
		   */
	}
	else
		thrust::sort_by_key(thrust::cuda::par, m_keys, m_keys + ARRAY_SIZE, m_values);
	cudaDeviceSynchronize();

	cout << "After: " << endl;
	for (int i = 0; i < ARRAY_SIZE; i++)
		cout << "(" << m_keys[i] << ", " << m_values[i] << ") ";
	cout << endl;


	cudaFree(m_keys);
	cudaFree(m_values);
}

int main(int argc, char **argv)
{
	sort_test(false);
	sort_test(true);
	sort_by_key_test(false);
	sort_by_key_test(true);
	return 0;
}
