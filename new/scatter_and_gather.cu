/* The scattering on device is problematic. */

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

void scatter_test() {
	cout << "Scatter test ... " << flush;

	double *m_input;
	double *m_output;
	int    *m_map;

	cudaMallocManaged(&m_input, sizeof(double) * 10);
	cudaMallocManaged(&m_output , sizeof(double) * 10);
	cudaMallocManaged(&m_map, sizeof(int) * 10);

	m_map[0] = 9;
	m_map[1] = 6;
	m_map[2] = 8;
	m_map[3] = 0;
	m_map[4] = 4;
	m_map[5] = 2;
	m_map[6] = 3;
	m_map[7] = 7;
	m_map[8] = 5;
	m_map[9] = 1;

	for (int i = 0; i < 10; i++)
		m_input[i] = 10.0 + (i + 1);

	//// thrust::scatter(thrust::cuda::par, m_input, m_input + 10, m_map, m_output);
	{
		thrust::device_ptr<double> input_begin(m_input), input_end(m_input + 10), output_begin(m_output);
		thrust::device_ptr<int>    map_begin(m_map);
		thrust::scatter(thrust::cuda::par, input_begin, input_end, map_begin, output_begin);
	}
	cudaDeviceSynchronize();

	bool correct = true;
	for (int i = 0; i < 10; i++)
		if (m_output[m_map[i]] != m_input[i]) {
			correct = false;
			break;
		}

	cudaFree(m_map);
	cudaFree(m_input);
	cudaFree(m_output);

	if (correct)
		cout << "OK" << endl;
	else
		cout << "Failed" << endl;
}

void scatter_if_test() {
	cout << "Scatter if test ... " << flush;

	const double PRESET_VALUE = 10000.0;
	double *m_input;
	double *m_output;
	int    *m_map;
	bool   *m_stencil;
	bool   h_output_visited[10] = {0};

	cudaMallocManaged(&m_input, sizeof(double) * 10);
	cudaMallocManaged(&m_output , sizeof(double) * 10);
	cudaMallocManaged(&m_map, sizeof(int) * 10);
	cudaMallocManaged(&m_stencil, sizeof(bool) * 10);

	m_map[0] = 9;
	m_map[1] = 6;
	m_map[2] = 8;
	m_map[3] = 0;
	m_map[4] = 4;
	m_map[5] = 2;
	m_map[6] = 3;
	m_map[7] = 7;
	m_map[8] = 5;
	m_map[9] = 1;

	for (int i = 0; i < 10; i++) {
		m_input[i] = 10.0 + (i + 1);
		m_stencil[i] = ((i & 1) ? true : false);
		m_output[i] = PRESET_VALUE;
	}

	//// thrust::scatter_if(thrust::cuda::par, m_input, m_input + 10, m_map, m_stencil, m_output);
	{
		thrust::device_ptr<double> input_begin(m_input), input_end(m_input + 10), output_begin(m_output);
		thrust::device_ptr<int>    map_begin(m_map);
		thrust::device_ptr<bool>   stencil_begin(m_stencil);
		thrust::scatter_if(thrust::cuda::par, input_begin, input_end, map_begin, stencil_begin, output_begin, thrust::identity<bool>());
	}
	cudaDeviceSynchronize();
		//// thrust::scatter_if(h_input, h_input + 10, h_map, h_stencil, h_output, thrust::identity<bool>());

	bool correct = true;
	for (int i = 0; i < 10; i++)
		if (m_stencil[i]) {
			h_output_visited[m_map[i]] = true;
			if (m_output[m_map[i]] != m_input[i]) {
				correct = false;
				break;
			}
		}

	if (correct) {
		for (int i = 0; i < 10; i++)
			if (!h_output_visited[i]) {
				if (m_output[i] != PRESET_VALUE) {
					correct = false;
					break;
				}
			}
	}

	cudaFree(m_map);
	cudaFree(m_input);
	cudaFree(m_output);
	cudaFree(m_stencil);

	if (correct)
		cout << "OK" << endl;
	else
		cout << "Failed" << endl;
}

int main(int argc, char **argv) 
{
	scatter_test();
	scatter_if_test();
	return 0;
}
