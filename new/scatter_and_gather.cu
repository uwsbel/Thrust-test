#include <iostream>
#include <cmath>

#include <thrust/version.h>
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

const int    ARRAY_SIZE = 10;
const double PRESET_VALUE = 10000.0;

enum Method {
  RAW,
  WRAPPED
};

bool check_scatter(const double *m_output, const double *m_input, const int *m_map)
{
  for (int i = 0; i < ARRAY_SIZE; i++)
    if (m_output[m_map[i]] != m_input[i])
      return false;

  return true;
}

bool scatter_test(Method method) {
  double *m_input;
  double *m_output;
  int    *m_map;

  cudaMallocManaged(&m_input, sizeof(double) * ARRAY_SIZE);
  cudaMallocManaged(&m_output , sizeof(double) * ARRAY_SIZE);
  cudaMallocManaged(&m_map, sizeof(int) * ARRAY_SIZE);

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

  for (int i = 0; i < ARRAY_SIZE; i++) {
    m_input[i]  = 10.0 + (i + 1);
    m_output[i] = PRESET_VALUE;
  }

  switch (method) 
  {
  case RAW:
    thrust::scatter(thrust::cuda::par, m_input, m_input + ARRAY_SIZE, m_map, m_output);
    break;
  case WRAPPED:
    {
      thrust::device_ptr<double> wInput(m_input), wOutput(m_output);
      thrust::device_ptr<int>    wMap(m_map);
      thrust::scatter(thrust::cuda::par, wInput, wInput + ARRAY_SIZE, wMap, wOutput);
      break;
    }
  default: break;
  }
  cudaDeviceSynchronize();

  bool result = check_scatter(m_output, m_input, m_map);

  cudaFree(m_map);
  cudaFree(m_input);
  cudaFree(m_output);

  return result;
}

bool check_scatter_if(const double *m_output, const double *m_input, const int *m_map, const bool *m_stencil)
{
  bool   h_output_visited[ARRAY_SIZE] = {0};

  for (int i = 0; i < ARRAY_SIZE; i++)
    if (m_stencil[i]) {
      h_output_visited[m_map[i]] = true;
      if (m_output[m_map[i]] != m_input[i])
        return false;
    }

  for (int i = 0; i < ARRAY_SIZE; i++)
    if (!h_output_visited[i]) {
      if (m_output[i] != PRESET_VALUE)
        return false;
    }

  return true;
}

bool scatter_if_test(Method method) {
  double *m_input;
  double *m_output;
  int    *m_map;
  bool   *m_stencil;

  cudaMallocManaged(&m_input, sizeof(double) * ARRAY_SIZE);
  cudaMallocManaged(&m_output , sizeof(double) * ARRAY_SIZE);
  cudaMallocManaged(&m_map, sizeof(int) * ARRAY_SIZE);
  cudaMallocManaged(&m_stencil, sizeof(bool) * ARRAY_SIZE);

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

  for (int i = 0; i < ARRAY_SIZE; i++) {
    m_input[i] = 10.0 + (i + 1);
    m_stencil[i] = ((i & 1) ? true : false);
    m_output[i] = PRESET_VALUE;
  }

  switch(method)
  {
  case RAW:
    thrust::scatter_if(thrust::cuda::par, m_input, m_input + ARRAY_SIZE, m_map, m_stencil, m_output);
    break;
  case WRAPPED:
    {
      thrust::device_ptr<double> wInput(m_input), wOutput(m_output);
      thrust::device_ptr<int>    wMap(m_map);
      thrust::device_ptr<bool>   wStencil(m_stencil);
      thrust::scatter_if(thrust::cuda::par, wInput, wInput + ARRAY_SIZE, wMap, wStencil, wOutput, thrust::identity<bool>());
      break;
    }
  default: break;
  }
  cudaDeviceSynchronize();

  bool result = check_scatter_if(m_output, m_input, m_map, m_stencil);

  cudaFree(m_map);
  cudaFree(m_input);
  cudaFree(m_output);
  cudaFree(m_stencil);

  return result;
}

bool check_gather(const double *m_output, const double *m_input, const int *m_map)
{
  for (int i = 0; i < ARRAY_SIZE; i++)
    if (m_output[i] != m_input[m_map[i]])
      return false;

  return true;
}

bool gather_test(Method method) {
  double *m_input;
  double *m_output;
  int    *m_map;

  cudaMallocManaged(&m_input, sizeof(double) * ARRAY_SIZE);
  cudaMallocManaged(&m_output , sizeof(double) * ARRAY_SIZE);
  cudaMallocManaged(&m_map, sizeof(int) * ARRAY_SIZE);

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

  for (int i = 0; i < ARRAY_SIZE; i++) {
    m_input[i]  = 10.0 + (i + 1);
    m_output[i] = PRESET_VALUE;
  }

  switch (method) {
    case RAW:
      thrust::gather(thrust::cuda::par, m_map, m_map + ARRAY_SIZE, m_input, m_output);
      break;
    case WRAPPED:
      {
        thrust::device_ptr<double> wInput(m_input), wOutput(m_output);
        thrust::device_ptr<int>    wMap(m_map);
        thrust::gather(thrust::cuda::par, wMap, wMap + ARRAY_SIZE, wInput, wOutput);
        break;
      }
    default:
      break;
  }
  cudaDeviceSynchronize();

  bool result = check_gather(m_output, m_input, m_map);

  cudaFree(m_map);
  cudaFree(m_input);
  cudaFree(m_output);

  return result;
}

bool check_gather_if(const double *mOutput, const double *mInput, const int * mMap, const bool *mStencil)
{
  for (int i = 0; i < ARRAY_SIZE; i++) {
    if (mStencil[i]) {
      if (mOutput[i] != mInput[mMap[i]]) 
        return false;
    } else {
      if (mOutput[i] != PRESET_VALUE)
        return false;
    }
  }
  return true;
}

bool gather_if_test(Method method) {
  double *m_input;
  double *m_output;
  int    *m_map;
  bool   *m_stencil;

  cudaMallocManaged(&m_input, sizeof(double) * ARRAY_SIZE);
  cudaMallocManaged(&m_output , sizeof(double) * ARRAY_SIZE);
  cudaMallocManaged(&m_map, sizeof(int) * ARRAY_SIZE);
  cudaMallocManaged(&m_stencil, sizeof(bool) * ARRAY_SIZE);

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

  for (int i = 0; i < ARRAY_SIZE; i++) {
    m_input[i] = 10.0 + (i + 1);
    m_stencil[i] = ((i & 1) ? true : false);
    m_output[i] = PRESET_VALUE;
  }

  switch(method){
  case RAW:
    thrust::gather_if(thrust::cuda::par, m_map, m_map + ARRAY_SIZE, m_stencil, m_input, m_output, thrust::identity<bool>());
    break;
  case WRAPPED:
    {
      thrust::device_ptr<double> wInput(m_input), wOutput(m_output);
      thrust::device_ptr<int>    wMap(m_map);
      thrust::device_ptr<bool>   wStencil(m_stencil);
      thrust::gather_if(thrust::cuda::par, wMap, wMap + ARRAY_SIZE, wStencil, wInput, wOutput, thrust::identity<bool>());
      break;
    }
  default: break;
  }
  cudaDeviceSynchronize();

  bool result = check_gather_if(m_output, m_input, m_map, m_stencil);

  cudaFree(m_map);
  cudaFree(m_input);
  cudaFree(m_output);
  cudaFree(m_stencil);

  return result;
}

int main(int argc, char **argv) 
{
  int major = THRUST_MAJOR_VERSION;
  int minor = THRUST_MINOR_VERSION;
  std::cout << "Thrust v" << major << "." << minor << std::endl << std::endl;

  std::cout << "Scatter DMR ... " << std::flush << scatter_test(RAW) << std::endl;
  std::cout << "Scatter DMW ... " << std::flush << scatter_test(WRAPPED) << std::endl;
  std::cout << "Scatter_if DMR ... " << std::flush << scatter_if_test(RAW) << std::endl;
  std::cout << "Scatter_if DMW ... " << std::flush << scatter_if_test(WRAPPED) << std::endl;
  std::cout << "Gather DMR ... " << std::flush << gather_test(RAW) << std::endl;
  std::cout << "Gather DMW ... " << std::flush << gather_test(WRAPPED) << std::endl;
  std::cout << "Gather_if DMR ... " << std::flush << gather_if_test(RAW) << std::endl;
  std::cout << "Gather_if DMW ... " << std::flush << gather_if_test(WRAPPED) << std::endl;
  return 0;
}
