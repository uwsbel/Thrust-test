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

using std::cout;
using std::cerr;
using std::endl;
using std::flush;

const int ARRAY_SIZE = 1000;

enum Order {
  ASCENDING,
  DESCENDING
};

enum Method {
  RAW,
  WRAPPED
};

// ------------------------------------------------------------------------------------

bool check_sort(Order order, double* mA)
{
  switch (order) {
  case ASCENDING:
    for (int i = 1; i < ARRAY_SIZE; i++) {
      if (mA[i] < mA[i-1])
        return false;
    }
    break;
  case DESCENDING:
    for (int i = 1; i < ARRAY_SIZE; i++) {
      if (mA[i] > mA[i-1])
        return false;
    }
    break;
  }

  return true;
}

bool sort_test(Order order, Method method) {
  double *mA;

  cudaMallocManaged(&mA, sizeof(double) * ARRAY_SIZE);

  for (int i = 0; i < ARRAY_SIZE; i++)
    mA[i] = 1.0 * (rand() % ARRAY_SIZE);

  switch (method) {
  case RAW:
    {
      switch (order) {
      case ASCENDING:
        thrust::sort(thrust::cuda::par, mA, mA + ARRAY_SIZE);
        break;
      case DESCENDING:
        thrust::sort(thrust::cuda::par, mA, mA + ARRAY_SIZE, thrust::greater<double>());
        break;
      default: break;
      }
      break;
    }
  case WRAPPED:
    {
      thrust::device_ptr<double> wmA(mA);
      switch (order) {
      case ASCENDING:
        thrust::sort(thrust::cuda::par, wmA, wmA + ARRAY_SIZE);
        break;
      case DESCENDING:
        thrust::sort(thrust::cuda::par, wmA, wmA + ARRAY_SIZE, thrust::greater<double>());
        break;
      default: break;
      }
      break;
    }
  default: break;
  }

  cudaDeviceSynchronize();

  bool result = check_sort(order, mA);

  cudaFree(mA);

  return result;
}

// ------------------------------------------------------------------------------------

void sort_by_key_test(Order order, Method method) {

  const int SIZE = 10;

  int    *m_keys;
  double *m_values;

  cudaMallocManaged(&m_keys, sizeof(int) * SIZE);
  cudaMallocManaged(&m_values, sizeof(double) * SIZE);

  m_keys[0] = 0;
  m_keys[1] = 2;
  m_keys[2] = 1;
  m_keys[3] = 4;
  m_keys[4] = 2;
  m_keys[5] = 4;
  m_keys[6] = 0;
  m_keys[7] = 1;
  m_keys[8] = 4;
  m_keys[9] = 2;

  m_values[0] = 8;
  m_values[1] = 2;
  m_values[2] = 8;
  m_values[3] = 7;
  m_values[4] = 8;
  m_values[5] = 3;
  m_values[6] = 5;
  m_values[7] = 3;
  m_values[8] = 7;
  m_values[9] = 4;

  std::cout << "     ";
  for (int i = 0; i < SIZE; i++) 
    std::cout << "(" << m_keys[i] << ", " << m_values[i] << ") ";

  std::cout << std::endl;

  switch (method) {
  case RAW:
    {
      switch (order) {
        case ASCENDING:
          thrust::sort_by_key(thrust::cuda::par, m_keys, m_keys + SIZE, m_values);
          break;
        case DESCENDING:
          thrust::sort_by_key(thrust::cuda::par, m_keys, m_keys + SIZE, m_values, thrust::greater<int>());
          break;
        default: break;
      }
      break;
    }
  case WRAPPED:
    {
      thrust::device_ptr<int> wmK = thrust::device_pointer_cast(m_keys);
      thrust::device_ptr<double> wmV = thrust::device_pointer_cast(m_values);
      switch (order) {
        case ASCENDING:
          thrust::sort_by_key(thrust::cuda::par, wmK, wmK + SIZE, wmV);
          break;
        case DESCENDING:
          thrust::sort_by_key(thrust::cuda::par, wmK, wmK + SIZE, wmV , thrust::greater<int>());
          break;
        default: break;
      }
      break;
    }
  }
  cudaDeviceSynchronize();

  std::cout << "     ";
  for (int i = 0; i < SIZE; i++)
    std::cout << "(" << m_keys[i] << ", " << m_values[i] << ") ";
  std::cout << std::endl;


  cudaFree(m_keys);
  cudaFree(m_values);
}

// ------------------------------------------------------------------------------------

int main(int argc, char **argv)
{
  int major = THRUST_MAJOR_VERSION;
  int minor = THRUST_MINOR_VERSION;
  std::cout << "Thrust v" << major << "." << minor << std::endl << std::endl;

  std::cout << "Sort ascending DMR ...  " << std::flush << sort_test(ASCENDING, RAW) << std::endl;
  std::cout << "Sort descending DMR ... " << std::flush << sort_test(DESCENDING, RAW) << std::endl;

  std::cout << "Sort ascending DMW ...  " << std::flush << sort_test(ASCENDING, WRAPPED) << std::endl;
  std::cout << "Sort descending DMW ... " << std::flush << sort_test(DESCENDING, WRAPPED) << std::endl;

  std::cout << std::endl << std::endl;

  std::cout << "Sort_by_key ascending DMR:" << std::endl;
  sort_by_key_test(ASCENDING, RAW);
  std::cout << "Sort_by_key descending DMR:" << std::endl;
  sort_by_key_test(DESCENDING, RAW);

  std::cout << std::endl << std::endl;

  std::cout << "Sort_by_key ascending DMW:" << std::endl;
  sort_by_key_test(ASCENDING, WRAPPED);
  std::cout << "Sort_by_key descending DMW:" << std::endl;
  sort_by_key_test(DESCENDING, WRAPPED);

  return 0;
}
