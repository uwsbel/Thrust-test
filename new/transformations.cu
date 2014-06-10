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

const int ARRAY_SIZE = 1000;

enum Method {
  RAW,
  WRAPPED
};

bool check_transform(double* mA)
{
  for (int i = 0; i < ARRAY_SIZE; i++) {
    if (mA[i] != - 1.0 * (i + 1))
      return false;
  }

  return true;
}

bool transform_test(Method method)
{
  double *mA;
  cudaMallocManaged(&mA, ARRAY_SIZE * sizeof(double));

  for (int i = 0; i < ARRAY_SIZE; i++)
    mA[i] = 1.0 * (i + 1);

  switch (method) {
  case RAW:
    thrust::transform(thrust::cuda::par, mA, mA + ARRAY_SIZE, mA, thrust::negate<double>());
    break;
  case WRAPPED:
    {
      thrust::device_ptr<double> A_begin(mA);
      thrust::device_ptr<double> A_end(mA + ARRAY_SIZE);
      thrust::transform(thrust::cuda::par, A_begin, A_end, A_begin, thrust::negate<double>());
      break;
    }
  default:
    break;
  }
  cudaDeviceSynchronize();

  bool result = check_transform(mA);

  cudaFree(mA);

  return result;
}

bool check_transform_if(double* mA)
{
  for (int i = 0; i < (ARRAY_SIZE >> 1); i++) {
    if (mA[i] != 2.0 * (i + 1))
      return false;
  }

  for (int i = (ARRAY_SIZE >> 1); i < ARRAY_SIZE; i++) {
    if (mA[i] != 1.0 * (i + 1))
      return false;
  }

  return true;
}

bool transform_if_test(Method method)
{
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

  switch (method) {
  case RAW:
    thrust::transform_if(thrust::cuda::par, mA, mA + ARRAY_SIZE, mB, m_stencil, mA, thrust::plus<double>(), thrust::identity<int>());
    break;
  case WRAPPED:
    { 
      thrust::device_ptr<double> A_begin(mA);
      thrust::device_ptr<double> A_end(mA + ARRAY_SIZE);
      thrust::device_ptr<double> B_begin(mB);
      thrust::device_ptr<int>    stencil_begin(m_stencil);
      thrust::transform_if(thrust::cuda::par, A_begin, A_end, B_begin, stencil_begin, A_begin, thrust::plus<double>(), thrust::identity<int>());
      break;
    }
  default: break;
  }
  cudaDeviceSynchronize();

  bool result = check_transform_if(mA);

  cudaFree(mA);
  cudaFree(mB);
  cudaFree(m_stencil);

  return result;
}

bool check_sequence(double *mA) {
  for (int i = 0; i < ARRAY_SIZE; i++)
    if (mA[i] != 1.0 * i)
      return false;

  return true;
}

bool sequence_test(Method method) {
  double *mA;

  cudaMallocManaged(&mA, sizeof(double) * ARRAY_SIZE);

  for (int i = 0; i < ARRAY_SIZE; i++)
    mA[i] = 0.0;

  switch (method) {
  case RAW:
      thrust::sequence(thrust::cuda::par, mA, mA + ARRAY_SIZE);
    break;
  case WRAPPED:
    {
      thrust::device_ptr<double> A_begin(mA);
      thrust::device_ptr<double> A_end(mA + ARRAY_SIZE);
      thrust::sequence(thrust::cuda::par, A_begin, A_end);
      break;
    }
  default:
    break;
  }
  cudaDeviceSynchronize();

  bool result = check_sequence(mA);

  cudaFree(mA);

  return result;
}

bool check_tabulate(double *mA)
{
  for (int i = 0; i < ARRAY_SIZE; i++)
    if (mA[i] != -1.0 * i)
      return false;

  return true;
}

bool tabulate_test(Method method)
{
  double *mA;

  cudaMallocManaged(&mA, sizeof(double) * ARRAY_SIZE);

  for (int i = 0; i < ARRAY_SIZE; i++)
    mA[i] = 0.0;

  switch (method) {
  case RAW:
    thrust::tabulate(thrust::cuda::par, mA, mA + ARRAY_SIZE, thrust::negate<double>());
    break;
  case WRAPPED:
    {
      thrust::device_ptr<double> A_begin(mA), A_end(mA + ARRAY_SIZE);
      thrust::tabulate(thrust::cuda::par, A_begin, A_end, thrust::negate<double>());
      break;
    }
  default: break;
  }
  cudaDeviceSynchronize();

  bool result = check_tabulate(mA);
  cudaFree(mA);

  return result;
}

int main(int argc, char **argv) 
{
  int major = THRUST_MAJOR_VERSION;
  int minor = THRUST_MINOR_VERSION;
  std::cout << "Thrust v" << major << "." << minor << std::endl << std::endl;

  std::cout << "Transform DMR ... " << std::flush << transform_test(RAW) << std::endl;
  std::cout << "Transform DMW ... " << std::flush << transform_test(WRAPPED) << std::endl;

  std::cout << "Transform_if DMR ... " << std::flush << transform_if_test(RAW) << std::endl;
  std::cout << "Transform_if DMW ... " << std::flush << transform_if_test(WRAPPED) << std::endl;

  std::cout << "Sequence DMR ... " << std::flush << sequence_test(RAW) << std::endl;
  std::cout << "Sequence DMW ... " << std::flush << sequence_test(WRAPPED) << std::endl;

  std::cout << "Tabulate DMR ... " << std::flush << tabulate_test(RAW) << std::endl;
  std::cout << "Tabulate DMW ... " << std::flush << tabulate_test(WRAPPED) << std::endl;
  return 0;
}
