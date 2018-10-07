#include <iostream>
#include <chrono>
#include <string>
#include <thread>
#include <vector>
#include <atomic>
#include <future>
#include <stdexcept>
#include "bacteria_p100.h"

#define WARP_SIZE 32

#define MAX_CONCURRENT_LOADS 4

int sm_count;
int thread_count;
int padding;

__global__ void _cuda_stochastic_precompute(long N, long M1, long* vector, long* second, long* one_l, long total, long complement, long total_l,
  double* dense_stochastic, double* vector_len);
__global__ void _cuda_compare_bacteria(long N, double* stochastic1, double* stochastic2, double* vector_len1, double* vector_len2, double* correlation);
void ProcessBacteria(Bacteria* b);
__global__ void _cuda_vector_len_sum(long N, double* dense_stochastic, double* vector_len);


int main(int argc, char *argv[])
{
  int device_count = 0;
  int current_device = 0;
  cudaGetDeviceCount(&device_count);
  std::cout << "Devices found: " << device_count << std::endl;

  while(current_device < device_count) {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, current_device);
    sm_count = prop.multiProcessorCount;
    thread_count = prop.maxThreadsPerBlock;
    std::cout << prop.name 
    << "\tSM Count: "
    << prop.multiProcessorCount
    << "\t Max threads: "
    << prop.maxThreadsPerBlock
    << std::endl;

    current_device++;
  }

	auto t1 = std::chrono::high_resolution_clock::now();

	Init();
	ReadInputFile("data/list.txt");
  number_bacteria = 2;

  Bacteria** bacteria = new Bacteria*[number_bacteria];

  for (int fi = 0; fi < number_bacteria; fi++) {
    // Read on CPU
    bacteria[fi] = new Bacteria(bacteria_name[fi]);
    std::cout << "Loaded " << fi + 1 << " of " << number_bacteria << std::endl;

    // Pre-compute on GPU
    ProcessBacteria(bacteria[fi]);
  }

  int count = 0;
  for (int i = 0; i < number_bacteria - 1; i++)
    for (int j = i + 1; j < number_bacteria; j++)
      count++;
  
  int pos = 0;
  // Compare on GPU
  double* d_correlation;
  cudaMalloc(&d_correlation, count * sizeof(double));
  cudaMemset(d_correlation, 0, count * sizeof(double));
  for (int i = 0; i < number_bacteria - 1; i++)
    for (int j = i + 1; j < number_bacteria; j++) {
      _cuda_compare_bacteria<<<sm_count, thread_count>>>(M + padding, bacteria[i]->dense_stochastic, 
        bacteria[j]->dense_stochastic, bacteria[i]->vector_len, bacteria[j]->vector_len, &d_correlation[pos++]);
      }
      
  cudaDeviceSynchronize();
  double* correlation = new double[count];
  cudaMemcpy(correlation, d_correlation, count * sizeof(double), cudaMemcpyDeviceToHost);
  cudaFree(d_correlation);

  pos = 0;
  for (int i = 0; i < number_bacteria - 1; i++)
    for (int j = i + 1; j < number_bacteria; j++) {
      printf("%02d %02d -> %.20lf\n", i, j, correlation[pos++]);
    }
  delete correlation;


  auto t2 = std::chrono::high_resolution_clock::now();
	std::cout	<< "Total time elapsed: "
				    << std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1).count()
				    << "s" << std::endl;

	return 0;
}

void ProcessBacteria(Bacteria* b) {
  cudaStream_t stream;
  cudaStreamCreate(&stream);

  int batch_size = sm_count * thread_count;
  padding = (batch_size - (M % batch_size)) % batch_size;

  // double* d_dense_stochastic;
  cudaMalloc(&b->dense_stochastic, (M + padding) * sizeof(double));
  cudaMemset(b->dense_stochastic, 0, (M + padding) * sizeof(double));
  cudaMalloc(&b->vector_len, sizeof(double));
  cudaMemset(b->vector_len, 0, sizeof(double));


  long *d_vector, *d_second, *d_one_l;
  cudaMalloc(&d_vector, M * sizeof(long));
  cudaMalloc(&d_second, M1 * sizeof(long));
  cudaMalloc(&d_one_l, AA_NUMBER * sizeof(long));

  cudaMemcpy(d_vector, b->vector, M * sizeof(long), cudaMemcpyHostToDevice);
  cudaMemcpy(d_second, b->second, M1 * sizeof(long), cudaMemcpyHostToDevice);
  cudaMemcpy(d_one_l, b->one_l, AA_NUMBER * sizeof(long), cudaMemcpyHostToDevice);

  _cuda_stochastic_precompute<<<sm_count, thread_count>>>(M, M1, d_vector, d_second, d_one_l, b->total, 
      b->complement, b->total_l, b->dense_stochastic, b->vector_len);
  std::cout << cudaPeekAtLastError() << std::endl;

  _cuda_vector_len_sum<<<sm_count, thread_count>>>(M + padding, b->dense_stochastic, b->vector_len);

  cudaDeviceSynchronize();
  cudaFree(d_vector);
  cudaFree(d_second);
  cudaFree(d_one_l);

  delete b->vector;
  delete b->second;
}

__global__ void _cuda_compare_bacteria(long N, double* stochastic1, double* stochastic2, double* vector_len1, double* vector_len2, double* correlation) {

  __shared__ double buffer[WARP_SIZE];
  int lane = threadIdx.x % WARP_SIZE;
  double temp;

  int i = blockIdx.x * blockDim.x + threadIdx.x;

	while(i < N)
	{
    temp = stochastic1[i] * stochastic2[i];

    for(int delta = WARP_SIZE / 2; delta > 0; delta /= 2)
         temp += __shfl_down_sync(-1, temp, delta);

    if(lane == 0)
        buffer[threadIdx.x / WARP_SIZE] = temp;

    __syncthreads();

    if(threadIdx.x < WARP_SIZE) 
    {
      temp = buffer[threadIdx.x];
      for(int delta = WARP_SIZE / 2; delta > 0; delta /= 2) 
        temp += __shfl_down_sync(-1, temp, delta);
    }

    if(threadIdx.x == 0)
      atomicAdd(correlation, temp);

    i += blockDim.x * gridDim.x;
    __syncthreads();
	}

  i = blockIdx.x * blockDim.x + threadIdx.x;
  __syncthreads();

  if (i == 0) 
    *correlation = *correlation / ( sqrt(*vector_len1) * sqrt(*vector_len2) );
}

__global__ void _cuda_stochastic_precompute(long N, long M1, long* vector, long* second, long* one_l, long total, long complement, long total_l,
  double* dense_stochastic, double* vector_len) {

  int i = blockIdx.x * blockDim.x + threadIdx.x;

  while(i < N) {
    double p1 = (double)second[i / AA_NUMBER] / (total + complement);
    double p2 = (double) one_l[i % AA_NUMBER] / total_l;
    double p3 = (double)second[i % M1] / (total + complement);
    double p4 = (double) one_l[i / M1] / total_l;
    double stochastic = ( p1 * p2 + p3 * p4 ) * total / 2;

    if (stochastic > EPSILON)
      dense_stochastic[i] = (vector[i] - stochastic) / stochastic;
    else
      dense_stochastic[i] = 0;

    i += blockDim.x * gridDim.x;
    __syncthreads();
  }
}

__global__ void _cuda_vector_len_sum(long N, double* dense_stochastic, double* vector_len) {

  __shared__ double buffer[WARP_SIZE];
  int lane = threadIdx.x % WARP_SIZE;
  double temp;

  int i = blockIdx.x * blockDim.x + threadIdx.x;

  while(i < N) {
    temp = dense_stochastic[i] * dense_stochastic[i];

    for(int delta = WARP_SIZE / 2; delta > 0; delta /= 2)
         temp += __shfl_down_sync(-1, temp, delta);

    if(lane == 0)
        buffer[threadIdx.x / WARP_SIZE] = temp;

    __syncthreads();

    if(threadIdx.x < WARP_SIZE) 
    {
      temp = buffer[threadIdx.x];
      for(int delta = WARP_SIZE / 2; delta > 0; delta /= 2) 
        temp += __shfl_down_sync(-1, temp, delta);
    }

    if(threadIdx.x == 0)
      atomicAdd(vector_len, temp);

    i += blockDim.x * gridDim.x;
    __syncthreads();
  }
}
