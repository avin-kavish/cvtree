#include <iostream>
#include <chrono>
#include <string>
#include <thread>
#include <vector>
#include <atomic>
#include <future>
#include <stdexcept>
#include "bacteria_p100.h"

#define MAX_CONCURRENT_LOADS 4

int sm_count;
int thread_count;

__global__ void _cuda_stochastic_precompute(long N, long M1, long* vector, long* second, long* one_l, long total, long complement, long total_l,
  double* dense_stochastic);
__global__ void _cuda_compare_bacteria(long N, double* stochastic1, double* stochastic2, double* correlation);
void ProcessBacteria(Bacteria* b);


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
  number_bacteria = 10;

  Bacteria** bacteria = new Bacteria*[number_bacteria];

  for (int fi = 0; fi < number_bacteria; fi++) {
    // Read on CPU
    bacteria[fi] = new Bacteria(bacteria_name[fi]);
    std::cout << "Loaded " << fi + 1 << " of " << number_bacteria << std::endl;

    // Pre-compute on GPU
    ProcessBacteria(bacteria[fi]);
  }

  // Compare on GPU
  double* d_correlation;
  cudaMalloc(&d_correlation, number_bacteria * number_bacteria * sizeof(double));
  cudaMemset(d_correlation, 0, number_bacteria * number_bacteria * sizeof(double));
  for (int i = 0; i < number_bacteria - 1; i++)
    for (int j = i + 1; j < number_bacteria; j++) {
      _cuda_compare_bacteria<<<sm_count, thread_count>>>(M, bacteria[i]->dense_stochastic, 
        bacteria[j]->dense_stochastic, &d_correlation[i * number_bacteria + j]);
    }



}



void ProcessBacteria(Bacteria* b) {
  cudaStream_t stream;
  cudaStreamCreate(&stream);

  // double* d_dense_stochastic;
  cudaMalloc(&b->dense_stochastic, M * sizeof(double));
  cudaMemset(b->dense_stochastic, 0, M * sizeof(double));

  long *d_vector, *d_second, *d_one_l;
  cudaMalloc(&d_vector, M * sizeof(long));
  cudaMalloc(&d_second, M1 * sizeof(long));
  cudaMalloc(&d_one_l, AA_NUMBER * sizeof(long));

  cudaMemcpy(d_vector, b->vector, M * sizeof(long), cudaMemcpyHostToDevice);
  cudaMemcpy(d_second, b->second, M1 * sizeof(long), cudaMemcpyHostToDevice);
  cudaMemcpy(d_one_l, b->one_l, AA_NUMBER * sizeof(long), cudaMemcpyHostToDevice);

  _cuda_stochastic_precompute<<<10, 1024>>>(M, M1, d_vector, d_second, d_one_l, b->total, b->complement, b->total_l, b->dense_stochastic);
  std::cout << cudaPeekAtLastError() << std::endl;
  cudaDeviceSynchronize();
  cudaFree(d_vector);
  cudaFree(d_second);
  cudaFree(d_one_l);

  delete b->vector;
  delete b->second;
}

__global__ void _cuda_compare_bacteria(long N, double* stochastic1, double* stochastic2, double* correlation) {

  int i = blockIdx.x * blockDim.x + threadIdx.x;

	while(i < N)
	{
    atomicAdd(correlation, stochastic1[i] * stochastic2[i]);

    i += blockDim.x * gridDim.x;
	}
}


__global__ void _cuda_stochastic_precompute(long N, long M1, long* vector, long* second, long* one_l, long total, long complement, long total_l,
  double* dense_stochastic) {

  int i = blockIdx.x * blockDim.x + threadIdx.x;

  while(i < N) {
    double p1 = (double)second[i / AA_NUMBER] / (total + complement);
    double p2 = (double) one_l[i % AA_NUMBER] / total_l;
    double p3 = (double)second[i % M1] / (total + complement);
    double p4 = (double) one_l[i / M1] / total_l;
    double stochastic = ( p1 * p2 + p3 * p4 ) * total / 2;
    
    if (stochastic > EPSILON)
      dense_stochastic[i] = (vector[i] - stochastic) / stochastic;

    i += blockDim.x * gridDim.x;
  }
}