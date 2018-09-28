#include <iostream>
#include <chrono>
#include <string>
#include <thread>
#include <vector>
#include <atomic>
#include <future>
#include "bacteria_cuda.h"
#include "atomic_add.cuh"

#define MAX_CONCURRENT_LOADS 4

std::atomic<int> current_loads;
double *correlation;
int block_size;
int block_count;

struct Compare {
  Bacteria* b1;
  Bacteria* b2;
  double* result;
};

__global__
void _cuda_stochastic_precompute(long N, long M1, long* vector, long* second, long* one_l, long total, long complement, long total_l,
  double* dense_stochastic);
__global__
void _cuda_compare_bacteria(long N, Bacteria *b1, Bacteria* b2, double *correlation);
void gpu_compare(Bacteria** b);
void process_bacteria(Bacteria* b);
void print_correlation();


int main(int argc, char *argv[])
{
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, 0);

  std::cout << prop.name 
            << "\tSM Count: "
            << prop.multiProcessorCount
            << "\t Max threads: "
            << prop.maxThreadsPerBlock
            << std::endl;

	auto t1 = std::chrono::high_resolution_clock::now();

	Init();
	ReadInputFile("data/list.txt");
  number_bacteria = 41;
  current_loads = 0;
  std::future<void> threads[number_bacteria];
  Bacteria** bacteria;
  cudaMallocManaged(&bacteria, number_bacteria * sizeof(Bacteria*));
    
  for (int fi = 0; fi < number_bacteria;) {
    if(current_loads < MAX_CONCURRENT_LOADS) {
      cudaMallocManaged(&bacteria[fi], sizeof(bacteria));
      bacteria[fi] = new(bacteria[fi]) Bacteria(bacteria_name[fi]);
      std::cout << "Loaded " << fi + 1 << " of " << number_bacteria << std::endl;

      current_loads++;
      threads[fi] = std::async(std::launch::async, process_bacteria, bacteria[fi]);
      fi++;
    }
  }
  for (int fi = 0; fi < number_bacteria; fi++){
    threads[fi].get();    
  }

  int count = 0;
  for (int i = 0; i < number_bacteria - 1; i++)
    for (int j = i + 1; j < number_bacteria; j++)
      count++;

  correlation = new double[count];

  gpu_compare(bacteria);
  print_correlation();

	auto t2 = std::chrono::high_resolution_clock::now();
	std::cout	<< "Total time elapsed: "
				    << std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1).count()
				    << "s" << std::endl;

	return 0;
}

__global__
void _cuda_compare_bacteria(long N, double *b1, double* b2, double *correlation)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  
  while(i < N) {
    atomicdAdd(correlation, b1[i] * b2[i]);
    i += blockDim.x * gridDim.x;
  }
}

__global__
void _cuda_stochastic_precompute(long N, long M1, long* vector, long* second, long* one_l, 
long total, long complement, long total_l, double* dense_stochastic) {
    
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

void gpu_compare(Bacteria** b) {
  int grid, block;
  //cudaOccupancyMaxPotentialBlockSize(&grid, &block, _cuda_compare_bacteria);

  int pos = 0;
  for (int i = 0; i < number_bacteria - 1; i++)
  for (int j = i + 1; j < number_bacteria; j++) {
    _cuda_compare_bacteria<<<20, 1024>>>(M, b[i]->dense_stochastic, b[j]->dense_stochastic, &correlation[pos]);
    cudaDeviceSynchronize();
    //correlation[pos] = correlation[pos] / (sqrt(b[i]->vector_len) * sqrt(b[j]->vector_len));
    pos++;
  }
}
  
void print_correlation() {
  int pos = 0;
  for (int i = 0; i < number_bacteria - 1; i++)
  for (int j = i + 1; j < number_bacteria; j++) {
    printf("%02d %02d -> %.20lf\n", i, j, correlation[pos++]);
  }
}

void process_bacteria(Bacteria* b) {
  cudaStream_t stream;
  cudaStreamCreate(&stream);
  cudaMallocManaged(&b->dense_stochastic, M * sizeof(double));
  
  // Copy memory
  cudaStreamAttachMemAsync(stream, b->vector);
  cudaStreamAttachMemAsync(stream, b->second);
  cudaStreamAttachMemAsync(stream, b->dense_stochastic);
  cudaStreamAttachMemAsync(stream, b->one_l);

  cudaMemPrefetchAsync(b->vector, M * sizeof(long), 0, stream);
  cudaMemPrefetchAsync(b->second, M1 * sizeof(long), 0, stream);
  cudaMemPrefetchAsync(b->dense_stochastic, M * sizeof(double), 0, stream);
  cudaMemsetAsync(b->dense_stochastic, 0, M * sizeof(double), stream);
  cudaMemPrefetchAsync(b->one_l, AA_NUMBER * sizeof(long), 0, stream);

  // Launch
  _cuda_stochastic_precompute<<<10, 768, 0, stream>>>(M, M1, b->vector, b->second, b->one_l, 
  b->total, b->complement, b->total_l, b->dense_stochastic);

  cudaFree(b->vector);
  cudaFree(b->second);
  cudaFree(b->one_l);
  
  cudaStreamSynchronize(stream);
  current_loads--;
}