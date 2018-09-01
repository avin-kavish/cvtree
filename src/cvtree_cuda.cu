#include <iostream>
#include <chrono>
#include <string>
#include <thread>
#include <vector>
#include <atomic>
#include <future>
#include "bacteria_cuda.h"

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
void ProcessBacteria(Bacteria* b);
void CompareBacteria(Compare c);
void MultiThreadedCPUCompare(Bacteria** b);
void PrintCorrelation();


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
      threads[fi] = std::async(std::launch::async, ProcessBacteria, bacteria[fi]);
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

  MultiThreadedCPUCompare(bacteria);
  PrintCorrelation();

	auto t2 = std::chrono::high_resolution_clock::now();
	std::cout	<< "Total time elapsed: "
				    << std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1).count()
				    << "s" << std::endl;

	return 0;
}

void ProcessBacteria(Bacteria* b){
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
  
  // Fetch mem
  cudaMemPrefetchAsync(b->dense_stochastic, M * sizeof(double), cudaCpuDeviceId, stream);
  cudaFree(b->vector);
  cudaFree(b->second);
  
  // This call will block the thread --> cudaStreamSynchronize(stream);
  // Instead we query stream status and yield to OS thread scheduler
  while(cudaStreamQuery(stream) != 0) std::this_thread::yield();
  
  auto t1 = std::chrono::high_resolution_clock::now();
  b->DenseToSparse();
  auto t2 = std::chrono::high_resolution_clock::now();
	// std::cout	<< "steam-compact: "
	// 			    << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count()
	// 			    << "ms" << std::endl;
  current_loads--;
}

void CompareBacteria(Compare c)
{
  Bacteria *b1 = c.b1;
  Bacteria *b2 = c.b2;
	double correlation = 0;
	long p1 = 0;
	long p2 = 0;
	while (p1 < b1->count && p2 < b2->count)
	{
		long n1 = b1->sparse_index[p1];
		long n2 = b2->sparse_index[p2];
		if (n1 < n2)
			p1++;
		else if (n2 < n1)
			p2++;
		else
			correlation += b1->sparse_vector[p1++] * b2->sparse_vector[p2++];
	}

	*c.result = correlation / (b1->vector_len_sqrt * b2->vector_len_sqrt);
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

void MultiThreadedCPUCompare(Bacteria** bacteria) {
  std::cout << "Please wait. Performing batch comparison..." << std::endl;
  int pos = 0;
  std::vector<std::future<void>> compare_threads;
  for (int i = 0; i < number_bacteria - 1; i++)
  for (int j = i + 1; j < number_bacteria; j++) {
    Compare c;
    c.b1 = bacteria[i];
    c.b2 = bacteria[j];
    c.result = &correlation[pos++];
    compare_threads.push_back(std::async(std::launch::async, CompareBacteria, c));
  }

  while(compare_threads.size() > 0) {
    compare_threads.back().get();
    compare_threads.pop_back();;
  }
}

void PrintCorrelation() {
  int pos = 0;
  for (int i = 0; i < number_bacteria - 1; i++)
    for (int j = i + 1; j < number_bacteria; j++) {
      printf("%02d %02d -> %.20lf\n", i, j, correlation[pos++]);
    }
}