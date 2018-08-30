#include <iostream>
#include <chrono>
#include <string>
#include <thread>
#include <deque>
#include <atomic>
#include <thrust/scan.h>


#include "bacteria_cuda.h"

#define MAX_CONCURRENT_LOADS 1

std::atomic<int> current_loads;
bool* compared;
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
__global__ void _cuda_sparse_block_count(long N, double* dense, int* blockCount);
__global__ void _cuda_parallel_sparse(long N, double* dense, double* sparse_vector, double* sparse_index);
void ProcessBacteria(Bacteria* b);
void CompareBacteria(Compare c);
void MultiThreadedCPUCompare(Bacteria** b);
void PrintCorrelation();



int main(int argc, char *argv[])
{
	auto t1 = std::chrono::high_resolution_clock::now();

	Init();
	ReadInputFile("data/list.txt");
  number_bacteria = 10;

  cudaOccupancyMaxPotentialBlockSize(&block_count, &block_size, _cuda_stochastic_precompute);
  std::cout  << "Launching " 
        << block_count
        << " blocks with "
        << block_size
        << " threads per block" 
        << std::endl;

  // nCr value, no. of unique comparisons
  int count = 0;
  for (int i = 0; i < number_bacteria - 1; i++)
    for (int j = i + 1; j < number_bacteria; j++)
      count++;

  current_loads = 0;
  std::thread threads[number_bacteria];
  Bacteria** bacteria;
  cudaMallocManaged(&bacteria, number_bacteria * sizeof(Bacteria*));
    
  correlation = new double[count];
   
  for (int fi = 0; fi < number_bacteria;) {
    if(current_loads < MAX_CONCURRENT_LOADS) {
      cudaMallocManaged(&bacteria[fi], sizeof(bacteria));
      bacteria[fi] = new(bacteria[fi]) Bacteria(bacteria_name[fi]);
      std::cout << "File read " << fi + 1 << " of " << number_bacteria << std::endl;
      
      current_loads++;
      threads[fi] = std::thread(ProcessBacteria, bacteria[fi]);
      fi++;
    }
  }

  for (int fi = 0; fi < number_bacteria; fi++){
    threads[fi].join(); threads[fi].~thread();    // We are not re-using threads for now
  }
  
  current_loads = 0;
  //MultiThreadedCPUCompare(bacteria); 
  //PrintCorrelation();
  printf("\n");
  

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
  _cuda_stochastic_precompute<<<block_count, block_size, 0, stream>>>(M, M1, b->vector, b->second, b->one_l, 
  b->total, b->complement, b->total_l, b->dense_stochastic);
  
  int num_blocks = M / 1024;
  int* blockCount;
  cudaMallocManaged(&blockCount, num_blocks * sizeof(int));
  cudaStreamAttachMemAsync(stream, blockCount);
  
  _cuda_sparse_block_count<<<num_blocks, 1024, 0, stream>>>(M, b->dense_stochastic, blockCount);
  b->count = thrust::reduce(thrust::cuda::par.on(stream), blockCount, blockCount + num_blocks);
  
  cudaMallocManaged(&b->sparse_vector, b->count * sizeof(double));
  cudaMallocManaged(&b->sparse_index, b->count * sizeof(long));
  int* count;
  cudaMalloc(&count, sizeof(int));

  _cuda_parallel_sparse<<<block_count, block_size, 0, stream>>>(M, b->dense_stochastic, b->sparse_vector, b->sparse_index, count);
  
  // thrust::exclusive_scan(thrust::cuda::par.on(stream), blockCount, blockCount + num_blocks, blockCount);
  std::cout << b->count << std::endl;
  // Fetch mem
  cudaMemPrefetchAsync(b->dense_stochastic, M * sizeof(double), cudaCpuDeviceId, stream);
  cudaFree(b->count)
  cudaFree(b->vector);
  cudaFree(b->second);
  cudaFree(b->dense_stochastic);
  
  cudaStreamSynchronize(stream);
  // This call will block the thread --> 
  // Instead we query stream status and yield to OS thread scheduler
  //while(cudaStreamQuery(stream) != 0) std::this_thread::yield();
  // auto t1 = std::chrono::high_resolution_clock::now();
  // b->DenseToSparse();
  // auto t2 = std::chrono::high_resolution_clock::now();
  b->processed = true;
  current_loads--;


	// std::cout	<< "Dense to sparse time: "
  // << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count()
  // << "ms" << std::endl;
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

__global__ void _cuda_parallel_sparse(long N, double* dense, double* sparse_vector, double* sparse_index, int* count) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  while(i < N) {
    if (dense[idx] != 0) {
      int i = atomicAdd(count, 1);
      sparse_vector[i] = dense[idx];
      sparse_index[i] = i;
    }
    
    idx += blockDim.x * gridDim.x;
  }

}

__global__ void _cuda_sparse_block_count(long N, double* dense, int* blockCount){
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i < N) {
    bool test = dense[i] != 0;
    int count = __syncthreads_count(test);

    if (threadIdx.x == 0)
      blockCount[blockIdx.x] = count;
  }
}

__global__ void _cuda_stochastic_precompute(long N, long M1, long* vector, long* second, 
long* one_l, long total, long complement, long total_l, double* dense_stochastic) {

  int i = blockIdx.x * blockDim.x + threadIdx.x;
  long total_p_complement =  total + complement;
  while(i < N) {
    double p1 = (double)second[i / AA_NUMBER] / total_p_complement;
    double p2 = (double) one_l[i % AA_NUMBER] / total_l;
    double p3 = (double)second[i % M1] / total_p_complement;
    double p4 = (double) one_l[i / M1] / total_l;
    double stochastic = ( p1 * p2 + p3 * p4 ) * total / 2;
    
    if (stochastic > EPSILON)
      dense_stochastic[i] = (vector[i] - stochastic) / stochastic;

    i += blockDim.x * gridDim.x;
  }
}


void MultiThreadedCPUCompare(Bacteria** bacteria) {
  int pos = 0;
  std::deque<std::thread> compare_threads;
  for (int i = 0; i < number_bacteria - 1; i++)
  for (int j = i + 1; j < number_bacteria; j++) {
    Compare c;
    c.b1 = bacteria[i];
    c.b2 = bacteria[j];
    c.result = &correlation[pos++];
    compare_threads.push_back(std::thread(CompareBacteria, c));
    while(compare_threads.size() > 5)
    { 
      std::thread& thread = compare_threads.front();
      thread.join(); thread.~thread();
      compare_threads.pop_front();
    }
  }

  while(compare_threads.size() > 0) {
    std::thread& thread = compare_threads.front();
    thread.join(); thread.~thread();
    compare_threads.pop_front();;
  }
}

void PrintCorrelation() {
  int pos = 0;
  for (int i = 0; i < number_bacteria - 1; i++)
    for (int j = i + 1; j < number_bacteria; j++) {
      printf("%02d %02d -> %.20lf\n", i, j, correlation[pos++]);
    }
}