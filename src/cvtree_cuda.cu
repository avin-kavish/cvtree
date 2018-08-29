#include <iostream>
#include <chrono>
#include <string>
#include <thread>
#include <atomic>
#include "bacteria_cuda.h"

int MAX_CONCURRENT_LOADS = 5;
std::atomic<int> current_loads;

__global__
void _cuda_stochastic_precompute(long N, long M1, long* vector, long* second, long* one_l, long total, long complement, long total_l,
  double* dense_stochastic);

void ProcessBacteria(Bacteria* b);

int main(int argc, char *argv[])
{
	auto t1 = std::chrono::high_resolution_clock::now();

	Init();
	ReadInputFile("data/list.txt");
  number_bacteria = 10;

  std::thread threads[number_bacteria];
  Bacteria** bacteria;
  cudaMallocManaged(&bacteria, number_bacteria * sizeof(Bacteria*));
    
    for (int fi = 0; fi < number_bacteria; fi++) {
      if(current_loads < MAX_CONCURRENT_LOADS) {
        cudaMallocManaged(&bacteria[fi], sizeof(bacteria));
        bacteria[fi] = new(bacteria[fi]) Bacteria(bacteria_name[fi]);
        std::cout << "Loaded " << fi + 1 << " of " << number_bacteria << std::endl;

        threads[fi] = std::thread(ProcessBacteria, bacteria[fi]);
        current_loads++;
      }
    }

    for (int fi = 0; fi < number_bacteria; fi++)
      threads[fi].join();
  
    // int temp;
    // std::cin >> temp;

	auto t2 = std::chrono::high_resolution_clock::now();
	std::cout	<< "Total time elapsed: "
				    << std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1).count()
				    << "s" << std::endl;

	return 0;
}

void ProcessBacteria(Bacteria* b){
  cudaStream_t stream;
  cudaStreamCreate(&stream);
  // Copy memory
  cudaMallocManaged(&b->dense_stochastic, M * sizeof(double));

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
  _cuda_stochastic_precompute<<<5, 1024, 0, stream>>>(M, M1, b->vector, b->second, b->one_l, 
  b->total, b->complement, b->total_l, b->dense_stochastic);
  
  // Fetch mem
  cudaMemPrefetchAsync(b->dense_stochastic, M * sizeof(double), cudaCpuDeviceId, stream);
  cudaFree(b->vector);
  cudaFree(b->second);

  cudaStreamSynchronize(stream);

  b->DenseToSparse();
  current_loads--;
}

__global__ void _cuda_stochastic_precompute(long N, long M1, long* vector, long* second, long* one_l, long total, long complement, long total_l,
  double* dense_stochastic) {

  int i = blockIdx.x * blockDim.x + threadIdx.x;

  while(i < N) {
    double p1 = (double)second[i / AA_NUMBER] / (total + complement);
    double p2 = (double) one_l[i % AA_NUMBER] / total_l;
    double p3 = (double)second[i % M1] / (total + complement);
    double p4 = (double) one_l[i / M1] / total_l;
    double stochastic = ( p1 * p2 + p3 * p4 ) 
    * total / 2;
    
    if (stochastic > EPSILON)
      dense_stochastic[i] = (vector[i] - stochastic) / stochastic;


    i += blockDim.x * gridDim.x;
  }
}

// for (int fi = 0; fi < number_bacteria; fi++) {
//   cudaMallocManaged(&bacteria[fi], sizeof(bacteria));
//   bacteria[fi] = new(bacteria[fi]) Bacteria(bacteria_name[fi]);
//   std::cout << "Loaded " << fi + 1 << " of " << number_bacteria << std::endl;

//   Bacteria* b = bacteria[fi];
//   cudaStream_t stream;
//   cudaStreamCreate(&stream);
//   // Copy memory
//   cudaMallocManaged(&bacteria[fi]->dense_stochastic, M * sizeof(double));

//   cudaStreamAttachMemAsync(stream, b->vector);
//   cudaStreamAttachMemAsync(stream, b->second);
//   cudaStreamAttachMemAsync(stream, b->dense_stochastic);
//   cudaStreamAttachMemAsync(stream, b->one_l);

//   cudaMemPrefetchAsync(b->vector, M * sizeof(long), 0, stream);
//   cudaMemPrefetchAsync(b->second, M1 * sizeof(long), 0, stream);
//   cudaMemPrefetchAsync(b->dense_stochastic, M * sizeof(double), 0, stream);
//   cudaMemPrefetchAsync(b->one_l, AA_NUMBER * sizeof(long), 0, stream);

//   // Launch
//   _cuda_stochastic_precompute<<<5, 1024>>>(M, M1, b->vector, b->second, b->one_l, 
//     b->total, b->complement, b->total_l, b->dense_stochastic);
  

//   cudaDeviceSynchronize();
//   // Fetch mem
// }


    // Event loop
    
    // Load files on worker thread


    // Launch asynchronous kernels


    // Launch sparse generation on worker threads