#include <iostream>
#include <chrono>
#include <string>
#include <thread>
#include <atomic>
#include "bacteria_cuda.h"

__global__
void _cuda_stochastic_precompute(long N, long M1, long* vector, long* second, long* one_l, long total, long complement, long total_l,
  double* dense_stochastic);

int main(int argc, char *argv[])
{
	auto t1 = std::chrono::high_resolution_clock::now();

	Init();
	ReadInputFile("data/list.txt");
    number_bacteria = 3;
    // Sequential load-single CUDA Accelerated kernel
    Bacteria** bacteria;
    cudaMallocManaged(&bacteria, number_bacteria * sizeof(Bacteria*));
    
    for (int fi = 0; fi < number_bacteria; fi++) {
      cudaMallocManaged(&bacteria[fi], sizeof(bacteria));
      bacteria[fi] = new(bacteria[fi]) Bacteria(bacteria_name[fi]);
      std::cout << "Loaded " << fi + 1 << " of " << number_bacteria << std::endl;

      // Copy memory
      cudaMallocManaged(&bacteria[fi]->dense_stochastic, M * sizeof(double));

      // Launch
      _cuda_stochastic_precompute<<<5 * 32, 256>>>(M, M1, bacteria[fi]->vector, bacteria[fi]->second, bacteria[fi]->one_l, 
        bacteria[fi]->total, bacteria[fi]->complement, bacteria[fi]->total_l, bacteria[fi]->dense_stochastic);
      

      cudaDeviceSynchronize();
      // Fetch mem
    }
    
    
    // int temp;
    // std::cin >> temp;



	auto t2 = std::chrono::high_resolution_clock::now();
	std::cout	<< "Total time elapsed: "
				    << std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1).count()
				    << "s" << std::endl;

	return 0;
}

__global__ void _cuda_stochastic_precompute(long N, long M1, long* vector, long* second, long* one_l, long total, long complement, long total_l,
  double* dense_stochastic) {

  int i = blockIdx.x * blockDim.x + threadIdx.x;

  while(i < N) {
    double p1 = (double)second[i / AA_NUMBER] / (total + complement);
    double p2 = (double) one_l[i % AA_NUMBER] / total_l;
    double p3 = (double)second[i % M1] / (total + complement);
    double p4 = (double) one_l[i / M1] / total_l;
    double stochastic = (p1*p2 + p3*p4) * total / 2;

    dense_stochastic[i] = (vector[i] - stochastic) / stochastic;

    i += blockDim.x * gridDim.x;
  }

}


    // Event loop
    
    // Load files on worker thread


    // Launch asynchronous kernels


    // Launch sparse generation on worker threads