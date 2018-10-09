#include <future>
#include <math.h>
#include <queue>
#include <stdio.h>
#include <string.h>
#include <vector>

#include "bacteria_parallel.h"
#include <chrono>
#include <iostream>
#include <string>

#define MAX_LAUNCHES 3

void ProcessBacteria(Bacteria **b, char *bacteria_name);
void CompareAllBacteria();
double CompareBacteria(Bacteria *b1, Bacteria *b2);

int main(int argc, char *argv[]) {
  auto t1 = std::chrono::high_resolution_clock::now();

  Init();
  ReadInputFile("data/list.txt");
  number_bacteria = 10;
  CompareAllBacteria();

  auto t2 = std::chrono::high_resolution_clock::now();
  std::cout << "Total time elapsed: "
            << std::chrono::duration_cast<std::chrono::duration<double>>(t2 -
                                                                         t1)
                   .count()
            << "s" << std::endl;

  return 0;
}

void CompareAllBacteria() {
  std::queue<std::future<void>> loads;
  
  auto t1 = std::chrono::high_resolution_clock::now();
  Bacteria **b = new Bacteria *[number_bacteria];
  for (int i = 0; i < number_bacteria; i++) {
    printf("Launching %i of %i\n", i, number_bacteria);
    loads.push(std::async(std::launch::async, ProcessBacteria, b + i,
                          bacteria_name[i]));

    while (loads.size() > MAX_LAUNCHES) {
      loads.front().get();
      loads.pop();
    }
  }

  while (loads.size() > 0) {
    loads.front().get();
    loads.pop();
  }

  auto t2 = std::chrono::high_resolution_clock::now();

  std::vector<std::future<double>> comparisons;

  auto t5 = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < number_bacteria - 1; i++)
    for (int j = i + 1; j < number_bacteria; j++)
      comparisons.push_back(
          std::async(std::launch::async, CompareBacteria, b[i], b[j]));

  auto it = comparisons.begin();
  for (int i = 0; i < number_bacteria - 1; i++)
    for (int j = i + 1; j < number_bacteria; j++) {
      printf("%2d %2d -> ", i, j);
      double correlation = it->get();
      printf("%.20lf\n", correlation);
      it++;
    }

  auto t6 = std::chrono::high_resolution_clock::now();
  auto milli1 =
      std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();
  auto milli2 =
      std::chrono::duration_cast<std::chrono::milliseconds>(t6 - t5).count();
  std::cout << "Total Load: " << milli1 << "ms" << std::endl;
  std::cout << "Total Comparison: " << milli2 << "ms" << std::endl;
}

void ProcessBacteria(Bacteria **b, char *bacteria_name) {
  *b = new Bacteria(bacteria_name);
  (*b)->GenerateStochastic();
  (*b)->GenerateSparse();
  std::cout << "Loaded: " << bacteria_name << std::endl;
}

double CompareBacteria(Bacteria *b1, Bacteria *b2) {
  double correlation = 0;
  long p1 = 0;
  long p2 = 0;

  while (p1 < b1->count && p2 < b2->count) {
    long n1 = b1->ti[p1];
    long n2 = b2->ti[p2];
    if (n1 < n2)
      p1++;
    else if (n2 < n1)
      p2++;
    else {
      double t1 = b1->tv[p1++];
      double t2 = b2->tv[p2++];
      correlation += t1 * t2;
    }
  }

  return correlation / ( b1->vector_len_sqrt * b2->vector_len_sqrt );
}
