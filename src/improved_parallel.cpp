#include <future>
#include <math.h>
#include <stdio.h>
#include <string.h>
#include <vector>

#include "bacteria.h"
#include <chrono>
#include <iostream>
#include <string>

void ProcessBacteria(Bacteria **b, char *bacteria_name);
void CompareAllBacteria();
double CompareBacteria(Bacteria *b1, Bacteria *b2);

int main(int argc, char *argv[]) {
  auto t1 = std::chrono::high_resolution_clock::now();

  Init();
  ReadInputFile("data/list.txt");
  number_bacteria = 5;
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
  std::vector<std::future<void>> loads;
  auto t1 = std::chrono::high_resolution_clock::now();
  Bacteria **b = new Bacteria *[number_bacteria];
  for (int i = 0; i < number_bacteria; i++) {
    loads.push_back(std::async(std::launch::async, ProcessBacteria, b + i,
                               bacteria_name[i]));
  }

  for (int i = 0; i < number_bacteria; i++)
    loads[i].get();

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
  double vector_len1 = 0;
  double vector_len2 = 0;
  long p1 = 0;
  long p2 = 0;
  while (p1 < b1->count && p2 < b2->count) {
    long n1 = b1->ti[p1];
    long n2 = b2->ti[p2];
    if (n1 < n2) {
      double t1 = b1->tv[p1];
      vector_len1 += (t1 * t1);
      p1++;
    } else if (n2 < n1) {
      double t2 = b2->tv[p2];
      p2++;
      vector_len2 += (t2 * t2);
    } else {
      double t1 = b1->tv[p1++];
      double t2 = b2->tv[p2++];
      vector_len1 += (t1 * t1);
      vector_len2 += (t2 * t2);
      correlation += t1 * t2;
    }
  }
  while (p1 < b1->count) {
    long n1 = b1->ti[p1];
    double t1 = b1->tv[p1++];
    vector_len1 += (t1 * t1);
  }
  while (p2 < b2->count) {
    long n2 = b2->ti[p2];
    double t2 = b2->tv[p2++];
    vector_len2 += (t2 * t2);
  }

  return correlation / (sqrt(vector_len1) * sqrt(vector_len2));
}
