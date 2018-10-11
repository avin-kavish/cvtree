#include <atomic>
#include <future>
#include <list>
#include <math.h>
#include <queue>
#include <stdio.h>
#include <cstdlib>
#include <string.h>
#include <vector>

#include "bacteria_parallel.h"
#include <chrono>
#include <iostream>
#include <string>


std::atomic<int> processed;

int LoadBacteria(Bacteria **b, char *bacteria_name, int index);
void ProcessBacteria(Bacteria *b);
void CompareAllBacteria();
double CompareBacteria(Bacteria *b1, Bacteria *b2);

int main(int argc, char *argv[]) {
  auto t1 = std::chrono::high_resolution_clock::now();

  Init();
  ReadInputFile("data/list.txt");
  number_bacteria = 41;
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
  int max_file_loads = 3;
  if(const char* max_file_loads_str = std::getenv("MAX_FILE_LOADS"))
    max_file_loads = std::stoi(max_file_loads_str);

  printf("Reading %i bacteria in parallel\n", max_file_loads);

  std::list<std::future<int>> loads;
  std::vector<std::future<void>> processing;

  int fi = 0, current_loads = 0;

  Bacteria **b = new Bacteria *[number_bacteria];

  std::chrono::time_point<std::chrono::high_resolution_clock> t2;
  auto t1 = std::chrono::high_resolution_clock::now();
  while (processed < number_bacteria) {
    if (current_loads < max_file_loads && fi < number_bacteria) {
      printf("Launching %i of %i\n", fi + 1, number_bacteria);
      current_loads++;
      loads.push_back(std::async(std::launch::async, LoadBacteria, b + fi,
                                 bacteria_name[fi], fi));
      fi++;
    }

    for (auto it = loads.begin(); it != loads.end(); ++it) {
      if (it->wait_for(std::chrono::seconds(0)) == std::future_status::ready) {
        processing.push_back(
            std::async(std::launch::async, ProcessBacteria, b[it->get()]));
        current_loads--;
        it = loads.erase(it);

        if (fi == number_bacteria - 1)
          t2 = std::chrono::high_resolution_clock::now();
      }
    }
  }

  while (processing.size() > 0) {
    processing.back().get();
    processing.pop_back();
  }
  auto t3 = std::chrono::high_resolution_clock::now();

  std::vector<std::future<double>> comparisons;

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
  auto t4 = std::chrono::high_resolution_clock::now();

  auto milli1 =
      std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();
  auto milli2 =
      std::chrono::duration_cast<std::chrono::milliseconds>(t3 - t1).count();
  auto milli3 =
      std::chrono::duration_cast<std::chrono::milliseconds>(t4 - t3).count();

  std::cout << "Load: " << milli1 << "ms\n"
            << "Processing: " << milli1 << "ms\n"
            << "Comparison: " << milli2 << "ms" << std::endl;
}

void ProcessBacteria(Bacteria *b) {
  b->GenerateStochastic();
  b->GenerateSparse();
  processed++;
}

int LoadBacteria(Bacteria **b, char *bacteria_name, int index) {
  *b = new Bacteria(bacteria_name);
  std::cout << "Loaded: " << bacteria_name << std::endl;
  return index;
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

  return correlation / (b1->vector_len_sqrt * b2->vector_len_sqrt);
}
