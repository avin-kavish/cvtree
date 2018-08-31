#include <stdio.h>
#include <string.h>
#include <math.h>

#include <iostream>
#include <chrono>
#include <string>
#include "bacteria.h"

void CompareAllBacteria();
double CompareBacteria(Bacteria *b1, Bacteria *b2);

int main(int argc, char *argv[])
{
	auto t1 = std::chrono::high_resolution_clock::now();

	Init();
	ReadInputFile("data/list.txt");
	CompareAllBacteria();

	auto t2 = std::chrono::high_resolution_clock::now();
	std::cout	<< "Total time elapsed: "
				<< std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1).count()
				<< "s" << std::endl;

	return 0;
}

void CompareAllBacteria()
{
	Bacteria **b = new Bacteria *[number_bacteria];
	for (int i = 0; i < number_bacteria; i++)
	{
		printf("load %d of %d\t", i + 1, number_bacteria);
		auto t1 = std::chrono::high_resolution_clock::now();
		b[i] = new Bacteria(bacteria_name[i]);
		auto t2 = std::chrono::high_resolution_clock::now();
		b[i]->GenerateStochastic();
		auto t3 = std::chrono::high_resolution_clock::now();
		b[i]->GenerateSparse();
		auto t4 = std::chrono::high_resolution_clock::now();
		
		auto milli1 = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();
		auto milli2 = std::chrono::duration_cast<std::chrono::milliseconds>(t3 - t2).count();
		auto milli3 = std::chrono::duration_cast<std::chrono::milliseconds>(t4 - t3).count();

		std::cout << "Load: " << milli1 <<  "ms \tstochastic compute: " << milli2 << "ms\tdense to sparse: " << milli3 << "ms" << std::endl;
	}

	auto t5 = std::chrono::high_resolution_clock::now();
	for (int i = 0; i < number_bacteria - 1; i++)
		for (int j = i + 1; j < number_bacteria; j++)
		{
			printf("%2d %2d -> ", i, j);
			double correlation = CompareBacteria(b[i], b[j]);
			printf("%.20lf\n", correlation);
		}
	auto t6 = std::chrono::high_resolution_clock::now();
}

double CompareBacteria(Bacteria *b1, Bacteria *b2)
{
	double correlation = 0;
	double vector_len1 = 0;
	double vector_len2 = 0;
	long p1 = 0;
	long p2 = 0;
	while (p1 < b1->count && p2 < b2->count)
	{
		long n1 = b1->ti[p1];
		long n2 = b2->ti[p2];
		if (n1 < n2)
		{
			double t1 = b1->tv[p1];
			vector_len1 += (t1 * t1);
			p1++;
		}
		else if (n2 < n1)
		{
			double t2 = b2->tv[p2];
			p2++;
			vector_len2 += (t2 * t2);
		}
		else
		{
			double t1 = b1->tv[p1++];
			double t2 = b2->tv[p2++];
			vector_len1 += (t1 * t1);
			vector_len2 += (t2 * t2);
			correlation += t1 * t2;
		}
	}
	while (p1 < b1->count)
	{
		long n1 = b1->ti[p1];
		double t1 = b1->tv[p1++];
		vector_len1 += (t1 * t1);
	}
	while (p2 < b2->count)
	{
		long n2 = b2->ti[p2];
		double t2 = b2->tv[p2++];
		vector_len2 += (t2 * t2);
	}

	return correlation / (sqrt(vector_len1) * sqrt(vector_len2));
}

