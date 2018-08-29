#ifndef BACTERIA
#define BACTERIA

int number_bacteria;
char **bacteria_name;
long M, M1, M2;
short code[27] = {0, 2, 1, 2, 3, 4, 5, 6, 7, -1, 8, 9, 10, 11, -1, 12, 13, 14, 15, 16, 1, 17, 18, 5, 19, 3};
#define encode(ch) code[ch - 'A']
#define LEN 6
#define AA_NUMBER 20
#define EPSILON 1e-010

void Init()
{
	M2 = 1;
	for (int i = 0; i < LEN - 2; i++) // M2 = AA_NUMBER ^ (LEN-2);
		M2 *= AA_NUMBER;
	M1 = M2 * AA_NUMBER; // M1 = AA_NUMBER ^ (LEN-1);
	M = M1 * AA_NUMBER;  // M  = AA_NUMBER ^ (LEN);

	std::cout << "Init Constants\n"
		 << "M1: " << M1
		 << "\tM: " << M
		 << std::endl;
}

class Bacteria
{
  private:
	long indexs;
	
	void Inisparse_vectorectors()
	{
		cudaMallocManaged(&vector, M * sizeof(long));
		cudaMallocManaged(&second, M1 * sizeof(long));
		cudaMallocManaged(&one_l, AA_NUMBER * sizeof(long));
		memset(vector, 0, M * sizeof(long));
		memset(second, 0, M1 * sizeof(long));
		memset(one_l, 0, AA_NUMBER * sizeof(long));
		total = 0;
		total_l = 0;
		complement = 0;
	}

	void init_buffer(char *buffer)
	{
		complement++;
		indexs = 0;
		for (int i = 0; i < LEN - 1; i++)
		{
			short enc = encode(buffer[i]);
			one_l[enc]++;
			total_l++;
			indexs = indexs * AA_NUMBER + enc;
		}
		second[indexs]++;
	}

	void cont_buffer(char ch)
	{
		short enc = encode(ch);
		one_l[enc]++;
		total_l++;
		long index = indexs * AA_NUMBER + enc;
		vector[index]++;
		total++;
		indexs = (indexs % M2) * AA_NUMBER + enc;
		second[indexs]++;
	}

  public:
	long count;
	double *sparse_vector;
	long *sparse_index;
	double* dense_stochastic;
	long *vector;
	long *second;
	long* one_l;
	long total;
	long total_l;
	long complement;
	std::string name;

	Bacteria(std::string filename)
	{
		name = filename;
		FILE *bacteria_file = fopen(("data/" + filename).c_str(), "r");
		Inisparse_vectorectors();

		char ch, chk;
		while ((ch = fgetc(bacteria_file)) != EOF)
		{
			if (ch == '>')
			{
				while (chk = fgetc(bacteria_file) != '\n' && chk != '\r'); // skip rest of line

				char buffer[LEN - 1];
				fread(buffer, sizeof(char), LEN - 1, bacteria_file);
				init_buffer(buffer);
			}
			else if (ch != '\n' && ch != '\r')
				cont_buffer(ch);
		}
		fclose(bacteria_file);
    }

    void DenseToSparse() {
		double* tempv = (double*) malloc(M * sizeof(double));
		long* tempi = (long*) malloc(M * sizeof(long));

		int pos = 0;
		for (long i = 0; i < M; i++)
		{
			if (dense_stochastic[i] != 0)
			{
				tempv[pos] = dense_stochastic[i];
				tempi[pos] = i;
				pos++;
			}
		}
		cudaMallocManaged(&sparse_vector, pos * sizeof(double));
		cudaMallocManaged(&sparse_index, pos * sizeof(long));
		cudaMemcpy(sparse_vector, tempv, pos * sizeof(double), cudaMemcpyHostToHost);
		cudaMemcpy(sparse_index, tempi, pos * sizeof(long), cudaMemcpyHostToHost);
		free(tempv);
		free(tempi);
		cudaFree(dense_stochastic);
		}
};

bool ReadInputFile(std::string input_name)
{
	if (FILE *input_file = fopen(input_name.c_str(), "r"))
	{
		fscanf(input_file, "%d", &number_bacteria);
		bacteria_name = new char *[number_bacteria];

		for (long i = 0; i < number_bacteria; i++)
		{
			bacteria_name[i] = new char[20];
			fscanf(input_file, "%s", bacteria_name[i]);
			strcat(bacteria_name[i], ".faa");
		}
		fclose(input_file);
		return true;
	}
	else
		return false;
}

#endif