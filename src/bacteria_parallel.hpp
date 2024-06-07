#ifndef BACTERIA
#define BACTERIA
#include <cmath>
#include <cstring>

int number_bacteria;
char **bacteria_name;
long M, M1, M2;
short code[27] = {0,  2,  1,  2,  3,  4,  5,  6, 7,  -1, 8, 9,  10,
                  11, -1, 12, 13, 14, 15, 16, 1, 17, 18, 5, 19, 3};
#define encode(ch) code[ch - 'A']
#define LEN 6
#define AA_NUMBER 20
#define EPSILON 1e-010

void Init() {
  M2 = 1;
  for (int i = 0; i < LEN - 2; i++) // M2 = AA_NUMBER ^ (LEN-2);
    M2 *= AA_NUMBER;
  M1 = M2 * AA_NUMBER; // M1 = AA_NUMBER ^ (LEN-1);
  M = M1 * AA_NUMBER;  // M  = AA_NUMBER ^ (LEN);
}

class Bacteria {
private:
  long *vector;
  long *second;
  long one_l[AA_NUMBER];
  long indexs;
  long total;
  long total_l;
  long complement;
  double *t;

  void InitVectors() {
    vector = new long[M];
    second = new long[M1];
    memset(vector, 0, M * sizeof(long));
    memset(second, 0, M1 * sizeof(long));
    memset(one_l, 0, AA_NUMBER * sizeof(long));
    total = 0;
    total_l = 0;
    complement = 0;
  }

  void init_buffer(char *buffer) {
    complement++;
    indexs = 0;
    for (int i = 0; i < LEN - 1; i++) {
      short enc = encode(buffer[i]);
      one_l[enc]++;
      total_l++;
      indexs = indexs * AA_NUMBER + enc;
    }
    second[indexs]++;
  }

  void cont_buffer(char ch) {
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
  double *tv;
  long *ti;
  double vector_len_sqrt;
  std::string name;

  Bacteria(std::string filename) {
    FILE *bacteria_file = fopen(("data/" + filename).c_str(), "r");
    InitVectors();

    char ch, chk;
    while ((ch = fgetc(bacteria_file)) != EOF) {
      if (ch == '>') {
        while (chk = fgetc(bacteria_file) != '\n' && chk != '\r')
          ; // skip rest of line

        char buffer[LEN - 1];
        fread(buffer, sizeof(char), LEN - 1, bacteria_file);
        init_buffer(buffer);
      } else if (ch != '\n' && ch != '\r')
        cont_buffer(ch);
    }
    fclose(bacteria_file);
  }

  void GenerateStochastic() {
    double one_l_div_total[AA_NUMBER];

#pragma omp simd
    for (int i = 0; i < AA_NUMBER; i++)
      one_l_div_total[i] = (double)one_l[i] / total_l;

    long total_plus_complement = total + complement;
    double *second_div_total = new double[M1];
#pragma omp simd

    for (int i = 0; i < M1; i++)
      second_div_total[i] = (double)second[i] / total_plus_complement;

    count = 0;
    double total_div_2 = total * 0.5;
    int i_mod_aa_number = 0;
    int i_div_aa_number = 0;
    long i_mod_M1 = 0;
    long i_div_M1 = 0;
    double vector_len = 0;

    tv = static_cast<double *>(malloc(M * sizeof(double)));
    ti = static_cast<long *>(malloc(M * sizeof(long)));

    for (long i = 0; i < M; i++) {
      double p1 = second_div_total[i_div_aa_number];
      double p2 = one_l_div_total[i_mod_aa_number];
      double p3 = second_div_total[i_mod_M1];
      double p4 = one_l_div_total[i_div_M1];
      double stochastic = (p1 * p2 + p3 * p4) * total_div_2;

      if (i_mod_aa_number == AA_NUMBER - 1) {
        i_mod_aa_number = 0;
        i_div_aa_number++;
      } else
        i_mod_aa_number++;

      if (i_mod_M1 == M1 - 1) {
        i_mod_M1 = 0;
        i_div_M1++;
      } else
        i_mod_M1++;

      if (stochastic > EPSILON) {
        auto val = (vector[i] - stochastic) / stochastic;
        tv[count] = val;
        ti[count] = i;
        vector_len += val * val;
        count++;
      }
    }

    tv = static_cast<double *>(realloc(tv, count * sizeof(double)));
    ti = static_cast<long *>(realloc(ti, count * sizeof(long)));
    vector_len_sqrt = sqrt(vector_len);

    delete second_div_total;
    delete vector;
    delete second;
  }
};

bool ReadInputFile(std::string input_name) {
  if (FILE *input_file = fopen(input_name.c_str(), "r")) {
    fscanf(input_file, "%d", &number_bacteria);
    bacteria_name = new char *[number_bacteria];

    for (long i = 0; i < number_bacteria; i++) {
      bacteria_name[i] = new char[20];
      fscanf(input_file, "%s", bacteria_name[i]);
      strcat(bacteria_name[i], ".faa");
    }

    fclose(input_file);
    return true;
  } else
    return false;
}

#endif
