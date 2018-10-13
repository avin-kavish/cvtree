export MAX_FILE_LOADS=6
g++ -std=c++11 -O3 ./src/improved_parallel.cpp -o ./bin/improved_parallel.bin -pthread -fopenmp -fopt-info &&
./bin/improved_parallel.bin -t 16
