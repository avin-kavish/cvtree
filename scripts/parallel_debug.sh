export MAX_FILE_LOADS=1
g++ -g ./src/improved_parallel.cpp -o ./bin/improved_parallel.bin -pthread -fopenmp &&
gdb ./bin/improved_parallel.bin
