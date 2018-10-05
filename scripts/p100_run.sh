nvcc -gencode arch=compute_60,code=sm_60 ./src/cvtree_p100.cu -o ./bin/cvtree_p100.bin && 
./bin/cvtree_p100.bin 
