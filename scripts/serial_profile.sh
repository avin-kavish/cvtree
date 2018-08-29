g++ -pg ./src/improved_serial.cpp -o ./bin/improved_serial_profile.bin &&
./bin/improved_serial_profile.bin &&
gprof ./bin/improved_serial_profile.bin > ./profiling/serial_profile.txt
