# 4Dmeanfilter
<<Building & Execution Instruction>>
```
git clone https://github.com/keith4ever/4Dmeanfilter
cd 4Dmeanfilter
mkdir build
cd build
cmake ..
make clean; make -j 8
./meanfilter 
or
./meanfilter -cpu

./meanfilter -d1 256 -d2 128 -d3 128 -d4 64
or 
./meanfilter -d1 256 -d2 128 -d3 128 -d4 64 -cpu
```
Sample Test Result (with Intel i5 4200U & NVIDIA GTX 860M)
```
./meanfilter -d1 128 -d2 128 -d3 64 -d4 64
Input data dimensions: [128, 128, 64, 64]
======================================
CUDA Device Number: 0
  Device name: GeForce GTX 860M
  Compute Capability: 5.0
  Memory Capacity: 2000MB
======================================
Elapsed time: 345.155 msecs..

./meanfilter -d1 256 -d2 128 -d3 64 -d4 64 -cpu
Input data dimensions: [256, 128, 64, 64]
Elapsed time: 4890.56 msecs..
```