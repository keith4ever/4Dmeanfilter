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
