# dgemm on gpu
Implement and tune gpu dgemm on gpu.

## Install cuda driver/toolkit
https://developer.nvidia.com/cuda-downloads
Super important!!! Click documentation and make sure that you have the same compiler version as docs.
For example on mac, `clang --version` and make sure the LLVM number matches, if not go to archived version and get one that matches.

Then add environment variables like this if mac/ubuntu:
```
export PATH=/Developer/NVIDIA/CUDA-9.1/bin${PATH:+:${PATH}}
export DYLD_LIBRARY_PATH=/Developer/NVIDIA/CUDA-9.1/lib\
                         ${DYLD_LIBRARY_PATH:+:${DYLD_LIBRARY_PATH}}
```

## Compile program
Navigate to this directory
```
nvcc -std=c++11 matrix.cu -o matrix
```
## Run Program
```
./matrix
```