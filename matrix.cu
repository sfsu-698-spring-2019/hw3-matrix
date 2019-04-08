#include <iostream>
#include <cstdlib>
#include <math.h>
#include <chrono>

// matrix multiply on gpu
__global__
void dgem_gpu(int n, float *A, float *B, float *C)
{
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  int j = blockIdx.y*blockDim.y + threadIdx.y;

  // demo filler
  C[i+j*n] = B[i+j*n];
}

void square_dgemm_naive (int n, float* A, float* B, float* C)
{
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
        {
            float cij = C[i+j*n];
            for( int k = 0; k < n; k++ )
                cij += A[i+k*n] * B[k+j*n];
            C[i+j*n] = cij;
        }
}

int check(int n, float *A, float *B) {
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j) {
            double diff = std::abs(A[i + j * n] - B[i + j * n]);
            if (diff > 0.0003) {
                printf("diff is %f\n", diff);
                return 0;
            }
        }
    return 1;
}

int main(void)
{
  int N = 1000;
  int size = N*N; // square matrix
  float *A, *B, *C, *verify;

  // Works on cpu and gpu
  cudaMallocManaged(&A, size*sizeof(float));
  cudaMallocManaged(&B, size*sizeof(float));
  cudaMallocManaged(&C, size*sizeof(float));
  cudaMallocManaged(&verify, size*sizeof(float));


  // initialize x and y arrays on the host
  for (int i = 0; i < size; i++) {
    A[i] = 1.0f;
    B[i] = 2.0f;
    C[i] = 0.0f;
    verify[i] = 0.0f;
  }

  // this is to generate answer
  auto serialStart = std::chrono::system_clock::now();
  square_dgemm_naive(N, A, B, verify);
  auto serialEnd = std::chrono::system_clock::now();
  std::chrono::duration<double> serialElapsed = serialEnd - serialStart;
  std::cout << serialElapsed.count() << "s\n";

  // Run kernel on the GPU
  // use this one for actual work
  auto gpuStart = std::chrono::system_clock::now();
  // dgem_gpu<<<N, N>>>(N, A, B, C);
  // comment this one out, just for testing
  dgem_gpu<<<N, N>>>(N, A, C, verify);
  auto gpuEnd = std::chrono::system_clock::now();
  std::chrono::duration<double> gpuElapsed = gpuEnd - gpuStart;
  std::cout << gpuElapsed.count() << "s\n";

  // wait for threads to finish
  cudaDeviceSynchronize();

  int correct = check(N, C, verify);

  // Free memory
  cudaFree(A);
  cudaFree(B);
  cudaFree(C);
  cudaFree(verify);

  if (correct == 0) {
    printf("INVALID OUTPUT\n");
    exit(1);
  }

  printf("Correct output!\n");
  return 0;
}