#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <thrust/swap.h>

using namespace std;

#define CSC(call)                                                   \
do {                                                                \
    cudaError_t res = call;                                         \
    if (res != cudaSuccess) {                                       \
        fprintf(stderr, "ERROR in %s:%d. Message: %s\n",            \
                __FILE__, __LINE__, cudaGetErrorString(res));       \
        exit(0);                                                    \
    }                                                               \
} while(0)

#define NUM_BLOCKS 10
#define BLOCK_SIZE 1024

__global__ void oddEvenSortingStep(int * A, int i, int n, int batch) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int shift = blockDim.x * gridDim.x;
    for (int start = idx * batch; start < n; start += shift * batch)
        for (int j = start + (i % 2); j + 1 < min(start + batch, n); j += 2)
            if (A[j] > A[j + 1])
                thrust::swap(A[j], A[j + 1]);
}

__device__ void swp(int* nums, int* tmp, int size, int start, int stop, int step, int i) {
	__shared__ int sh[BLOCK_SIZE];

	for (int shift = start; shift < stop; shift += step) {
		tmp = nums + shift;

		if (i >= BLOCK_SIZE / 2)
			sh[i] = tmp[BLOCK_SIZE * 3 / 2 - 1 - i];
		else
			sh[i] = tmp[i];
		__syncthreads();

		for (int j = BLOCK_SIZE / 2; j > 0; j /= 2) {
			unsigned int XOR = i ^ j;
			if (XOR > i) {
				if ((i & BLOCK_SIZE) != 0) {
					if (sh[i] < sh[XOR])
						thrust::swap(sh[i], sh[XOR]);
				} else {
					if (sh[i] > sh[XOR])
						thrust::swap(sh[i], sh[XOR]);
				}
			}
			__syncthreads();
		}
		tmp[i] = sh[i];
	}
}

__global__ void bitonic_merge(int * arr, int size, bool is_odd) {
	int * tmp = arr;
	if (is_odd)
		swp(arr, tmp, size, (BLOCK_SIZE / 2) + blockIdx.x * BLOCK_SIZE, size - BLOCK_SIZE, gridDim.x * BLOCK_SIZE, threadIdx.x);
	else
		swp(arr, tmp, size, blockIdx.x * BLOCK_SIZE, size, gridDim.x * BLOCK_SIZE, threadIdx.x);
}

int main() {
	int n, upd_n;
	fread(&n, 4, 1, stdin);
	upd_n = ceil((double)n / BLOCK_SIZE) * BLOCK_SIZE;
	int * data = (int *)malloc(sizeof(int) * upd_n);
	int * dev_data;
	CSC(cudaMalloc(&dev_data, sizeof(int) * upd_n));
	fread(data, 4, n, stdin);

	for (int i = n; i < upd_n; ++i)
		data[i] = INT_MAX;

	CSC(cudaMemcpy(dev_data, data, sizeof(int) * upd_n, cudaMemcpyHostToDevice));

	for (int i = 0; i < BLOCK_SIZE; i++)
        oddEvenSortingStep <<<NUM_BLOCKS,BLOCK_SIZE>>> (dev_data, i, n, BLOCK_SIZE);
	CSC(cudaGetLastError());

	for (int i = 0; i < 2 * (upd_n / BLOCK_SIZE); i++)
		bitonic_merge<<<NUM_BLOCKS, BLOCK_SIZE>>>(dev_data, upd_n, (bool)(i % 2));
	CSC(cudaGetLastError());

	CSC(cudaMemcpy(data, dev_data, sizeof(int) * upd_n, cudaMemcpyDeviceToHost));

    fwrite(data, 4, n, stdout);

	CSC(cudaFree(dev_data));
	free(data);
	return 0;
}
