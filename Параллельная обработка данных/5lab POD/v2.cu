#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <algorithm>
#include <climits>
#include <thrust/swap.h>
#include <thrust/extrema.h>
#include <thrust/functional.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

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

__device__ void swap_step(int* nums, int* tmp, int size, int start, int stop, int step, int i) {
	// Using shared memory to store blocks and sort them
	__shared__ int sh_array[BLOCK_SIZE];

	// Step for bitonic merge inside merging
	for (int shift = start; shift < stop; shift += step) {
		// New start pointer
		tmp = nums + shift;

		// Right side
		if (i >= BLOCK_SIZE / 2)
			sh_array[i] = tmp[BLOCK_SIZE * 3 / 2 - 1 - i];
		else
			sh_array[i] = tmp[i];

		__syncthreads();

		// From half
		for (int j = BLOCK_SIZE / 2; j > 0; j /= 2) {
			unsigned int XOR = i ^ j;
			// The threads with the lowest ids sort the array
			if (XOR > i) {
				if ((i & BLOCK_SIZE) != 0) {
					// Step descending, swap(i, XOR)
					if (sh_array[i] < sh_array[XOR])
						thrust::swap(sh_array[i], sh_array[XOR]);
				} else {
					// Step ascending, swap(i, XOR)
					if (sh_array[i] > sh_array[XOR])
						thrust::swap(sh_array[i], sh_array[XOR]);
				}
			}

			__syncthreads();
		}

		// Back from shared to temporary
		tmp[i] = sh_array[i];
	}
}

__global__ void kernel_bitonic_merge_step(int* nums, int size, bool is_odd, bool flag) {
	// Temporary array for splitting into blocks
	int* tmp = nums;

	// Every thread gets exactly one value in the unsorted array
	unsigned int i = threadIdx.x;
	int id_block = blockIdx.x;
	int offset = gridDim.x;

	// For odd step
	if(is_odd) {
		swap_step(nums, tmp, size, (BLOCK_SIZE / 2) + id_block * BLOCK_SIZE, size - BLOCK_SIZE, offset * BLOCK_SIZE, i);
	} else { // For even step
		swap_step(nums, tmp, size, id_block * BLOCK_SIZE, size, offset * BLOCK_SIZE, i);
	}
}

__global__ void bitonic_sort_step(int *nums, int j, int k, int size) {
	// Using shared memory to store blocks and sort them
	__shared__ int sh_array[BLOCK_SIZE];

	// Temporary array for splitting into blocks
	int* tmp = nums;

	// Every thread gets exactly one value in the unsorted array
	unsigned int i = threadIdx.x;
	int id_block = blockIdx.x;
	int offset = gridDim.x;

	// Step for bitonic sort
	for (int shift = id_block * BLOCK_SIZE; shift < size; shift += offset * BLOCK_SIZE) {
			// New start pointer
			tmp = nums + shift;

			// Store in shared memory
			sh_array[i] = tmp[i];

			__syncthreads();

			// From half
			for (j = k / 2; j > 0; j /= 2) {
				unsigned int XOR = i ^ j;
				// The threads with the lowest ids sort the array
				if (XOR > i) {
					if ((i & k) != 0) {
						// Step descending, swap(i, XOR)
						if (sh_array[i] < sh_array[XOR])
							thrust::swap(sh_array[i], sh_array[XOR]);
					} else {
						// Step ascending, swap(i, XOR)
						if (sh_array[i] > sh_array[XOR])
							thrust::swap(sh_array[i], sh_array[XOR]);
					}
				}

				__syncthreads();
			}

			// Back from shared to temporary
			tmp[i] = sh_array[i];
		}
}

int main() {
	bool verbose = false; // 0 for binary, 1 for normal
	int n, upd_n;

	if (verbose)
        cin >> n;
    else
        fread(&n, 4, 1, stdin);

	// To the degree of 2^n (1024 max)
	upd_n = ceil((double)n / BLOCK_SIZE) * BLOCK_SIZE;
	int * data = (int *)malloc(sizeof(int) * upd_n);
	int * dev_data;
	CSC(cudaMalloc(&dev_data, sizeof(int) * upd_n));

	if (verbose)
        for (int i = 0; i < n; i++)
            cin >> data[i];
    else
        fread(data, 4, n, stdin);

	for (int i = n; i < upd_n; ++i)
		data[i] = INT_MAX;

	CSC(cudaMemcpy(dev_data, data, sizeof(int) * upd_n, cudaMemcpyHostToDevice));

	// Pre sort of all blocks by bitonic sort
	// Main step
	for (int k = 2; k <= upd_n; k *= 2) {
		if (k > BLOCK_SIZE)
			break;
		// Merge and split step
		for (int j = k / 2; j > 0; j /= 2) {
			bitonic_sort_step<<<NUM_BLOCKS, BLOCK_SIZE>>>(dev_data, j, k, upd_n);
			CSC(cudaGetLastError());
		}
	}

	/*
	Implementation of odd-even sort
	Sort of buckets with bitonic merge inside
	| 1 3 5 7 | 2 4 6 8 | -> | 1 2 3 4 5 6 7 8| (n == 8)

	Including 2 steps merge + splitting
	*/
	for (int i = 0; i < 2 * (upd_n / BLOCK_SIZE); ++i) {
		kernel_bitonic_merge_step<<<NUM_BLOCKS, BLOCK_SIZE>>>(dev_data, upd_n, (bool)(i % 2), true);
		CSC(cudaGetLastError());
	}

	CSC(cudaMemcpy(data, dev_data, sizeof(int) * upd_n, cudaMemcpyDeviceToHost))

	if (verbose) {
        for (int i = 0; i < n; i++)
            cout << data[i] << " ";
        cout << endl;
    } else {
        fwrite(data, 4, n, stdout);
    }

	CSC(cudaFree(dev_data));
	free(data);
	return 0;
}
