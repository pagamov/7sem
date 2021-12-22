#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <fstream>

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

__global__ void oddEvenSortingStep(int * A, int i, int n, int batch) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int shift = blockDim.x * gridDim.x;
    int piv;
    for (int start = idx * batch; start < n; start += shift * batch) {
        for (int j = start + (i % 2); j + 1 < min(start + batch, n); j += 2) {
            if (A[j] > A[j + 1]) {
                piv = A[j];
                A[j] = A[j + 1];
                A[j + 1] = piv;
            }
        }
    }
}

#define BLOCK_SIZE 5
#define NUM_BLOCKS 10

// __device__ void swap_step(int* nums, int* tmp, int size, int start, int stop, int step, int i) {
// 	// Using shared memory to store blocks and sort them
// 	__shared__ int sh_array[BLOCK_SIZE];

// 	// Step for bitonic merge inside merging
// 	for (int shift = start; shift < stop; shift += step) {
// 		// New start pointer
// 		tmp = nums + shift;

// 		// Right side
// 		if (i >= BLOCK_SIZE / 2)
// 			sh_array[i] = tmp[BLOCK_SIZE * 3 / 2 - 1 - i];
// 		else
// 			sh_array[i] = tmp[i];

// 		__syncthreads();

// 		// From half
// 		for (int j = BLOCK_SIZE / 2; j > 0; j /= 2) {
// 			unsigned int XOR = i ^ j;
// 			// The threads with the lowest ids sort the array
// 			if (XOR > i) {
// 				if ((i & BLOCK_SIZE) != 0) {
// 					// Step descending, swap(i, XOR)
// 					if (sh_array[i] < sh_array[XOR])
// 						thrust::swap(sh_array[i], sh_array[XOR]);
// 				} else {
// 					// Step ascending, swap(i, XOR)
// 					if (sh_array[i] > sh_array[XOR])
// 						thrust::swap(sh_array[i], sh_array[XOR]);
// 				}
// 			}

// 			__syncthreads();
// 		}

// 		// Back from shared to temporary
// 		tmp[i] = sh_array[i];
// 	}
// }


// __global__ void kernel_bitonic_merge_step(int* nums, int size, bool is_odd, bool flag) {
// 	// Temporary array for splitting into blocks
// 	int* tmp = nums;

// 	// Every thread gets exactly one value in the unsorted array
// 	unsigned int i = threadIdx.x;
// 	int id_block = blockIdx.x;
// 	int offset = gridDim.x;

// 	// For odd step
// 	if(is_odd) {
// 		swap_step(nums, tmp, size, (BLOCK_SIZE / 2) + id_block * BLOCK_SIZE, size - BLOCK_SIZE, offset * BLOCK_SIZE, i);
// 	} else { // For even step
// 		swap_step(nums, tmp, size, id_block * BLOCK_SIZE, size, offset * BLOCK_SIZE, i);
// 	}
// }


int main() {
    int n;
    int batch = 5;
    // fread(&n, 4, 1, stdin);
    cin >> n;

    int * arr = (int *)malloc(n * sizeof(int)); // ? what about size <= 16 * 10^6
    // fread(arr, 4, n, stdin);
    for (int i = 0; i < n; i++) {
        cin >> arr[i];
    }

    int * ARR_DEV;
    CSC(cudaMalloc(&ARR_DEV, sizeof(int) * n));
    CSC(cudaMemcpy(ARR_DEV, arr, sizeof(int) * n, cudaMemcpyHostToDevice));

    for (int i = 0; i < n; i++) {
        oddEvenSortingStep <<<2,2>>> (ARR_DEV, i, n, batch);
    }

    /*
	Implementation of odd-even sort
	Sort of buckets with bitonic merge inside
	| 1 3 5 7 | 2 4 6 8 | -> | 1 2 3 4 5 6 7 8| (size == 8)
	Including 2 steps merge + splitting
	*/
	// for (int i = 0; i < 2 * (upd_size / BLOCK_SIZE); ++i) {
	// 	kernel_bitonic_merge_step<<<NUM_BLOCKS, BLOCK_SIZE>>>(dev_data, upd_size, (bool)(i % 2), true);
	// }

    // oddEvenSorting(arr, n);

    CSC(cudaMemcpy(arr, ARR_DEV, sizeof(int) * n, cudaMemcpyDeviceToHost));

    // fwrite(arr, 4, n, stdout);
    cout << batch << endl;
    for (int i = 0; i < n; i++) {
        if (i % batch == 0) {
            cout << "| ";
        }
        cout << arr[i] << " ";

    }
    cout << "|" << endl;


    CSC(cudaFree(ARR_DEV));
    free(arr);
    return 0;
}
