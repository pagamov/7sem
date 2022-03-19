#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <sys/time.h>
#include <thrust/extrema.h>
#include <thrust/device_vector.h>
#include <algorithm>
#include <limits>

using namespace std;

#define CSC(call)  													\
do {																\
	cudaError_t res = call;											\
	if (res != cudaSuccess) {										\
		fprintf(stderr, "ERROR in %s:%d. Message: %s\n",			\
				__FILE__, __LINE__, cudaGetErrorString(res));		\
		exit(0);													\
	}																\
} while(0)

int pow(int n, int p) {
    int res = 1;
    for (int i = 0; i < p; i++)
        res *= n;
    return res;
}

__global__ void  B_shared(int * data, int size_p, int upd_n, int sign_shift_p) {
    int tmp;

	__shared__ int buff[512];

	for (int i = blockIdx.x * 512; i < upd_n; i += gridDim.x * 512) {
		for (int k = threadIdx.x; k < 512; k += blockDim.x)
			buff[k] = data[i + k];
		__syncthreads();

		for (int size_k_p = size_p; size_k_p >= 1; size_k_p--) {
			int size_k = 1 << size_k_p;
			for (int j = threadIdx.x; j < 256; j += blockDim.x) {
				int z = (int)(j >> (size_k_p-1)) * size_k + (j & ((1 << (size_k_p-1)) - 1));
				if ((buff[z] > buff[z + (size_k >> 1)]) != (((i+z) / (1 << sign_shift_p)) & 1)) {
					tmp = buff[z];
                    buff[z] = buff[z + (size_k >> 1)];
                    buff[z + (size_k >> 1)] = tmp;
				}
			}

			__syncthreads();
		}
		for(int k = threadIdx.x; k < 512; k += blockDim.x)
			data[i + k] = buff[k];
		__syncthreads();
	}
}


__global__ void  B_global(int* data, int size_p, int upd_n, int sign_shift_p) {
    int tmp;
	int size = 1 << size_p;
	for (int i = blockIdx.x * size; i < upd_n; i += gridDim.x * size)
		for (int j = threadIdx.x; j < size / 2; j += blockDim.x)
			if ((data[i+j] > data[i+j + size / 2]) == (((i+j) / (1 << sign_shift_p)) % 2 == 0)) {
				tmp = data[i+j];
                data[i+j] = data[i+j + size / 2];
                data[i+j + size / 2] = tmp;
			}
}

int main() {
    bool verbose = true; // 0 for binary, 1 for normal
    int n, upd_n;

    if (verbose)
        cin >> n;
    else
        fread(&n, 4, 1, stdin);

    int p = 0;
    while (pow(2,p) < n)
        p++;
    upd_n = pow(2, p);

	int * arr = (int *)malloc(4 * upd_n);

    if (verbose)
        for (int i = 0; i < n; i++)
            cin >> arr[i];
    else
        fread(arr, 4, n, stdin);

	for (int i = n; i < upd_n; i++)
		arr[i] = INT_MAX;

	int * dev_arr;
	CSC(cudaMalloc(&dev_arr, 4 * upd_n));
	CSC(cudaMemcpy(dev_arr, arr, 4 * upd_n, cudaMemcpyHostToDevice));

	for (int i = 1; pow(2,i) <= upd_n; i++) {
		for (int j = i; j >= 1; j--) {
			if (j <= 9)
				B_shared <<<128, 128>>> (dev_arr, j, upd_n, i);
			else
				B_global <<<128, 1024>>> (dev_arr, j, upd_n, i);
		}
	}

	CSC(cudaGetLastError());
	CSC(cudaMemcpy(arr, dev_arr, 4 * upd_n, cudaMemcpyDeviceToHost));

    if (verbose) {
        for (int i = 0; i < n; i++)
            cout << arr[i] << " ";
        cout << endl;
    } else {
        fwrite(arr, 4, n, stdout);
    }

    CSC(cudaFree(dev_arr));
	free(arr);
	return 0;
}
