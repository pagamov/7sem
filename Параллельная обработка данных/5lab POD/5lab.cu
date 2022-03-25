#include <stdio.h>
#include <iostream>
#include <math.h>
#include <stdlib.h>

using namespace std;

#define SR_S 512
#define ull unsigned long long

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

__global__ void B_shared(int * arr, int size_p, int upd_n, int sign_p) {
	int tmp;
	__shared__ int buf[SR_S];
	for (ull i = (ull)blockIdx.x * SR_S; i < upd_n; i += (ull)gridDim.x * SR_S) {
		for (int k = threadIdx.x; k < SR_S; k += blockDim.x)
			buf[k] = arr[i + k];
		__syncthreads();

		for (ull size_k_p = size_p; size_k_p >= 1; size_k_p--) {
			ull size_k = 1 << size_k_p;
			for (ull j = threadIdx.x; j < SR_S / 2; j += blockDim.x) {
				ull b = (ull)(j >> (size_k_p - 1)) * size_k + (j & ((1 << (size_k_p - 1)) - 1));
				if ((buf[b] > buf[b + (size_k >> 1)]) != (((i + b) / (1 << sign_p)) & 1)) {
					tmp = buf[b];
					buf[b] = buf[b + (size_k >> 1)];
					buf[b + (size_k >> 1)] = tmp;
				}
			}
			__syncthreads();
		}
		for (int k = threadIdx.x; k < SR_S; k += blockDim.x)
			arr[i + k] = buf[k];
		__syncthreads();
	}
}

__global__ void B_global(int * arr, int size_p, int upd_n, int sign_p) {
	int tmp;
	ull size = 1 << size_p;
	for (ull i = (ull)blockIdx.x * size; i < upd_n; i += (ull)gridDim.x * size) {
		for (ull j = threadIdx.x; j < size / 2; j += blockDim.x) {
			if ((arr[i + j] > arr[i + j + size / 2]) == (((i + j) / (1 << sign_p)) % 2 == 0)) {
				tmp = arr[i+j];
				arr[i + j] = arr[i + j + size / 2];
				arr[i + j + size / 2] = tmp;
			}
		}
	}
}

int main() {
    bool verbose = false; // 0 for binary, 1 for normal
    int n, upd_n;

    if (verbose)
        cin >> n;
    else
        fread(&n, 4, 1, stdin);

    int p = 0;
    while (pow(2, p) < n)
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

	for (int i = 1; pow(2, i) <= upd_n; i++) {
		for (int j = i; j >= 1; j--) {
			if (j <= 9) {
				B_shared <<<64, 64>>> (dev_arr, j, upd_n, i);
				break;
			} else
				B_global <<<64, 128>>> (dev_arr, j, upd_n, i);
			CSC(cudaDeviceSynchronize());
			CSC(cudaGetLastError());
		}
	}
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
