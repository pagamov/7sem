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


#define CSC(call)  													\
do {																\
	cudaError_t res = call;											\
	if (res != cudaSuccess) {										\
		fprintf(stderr, "ERROR in %s:%d. Message: %s\n",			\
				__FILE__, __LINE__, cudaGetErrorString(res));		\
		exit(0);													\
	}																\
} while(0)


__device__ void swap_GPU(int& a, int& b) {
	int tmp = a;
	a = b;
	b = tmp;
}

__device__ void print_data(int* data, int N) {
	for(int i = 0; i < N; i++) {
		printf("%d ", data[i]);
	}
	printf("\n");
}

void swap(int& a, int& b) {
	int tmp = a;
	a = b;
	b = tmp;
}

void B(int* data, int size, int N, int sign_shift) {
	for(int i = 0; i < N; i+=size) {
		for(int j = 0; j < size / 2; j++) {
			if ((data[i+j] > data[i+j + size / 2]) == (((i+j) / sign_shift) % 2 == 0)) {
				swap(data[i+j], data[i+j + size / 2]);
			}
		}
	}
}


__global__ void  B_shared_kernal(int* data, int size_p, int N, int sign_shift_p) {
	int sign_shift = 1<< sign_shift_p;

	__shared__ int buff[512];

	for(unsigned long long i = (unsigned long long)blockIdx.x * 512; i < N; i += (unsigned long long)gridDim.x * 512) {
		for(int k = threadIdx.x; k < 512; k += blockDim.x)
			buff[k] = data[i + k];
		__syncthreads();

		for(unsigned long long size_k_p = size_p; size_k_p >= 1; size_k_p--) {
			unsigned long long size_k = 1<<size_k_p;
			for(unsigned long long j = threadIdx.x; j < 256; j += blockDim.x) {
				unsigned long long z = (unsigned long long)(j >> (size_k_p-1)) * size_k + (j & ((1 << (size_k_p-1)) - 1));
				if ((buff[z] > buff[z + (size_k >> 1)]) != (((i+z) / sign_shift) & 1)) {
					swap_GPU(buff[z], buff[z + (size_k >> 1)]);
				}
			}

			__syncthreads();
		}
		for(int k = threadIdx.x; k < 512; k += blockDim.x)
			data[i + k] = buff[k];
		__syncthreads();
	}
}


__global__ void  B_global_kernal(int* data, int size_p, int N, int sign_shift_p) {
	unsigned long long sign_shift = 1<< sign_shift_p;
	unsigned long long size = 1<<size_p;
	for(unsigned long long i = (unsigned long long)blockIdx.x * size; i < N; i += (unsigned long long)gridDim.x * size) {
		for(unsigned long long j = threadIdx.x; j < size / 2; j += blockDim.x) {
			if ((data[i+j] > data[i+j + size / 2]) == (((i+j) / sign_shift) % 2 == 0)) {
				swap_GPU(data[i+j], data[i+j + size / 2]);
			}
		}
	}
}

void M(int* data, int size, int N) {
	for(int i = size; i >= 2; i /= 2) {
		B(data, i, N, size);

	}
}

void M_GPU(int* dev_data, int size, int N) {
	for(int i = size; i >= 1; i--) {
		if(i <= 9) {
			B_shared_kernal<<<128, 128>>>(dev_data, i, N, size);
			break;
		}
		B_global_kernal<<<128, 1024>>>(dev_data, i, N, size);

		CSC(cudaGetLastError());
	}
}

unsigned long upper_power_of_two(unsigned long v)
{
    v--;
    v |= v >> 1;
    v |= v >> 2;
    v |= v >> 4;
    v |= v >> 8;
    v |= v >> 16;
    v++;
    return v;

}

int main() {
	unsigned int  n;

	fread(&n, sizeof(int), 1, stdin);
	unsigned int  N = upper_power_of_two(n);
	int *data = (int *)malloc(sizeof(int) * N);

	fread(data, sizeof(int), n, stdin);

	for(int i = n; i < N; i++) {
		data[i] = INT_MAX;
	}

	int *dev_data;
	CSC(cudaMalloc(&dev_data, sizeof(int) * N));
	CSC(cudaMemcpy(dev_data, data, sizeof(int) * N, cudaMemcpyHostToDevice));

	for(int i = 1; (1<<i) <= N; i++) {
		M_GPU(dev_data, i, N);
	}
	CSC(cudaGetLastError());

	CSC(cudaMemcpy(data, dev_data, sizeof(int) * N, cudaMemcpyDeviceToHost));
	CSC(cudaFree(dev_data));

	fwrite(data, sizeof(int), n, stdout);

	free(data);

	return 0;
}
