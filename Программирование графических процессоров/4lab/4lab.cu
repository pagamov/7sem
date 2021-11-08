#include <iostream>
#include <stdio.h>
#include <stdlib.h>

#include <thrust/device_vector.h>
#include <thrust/extrema.h>
#include <thrust/count.h>

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

struct comparator {												
	__host__ __device__ bool operator()(double a, double b) {
		return abs(a) < abs(b);
	}
};

__global__ void LUP_swap(double * A, int i, int n, int newidx) {
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	int shift = blockDim.x * gridDim.x;
	double piv;
	for (int var = idx; var < n; var += shift) {
		piv = A[newidx + n * var];
        A[newidx + n * var] = A[i + n * var];
        A[i + n * var] = piv;
	}
}

__global__ void LUP_N(double * A, int i, int n) {
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	int shift = blockDim.x * gridDim.x;
	for (int var = idx + i + 1; var < n; var += shift) {
		A[var + n * i] /= A[i + n * i];
	}
}

__global__ void LUP(double * A, int i, int n) {
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	int idy = blockDim.y * blockIdx.y + threadIdx.y;
	
	int shiftx = blockDim.x * gridDim.x;
	int shifty = blockDim.y * gridDim.y;
	
	for (int var = idx + i + 1; var < n; var += shiftx) {
		for (int k = idy + i + 1; k < n; k += shifty) {
			A[var + n * k] -= A[var + n * i] * A[i + n * k];
		}
	}
}

int main() {
    int n;
	comparator comp;
    cin >> n;
    double * A = (double *)malloc(sizeof(double) * n * n);
    
    for (int y = 0; y < n; y++)
        for (int x = 0; x < n; x++)
            cin >> A[x * n + y];
            
    double * A_DEV;
    CSC(cudaMalloc(&A_DEV, sizeof(double) * n * n));
    CSC(cudaMemcpy(A_DEV, A, sizeof(double) * n * n, cudaMemcpyHostToDevice));
    
    int * SWP = (int *)malloc(sizeof(int) * n);
    int * SWP_DEV;
    CSC(cudaMalloc(&SWP_DEV, sizeof(int) * n));
    
	int newidx;
	
	int * newidxarr = (int *)malloc(sizeof(int) * n);
	
    for(int i = 0; i < n; i++) {
		thrust::device_ptr<double> d_ptr = thrust::device_pointer_cast(A_DEV) + (i * n + i);
		thrust::device_ptr<double> max = thrust::max_element(d_ptr, d_ptr + (n - i), comp);
		newidx = max - d_ptr + i;
		newidxarr[i] = newidx;
		LUP_swap <<<32,32>>> (A_DEV, i, n, newidx);
		LUP_N <<<32,32>>> (A_DEV, i, n);
		LUP <<<dim3(32,32),dim3(32,32)>>> (A_DEV, i, n);
    }
	                      
    CSC(cudaMemcpy(A, A_DEV, sizeof(double) * n * n, cudaMemcpyDeviceToHost));
    CSC(cudaMemcpy(SWP, SWP_DEV, sizeof(int) * n, cudaMemcpyDeviceToHost));
    
    for (int y = 0; y < n; y++) {
        for (int x = 0; x < n; x++)
            printf("%.10lf ", A[x * n + y]);
        printf("\n");
    }
    for (int i = 0; i < n; i++) {
		printf("%d ", newidxarr[i]);
	}
    printf("\n");
	
	free(newidxarr);
    CSC(cudaFree(A_DEV));
    CSC(cudaFree(SWP_DEV));
    free(A);
    free(SWP);
    return 0;
}