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

// __global__ void findMax(double * A, int * SWP, int i, int n) {
//     double pivotValue = 0;
//     int pivot = -1;
//     for(int row = i; row < n; row++) {
//         if(abs(A[n * i + row]) > pivotValue) {
//             pivotValue = abs(A[n * i + row]);
//             pivot = row;
//         }
//     }
//     SWP[i] = pivot;
// }

struct comparator {												
	__host__ __device__ bool operator()(double a, double b) {		// Функция которая сравнивает объекты на "<"
		return abs(a) < abs(b);										// operator() - переопределение оператора "()" для экземпляра этой структуры
	}
};

__global__ void LUP(double * A, int * SWP, int i, int n, int newidx) {
    // int pivot = SWP[i];
	
	// int idx = x = blockDim.x * blockIdx.x + threadIdx.x;
	// int shift = blockDim.x * gridDim.x;
	
	int pivot = newidx;
    double piv;
	// __shared__ double piv[blockDim.x];
	
    for (int sw = 0; sw < n; sw++) {
       piv = A[pivot + n * sw];
       A[pivot + n * sw] = A[i + n * sw];
       A[i + n * sw] = piv;
    }
	// if (idx < n) {}
	// piv = A[pivot + n * sw];
	
	// __syncthreads();
    for(int j = i+1; j < n; j++) {
       A[j + n * i] /= A[i + n * i];
       for(int k = i+1; k < n; k++) 
           A[j + n * k] -= A[j + n * i] * A[i + n * k];
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
        findMax <<<1,1>>> (A_DEV, SWP_DEV, i, n);
		
		thrust::device_ptr<double> d_ptr = thrust::device_pointer_cast(A_DEV) + (i * n + i);
		thrust::device_ptr<double> max = thrust::max_element(d_ptr, d_ptr + (n - i), comp);
		newidx = max - d_ptr + i;
		// cout << newidx << " " << *(max) << endl;
		newidxarr[i] = newidx;
		
        LUP <<<1,1>>> (A_DEV, SWP_DEV, i, n, newidx);
		// LUP <<<2,2>>> (A_DEV, SWP_DEV, i, n, newidx);
    }
                                      
    CSC(cudaMemcpy(A, A_DEV, sizeof(double) * n * n, cudaMemcpyDeviceToHost));
    CSC(cudaMemcpy(SWP, SWP_DEV, sizeof(int) * n, cudaMemcpyDeviceToHost));
    
    for (int y = 0; y < n; y++) {
        for (int x = 0; x < n; x++)
            printf("%lf ", A[x * n + y]);
        printf("\n");
    }
    for (int i = 0; i < n; i++) {
		printf("%d ", newidxarr[i]);
		// if (SWP[i] != newidxarr[i]) {
		// 	fprintf(stderr, "%d,%d", SWP[i], newidxarr[i]);
		// }
	}
		
        
    printf("\n");
	
	free(newidxarr);
    
    CSC(cudaFree(A_DEV));
    CSC(cudaFree(SWP_DEV));
    free(A);
    free(SWP);
    return 0;
}