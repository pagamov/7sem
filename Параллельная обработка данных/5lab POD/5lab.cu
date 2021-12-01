#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <fstream>

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

void oddEvenSorting(int * array, int N) {
	for (size_t i = 0; i < N; i++) {
		for (size_t j = (i % 2) ? 0 : 1; j + 1 < N; j += 2) {
			if (array[j] > array[j + 1]) {
				std::swap(array[j], array[j + 1]);
			}
		}
	}
}

__global__ void oddEvenSortingStep(int * A, int i, int n, int batch) {
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	int shift = blockDim.x * gridDim.x;
	// for (int var = idx + i + 1; var < n; var += shift)
	// 	A[var + n * i] /= A[i + n * i];
    int start = idx * batch;
    int end = start + batch;
    
    if (i % 2 == 0) {
        for (int j = start; j + 1 < min(end, n); j += 2) {
            if (array[j] > array[j + 1]) {
                int piv = A[j];
                A[j] = A[j + 1];
                A[j + 1] = piv;
            }
        }
    } else {
        for (int j = start + 1; j + 1 < min(end, n); j += 2) {
            if (array[j] > array[j + 1]) {
                int piv = A[j];
                A[j] = A[j + 1];
                A[j + 1] = piv;
            }
        }
    }
}

int main() {
	int n;
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
		oddEvenSortingStep <<<32,32>>> (A_DEV, i, n, 256);
	}
    
	// oddEvenSorting(arr, n);
    
    CSC(cudaMemcpy(arr, ARR_DEV, sizeof(int) * n, cudaMemcpyDeviceToHost));
	
	// fwrite(arr, 4, n, stdout);
    cout << 256 << endl;
    for (int i = 0; i < n; i++) {
        cout << arr[i] << " ";
    }
    cout << endl;
    
    
    CSC(cudaFree(ARR_DEV));
	free(arr);
    return 0;
}