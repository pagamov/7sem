#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <fstream>
#include <thrust/swap.h>
#include <thrust/extrema.h>
#include <thrust/functional.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

using namespace std;

#define BLOCK_SIZE 512
#define NUM_BLOCKS (2048+2048)

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

__global__ void mergeGPU(int * arr, int upd_n, int batch, int start) {
    __shared__ int l[BLOCK_SIZE];
    __shared__ int r[BLOCK_SIZE];

    for (int st = start + blockDim.x * 2 * blockIdx.x; st + 2 * blockDim.x < upd_n+1; st += blockDim.x * 2 * gridDim.x) {
        l[threadIdx.x] = arr[st + threadIdx.x];
        __syncthreads();
        r[threadIdx.x] = arr[st + threadIdx.x + blockDim.x];
        __syncthreads();

        if (threadIdx.x == 0) {
            int rc = 0;
            int lc = 0;
            int it = st; //?

            // int count = 0;
            while (true) {
                // count++;
                if (rc == batch)
                    arr[it] = l[lc++];
                else if (lc == batch)
                    arr[it] = r[rc++];
                else {
                    if (l[lc] < r[rc])
                        arr[it] = l[lc++];
                    else if (l[lc] > r[rc])
                        arr[it] = r[rc++];
                    else
                        arr[it] = l[lc++];
                }
                it++;
                if (lc == batch && rc == batch)
                    break;
                // if (count == blockDim.x * 2)
                //     break;
            }
        }
    }

    __syncthreads();
}

int main() {
    bool verbose = true; // 0 for binary, 1 for normal
    int n, upd_n;

    if (verbose)
        cin >> n;
    else
        fread(&n, 4, 1, stdin);

    if (n % BLOCK_SIZE != 0)
        upd_n = (n / BLOCK_SIZE + 1) * BLOCK_SIZE;
    else
        upd_n = n;

    int * arr = (int *)malloc(upd_n * sizeof(int));

    if (verbose)
        for (int i = 0; i < n; i++)
            cin >> arr[i];
    else
        fread(arr, 4, n, stdin);

    for (int i = n; i < upd_n; i++)
        arr[i] = INT_MAX;

    int * ARR_DEV;
    CSC(cudaMalloc(&ARR_DEV, sizeof(int) * upd_n));
    CSC(cudaMemcpy(ARR_DEV, arr, sizeof(int) * upd_n, cudaMemcpyHostToDevice));

    // odd even sort
    for (int i = 0; i < n; i++) {
        oddEvenSortingStep <<<NUM_BLOCKS,BLOCK_SIZE>>> (ARR_DEV, i, n, BLOCK_SIZE);
    }

    // bitonic merge sort
    for (int i = 0; i < 2 * (upd_n / BLOCK_SIZE); i++) {
        if (i % 2 == 0) {
            mergeGPU <<<NUM_BLOCKS,BLOCK_SIZE>>> (ARR_DEV, upd_n, BLOCK_SIZE, 0);
        } else {
            mergeGPU <<<NUM_BLOCKS,BLOCK_SIZE>>> (ARR_DEV, upd_n, BLOCK_SIZE, BLOCK_SIZE);
        }
    }

    CSC(cudaGetLastError());
    CSC(cudaMemcpy(arr, ARR_DEV, sizeof(int) * upd_n, cudaMemcpyDeviceToHost));

    if (verbose) {
        // cout << upd_n << ' ' << n << endl;
        for (int i = 0; i < n; i++) {
            // if (i % BLOCK_SIZE == 0)
                // cout << "| ";
            cout << arr[i] << " ";
        }
        cout << endl;
        // cout << "|" << endl;
    } else {
        fwrite(arr, 4, n, stdout);
    }




    CSC(cudaFree(ARR_DEV));
    free(arr);
    return 0;
}
