#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <string>

__global__ void sub (const double *v1, const double *v2, double *res, int n) {
    for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < n; i += gridDim.x * blockDim.x) {
        res[i] = v1[i] - v2[i];
    }
}

#define N sizeof(double)*n

int main() {
    int i, n;
    double p;
    std::cin >> n;

    double *arr1 = (double *)malloc(N);
    double *arr2 = (double *)malloc(N);
    double *res =  (double *)malloc(N);

    for (i = 0; i < n; i++) {
        std::cin >> p;
        arr1[i] = p;
    }
    for (i = 0; i < n; i++) {
        std::cin >> p;
        arr2[i] = p;
    }

    double *dev_arr1, *dev_arr2, *dev_res;
    cudaMalloc(&dev_arr1, N);
    cudaMalloc(&dev_arr2, N);
    cudaMalloc(&dev_res,  N);

    cudaMemcpy(dev_arr1, arr1, N, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_arr2, arr2, N, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_res, res, N, cudaMemcpyHostToDevice);

    sub <<<2,2>>>(dev_arr1, dev_arr2, dev_res, n);

    cudaDeviceSynchronize();

    cudaMemcpy(res, dev_res, N, cudaMemcpyDeviceToHost);

    for (i = 0; i < n; i++)
        printf("%lf ", res[i]);
    std::cout << std::endl;

    cudaFree(dev_arr1);
    cudaFree(dev_arr2);
    cudaFree(dev_res);

    free(arr1);
    free(arr2);
    free(res);
    return 0;
}
