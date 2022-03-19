#include <stdio.h>
#include <stdlib.h>

__global__ void kernel(double *arr, int n) {
	int i, idx = blockDim.x * blockIdx.x + threadIdx.x;
	int offset = blockDim.x * gridDim.x;
	for(i = idx; i < n; i += offset)	
		arr[i] = arr[i] < 0 ? -arr[i] : arr[i];
}

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

int main() {
	int i, n;
	scanf("%d",&n);
	double *arr = (double *)malloc(sizeof(double) * n);
	for(i = 0; i < n; i++)
		scanf("%lf", &arr[i]);

	double *dev_arr;
	gpuErrchk(cudaMalloc(&dev_arr, sizeof(double) * n));
	gpuErrchk(cudaMemcpy(dev_arr, arr, sizeof(double) * n, cudaMemcpyHostToDevice));

	kernel<<<256, 256>>>(dev_arr, n);
	gpuErrchk( cudaPeekAtLastError() );
	gpuErrchk( cudaDeviceSynchronize() );

	gpuErrchk( cudaMemcpy(arr, dev_arr, sizeof(double) * n, cudaMemcpyDeviceToHost));
	gpuErrchk( cudaFree(dev_arr));
	for(i = 0; i < n; i++)
		printf("%lf ", arr[i]);
	printf("\n");
	free(arr);
	return 0;
}
