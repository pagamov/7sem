#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <string>
#include <string.h>
#include <math.h>

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

__global__ void Kmean(uchar4 * pic, uchar4 * cl, int w, int h, int n) {
	// rebuild to cl as const mem :!
    for (int y = blockDim.y * blockIdx.y + threadIdx.y; y < h; y += blockDim.y * gridDim.y) {
        for (int x = blockDim.x * blockIdx.x + threadIdx.x; x < w; x += blockDim.x * gridDim.x) {
            int resClas = -1;
            float maxDist = sqrt((float)3*(255*255))+1.0F;
			uchar4 piv = pic[x + w * y];
            for (int i = 0; i < n; i++) {
                float pivDist = sqrt( 					   \
					pow((float)piv.x - (float)cl[i].x,2) + \
					pow((float)piv.y - (float)cl[i].y,2) + \
					pow((float)piv.z - (float)cl[i].z,2)   \
				);
                if (pivDist < maxDist) {
                    resClas = i;
                    maxDist = pivDist;
                }
            }
            pic[x + y * w].w = (unsigned char)resClas;
        }
    }
}

__global__ void reBuild(uchar4 * pic, uchar4 * cl, uchar4 * newcl, int w, int h, int n) {
	// rebuild to cl as const mem :!
    for (int y = blockDim.y * blockIdx.y + threadIdx.y; y < h; y += blockDim.y * gridDim.y) {
        for (int x = blockDim.x * blockIdx.x + threadIdx.x; x < w; x += blockDim.x * gridDim.x) {
			// will write to newcl new rgb of classes
        }
    }
}

#define SIZE_OF_PIC sizeof(uchar4) * w * h
#define SIZE_OF_CLU sizeof(uchar4) * n

int main() {
    string filename1, filename2;
    int w, h, n, x, y;
    cin >> filename1 >> filename2 >> n;
	
	// read data section
	FILE * f = fopen(filename1.c_str(), "rb");
	fread(&w, sizeof(int), 1, f);
	fread(&h, sizeof(int), 1, f);
	uchar4 * data = (uchar4 *)malloc(SIZE_OF_PIC); //#
	fread(data, sizeof(uchar4), w * h, f);
	fclose(f);
	
	// read data for classes
	uchar4 * clu = (uchar4 *)malloc(SIZE_OF_CLU); //#
    // int * cl = (int *)malloc(SIZE_OF_CL); //#
    for (int i = 0; i < n; i++) {
		cin >> x >> y;
		clu[i] = data[x + w * y];
	}
	// make dev struct for kernel
	uchar4 * dev_pic, * dev_cl;
	CSC(cudaMalloc(&dev_pic, SIZE_OF_PIC));
    CSC(cudaMalloc(&dev_cl, SIZE_OF_CLU));
	// copy data to dev struct
	CSC(cudaMemcpy(dev_pic, data, SIZE_OF_PIC, cudaMemcpyHostToDevice));
    CSC(cudaMemcpy(dev_cl, clu, SIZE_OF_CLU, cudaMemcpyHostToDevice));
    
	Kmean <<<dim3(16, 16), dim3(32, 32)>>> (dev_pic, dev_cl, w, h, n);

	CSC(cudaGetLastError());
	CSC(cudaMemcpy(data, dev_pic, SIZE_OF_PIC, cudaMemcpyDeviceToHost));
	
	CSC(cudaFree(dev_pic));
    CSC(cudaFree(dev_cl));
    
	f = fopen(filename2.c_str(), "wb");
	fwrite(&w, sizeof(int), 1, f);
	fwrite(&h, sizeof(int), 1, f);
	fwrite(data, sizeof(uchar4), w * h, f);
	fclose(f);

	free(data);
	free(clu);
	return 0;
}
