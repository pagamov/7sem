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

typedef struct { unsigned x, y, z; } Uint4; 									// +4 294 967 295 ~ 255 * 4100 * 4100
typedef struct { int w, h, n; } info;											// pic 4100 by 4100 of one claster max pixel
																				// mb need to be changed to higher
__constant__ uchar4 cl[32];														// contant classes no more than 32 by default
// __constant__ info inf;															// some else param that never changes

#define SIZE_OF_PIC sizeof(uchar4) * w * h
#define SIZE_OF_CLU sizeof(uchar4) * n

__global__ void Kmean(uchar4 * pic, info * inf) {
    for (int y = blockDim.y * blockIdx.y + threadIdx.y; y < inf->h; y += blockDim.y * gridDim.y) {
        for (int x = blockDim.x * blockIdx.x + threadIdx.x; x < inf->w; x += blockDim.x * gridDim.x) {
            int resClas = -1;
            float maxDist = sqrt((float)3*(255*255))+1.0F;
			uchar4 piv = pic[x + inf->w * y];
            for (int i = 0; i < inf->n; i++) {
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
            pic[x + y * inf->w].w = (unsigned char)resClas;
        }
    }
}

__global__ void reBuild(uchar4 * pic, Uint4 * newcl, info * inf) {
	int idx = threadIdx.x;
	if (idx < inf->n) {
		unsigned char uidx = (unsigned char)idx;
		int num = 0;
		newcl[uidx].x = 0; newcl[uidx].y = 0; newcl[uidx].z = 0;
	    for (int y = 0; y < inf->h; y++) {
	        for (int x = 0; x < inf->w; x++) {
				if (pic[x + inf->w * y].w == uidx) {
					newcl[uidx].x += pic[x + inf->w * y].x;
					newcl[uidx].y += pic[x + inf->w * y].y;
					newcl[uidx].z += pic[x + inf->w * y].z;
					num += 1;
				}
	        }
	    }
		if (num != 0) {
			newcl[uidx].x /= num;
			newcl[uidx].y /= num;
			newcl[uidx].z /= num;
		}
	}
}

int main() {
    string filename1, filename2;
    int w, h, n, x, y, flag = 1;
    cin >> filename1 >> filename2 >> n;
	
	FILE * f = fopen(filename1.c_str(), "rb");									// read data section
	fread(&w, sizeof(int), 1, f);
	fread(&h, sizeof(int), 1, f);
	uchar4 * data = (uchar4 *)malloc(SIZE_OF_PIC); 								// malloc data
	fread(data, sizeof(uchar4), w * h, f);
	fclose(f);
	
	info * infHost = (info *)malloc(sizeof(info));								// malloc infHost
	infHost->w = w; infHost->h = h; infHost->n = n;								// copy info data and never touch again
	info * inf;
	CSC(cudaMalloc(&inf, sizeof(info)));	
	CSC(cudaMemcpy(inf, infHost, sizeof(info), cudaMemcpyHostToDevice));
	// CSC(cudaMemcpyToSymbol(inf, infHost, sizeof(info), 0, cudaMemcpyHostToDevice));
	
	uchar4 clHost[32];															// read data for classes
    for (int i = 0; i < n; i++) {
		cin >> x >> y;
		clHost[i] = data[x + w * y];
	} 																			// we ll touch it every cicle
	CSC(cudaMemcpyToSymbol(cl, clHost, sizeof(clHost), 0, cudaMemcpyHostToDevice));
	
	uchar4 * dev_pic;															// make dev struct for kernel
	CSC(cudaMalloc(&dev_pic, SIZE_OF_PIC));										// cuda malloc dev_pic
	
	Uint4 * dev_clnew;
	CSC(cudaMalloc(&dev_clnew, sizeof(Uint4) * 32));							// cuda malloc dev_clnew
	
	Uint4 * clnew; 																// but why this way?
	clnew = (Uint4 *)malloc(sizeof(Uint4) * 32);								// malloc clnew
	CSC(cudaMemcpy(dev_pic, data, SIZE_OF_PIC, cudaMemcpyHostToDevice));
	
	while (flag) {
		Kmean <<<dim3(16, 16), dim3(32, 32)>>> (dev_pic, inf);						// find new clasters
		reBuild <<<1, 32>>> (dev_pic, dev_clnew, inf);								// save in clnew new centers by rgb
		
		CSC(cudaMemcpy(clnew, dev_clnew, sizeof(Uint4) * 32, cudaMemcpyDeviceToHost));
		
		flag = 0;
		for (int i = 0; i < n; i++) {
			if (!(clHost[i].x == (unsigned char)clnew[i].x && \
				  clHost[i].y == (unsigned char)clnew[i].y && \
				  clHost[i].z == (unsigned char)clnew[i].z)) {
					  flag = 1;
			}
		}
		
		for (int i = 0; i < n; i++) {
			clHost[i].x = (unsigned char)clnew[i].x;
			clHost[i].y = (unsigned char)clnew[i].y;
			clHost[i].z = (unsigned char)clnew[i].z;
		}
		
		CSC(cudaMemcpyToSymbol(cl, clHost, sizeof(clHost), 0, cudaMemcpyHostToDevice));
	}
    
	CSC(cudaGetLastError());
	
	CSC(cudaMemcpy(data, dev_pic, SIZE_OF_PIC, cudaMemcpyDeviceToHost));
	
	CSC(cudaFree(dev_pic));														// cuda free dev_pic
	CSC(cudaFree(dev_clnew));													// cuda free dev_clnew
    
	CSC(cudaFree(inf));
	
	f = fopen(filename2.c_str(), "wb");
	fwrite(&w, sizeof(int), 1, f);
	fwrite(&h, sizeof(int), 1, f);
	fwrite(data, sizeof(uchar4), w * h, f);
	fclose(f);

	free(data);																	// free data
	free(clnew);																// free clnew
	free(infHost);																// free infHost
	return 0;
}
