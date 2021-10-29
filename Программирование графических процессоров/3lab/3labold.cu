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

typedef struct { unsigned x, y, z; } Uint3; 									// +4 294 967 295 ~ 255 * 4100 * 4100
typedef struct { int w, h, n; } info;											// pic 4100 by 4100 of one claster max pixel
																				// mb need to be changed to higher
__constant__ uchar3 cl[32];														// contant classes no more than 32 by default
__constant__ info inf[1];														// some else param that never changes
#define SIZE_OF_PIC sizeof(uchar4) * w * h
#define SIZE_OF_CL sizeof(uchar3) * n
#define SIZE_OF_CLUINT sizeof(Uint3) * n
#define SIZE_OF_INFO sizeof(info)

__global__ void Kmean(uchar4 * pic) {
    for (int y = blockDim.y * blockIdx.y + threadIdx.y; y < inf[0].h; y += blockDim.y * gridDim.y) {
        for (int x = blockDim.x * blockIdx.x + threadIdx.x; x < inf[0].w; x += blockDim.x * gridDim.x) {
            int resClas = -1;
            float maxDist = sqrt((float)3*(255*255))+1.0F;
			uchar4 piv = pic[x + inf[0].w * y];
            for (int i = 0; i < inf[0].n; i++) {
                float pivDist = sqrt( 					   						      \
					(((float)piv.x-(float)cl[i].x) * ((float)piv.x-(float)cl[i].x)) + \
					(((float)piv.y-(float)cl[i].y) * ((float)piv.y-(float)cl[i].y)) + \
					(((float)piv.z-(float)cl[i].z) * ((float)piv.z-(float)cl[i].z))   \
				);
                if (pivDist < maxDist) {
                    resClas = i;
                    maxDist = pivDist;
                }
            }
            pic[x + y * inf[0].w].w = (unsigned char)resClas;
        }
    }
}

// __global__ void reBuild(uchar4 * pic, Uint3 * newcl) {
// 	int idx = threadIdx.x;
// 	if (idx < inf[0].n) {
// 		unsigned char uidx = (unsigned char)idx;
// 		int num = 0;
// 		newcl[uidx].x = 0; newcl[uidx].y = 0; newcl[uidx].z = 0;
// 	    for (int y = 0; y < inf[0].h; y++) {
// 	        for (int x = 0; x < inf[0].w; x++) {
// 				if (pic[x + inf[0].w * y].w == uidx) {
// 					newcl[uidx].x += (int)pic[x + inf[0].w * y].x;
// 					newcl[uidx].y += (int)pic[x + inf[0].w * y].y;
// 					newcl[uidx].z += (int)pic[x + inf[0].w * y].z;
// 					num += 1;
// 				}
// 	        }
// 	    }
// 		// # if __CUDA_ARCH__>=200
// 	    // 	printf("%d: %d\n", idx, num);
// 		// #endif 
// 		if (num != 0) {
// 			newcl[uidx].x /= num;
// 			newcl[uidx].y /= num;
// 			newcl[uidx].z /= num;
// 		}
// 	}
// }

__global__ void reBuild(uchar4 * pic, Uint3 * newcl, int * num) {
	for (int y = blockDim.y * blockIdx.y + threadIdx.y; y < inf[0].h; y += blockDim.y * gridDim.y) {
        for (int x = blockDim.x * blockIdx.x + threadIdx.x; x < inf[0].w; x += blockDim.x * gridDim.x) {
			unsigned char ixd = pic[x + inf[0].w * y].w;
			newcl[ixd].x += (int)pic[x + inf[0].w * y].x;
			newcl[ixd].y += (int)pic[x + inf[0].w * y].y;
			newcl[ixd].z += (int)pic[x + inf[0].w * y].z;
			num[ixd] += 1;
        }
    }
}

__global__ void makeZ(Uint3 * newcl, int * num) {
	int uidx = threadIdx.x;
	if (uidx < inf[0].n) {
		newcl[uidx].x = 0; newcl[uidx].y = 0; newcl[uidx].z = 0;
		num[uidx] = 0;
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
	
	uchar4 * dev_pic;															// make dev struct for kernel
	CSC(cudaMalloc(&dev_pic, SIZE_OF_PIC));										// cuda malloc dev_pic
	CSC(cudaMemcpy(dev_pic, data, SIZE_OF_PIC, cudaMemcpyHostToDevice));
	
	info infHost[1];															// infHost
	infHost[0].w = w; infHost[0].h = h; infHost[0].n = n;						// copy info data and never touch again
	CSC(cudaMemcpyToSymbol(inf, infHost, SIZE_OF_INFO, 0, cudaMemcpyHostToDevice));
	
	uchar3 clHost[32];															// read data for classes
    for (int i = 0; i < n; i++) {
		cin >> x >> y;
		clHost[i].x = data[x + w * y].x;
		clHost[i].y = data[x + w * y].y;
		clHost[i].z = data[x + w * y].z;
	} 																			// we ll touch it every cicle
	CSC(cudaMemcpyToSymbol(cl, clHost, SIZE_OF_CL, 0, cudaMemcpyHostToDevice));
	
	Uint3 * dev_clnew;
	CSC(cudaMalloc(&dev_clnew, SIZE_OF_CLUINT));								// cuda malloc dev_clnew
	
	Uint3 * clnew; 																// but why this way?
	clnew = (Uint3 *)malloc(SIZE_OF_CLUINT);									// malloc clnew
	
	// 
	int * numHost = (int *)malloc(sizeof(int) * 32);							// test
	// for (int i = 0; i < 32; i++)	numHost[i] = 0;
	int * dev_num;
	CSC(cudaMalloc(&dev_num, sizeof(int) * 32));								// test
	CSC(cudaMemcpy(dev_num, numHost, sizeof(int) * 32, cudaMemcpyHostToDevice));
	// 
	
	int count = 0;
	cudaEvent_t start, stop;													// create event for timing
	float elapsedTime;
	cudaEventCreate(&start);													// time event to start
	cudaEventCreate(&stop);
	while (flag) {
		cudaEventRecord(start, 0);												// Start record
		Kmean <<<dim3(16, 16), dim3(32, 32)>>> (dev_pic);						// find new clasters
		cudaEventRecord(stop, 0);												// Stop event
		cudaEventSynchronize(stop);
		
		cudaEventElapsedTime(&elapsedTime, start, stop); 						// thats our time
		
		// fprintf(stderr, "%d,%f,", count, elapsedTime);
		
		makeZ <<<1,32>>> (dev_clnew, dev_num);									// make clnew and num are zero
		
		cudaEventRecord(start, 0);												// Start record	
		// reBuild <<<1, 32>>> (dev_pic, dev_clnew);							// save in clnew new centers by rgb // SLOW??!!!
		// reBuild <<<dim3(16, 16), dim3(32, 32)>>> (dev_pic, dev_clnew, dev_num);
		reBuild <<<dim3(1, 1), dim3(1, 32)>>> (dev_pic, dev_clnew, dev_num);
		cudaEventRecord(stop, 0);												// Stop event
		cudaEventSynchronize(stop);
		
		cudaEventElapsedTime(&elapsedTime, start, stop); 						// thats our time
		// fprintf(stderr, "%f\n", elapsedTime);


		CSC(cudaMemcpy(clnew, dev_clnew, SIZE_OF_CLUINT, cudaMemcpyDeviceToHost));
		
		
		
		// 
		CSC(cudaMemcpy(numHost, dev_num, sizeof(int) * 32, cudaMemcpyDeviceToHost));
		
		// cout << "before " <<  clnew[0].x << " ";
		for (int i = 0; i < n; i++) {
			if (numHost[i] != 0) {
				clnew[i].x /= numHost[i];
				clnew[i].y /= numHost[i];
				clnew[i].z /= numHost[i];
			}
		}
		// cout << "after " <<  clnew[0].x << endl;
		// 
	
		// cout << "\ncount: " << count << endl; cout << "clHost:" << endl;
		// for (int i = 0; i < n; i++) cout << (int)clHost[i].x << " " << (int)clHost[i].y << " " << (int)clHost[i].z << endl;
		// cout << "clnew:" << endl;
		// for (int i = 0; i < n; i++) cout << clnew[i].x << " " << clnew[i].y << " " << clnew[i].z << endl;
		// 
		
		
		flag = 0;
		for (int i = 0; i < n; i++) {
			if (!(clHost[i].x == (unsigned char)clnew[i].x && \
				  clHost[i].y == (unsigned char)clnew[i].y && \
				  clHost[i].z == (unsigned char)clnew[i].z)) {
					  flag = 1;
			}
		}
		// if (memcmp(clnew, clHost, sizeof(uchar3) * n) == 0)
		// 	flag = 0;
		for (int i = 0; i < n; i++) {
			clHost[i].x = (unsigned char)clnew[i].x;
			clHost[i].y = (unsigned char)clnew[i].y;
			clHost[i].z = (unsigned char)clnew[i].z;
		}
	
		CSC(cudaMemcpyToSymbol(cl, clHost, SIZE_OF_CL, 0, cudaMemcpyHostToDevice));
		
		//
		// for (int i = 0; i < 32; i++)	numHost[i] = 0;
		// CSC(cudaMemcpy(dev_num, numHost, sizeof(int) * 32, cudaMemcpyHostToDevice));
		//
		
		count++;
		// break;
	}
	
	cudaEventDestroy(start);													// time event is destroyed
	cudaEventDestroy(stop);														// clean up
	
	CSC(cudaMemcpy(data, dev_pic, SIZE_OF_PIC, cudaMemcpyDeviceToHost));
	
	CSC(cudaFree(dev_pic));														// cuda free dev_pic
	CSC(cudaFree(dev_clnew));													// cuda free dev_clnew

	
	f = fopen(filename2.c_str(), "wb");
	fwrite(&w, sizeof(int), 1, f);
	fwrite(&h, sizeof(int), 1, f);
	fwrite(data, sizeof(uchar4), w * h, f);
	fclose(f);

	free(clnew);																// free clnew
	free(data);																	// free data
	
	// 
	CSC(cudaFree(dev_num));
	free(numHost);
	// 
	return 0;
}
