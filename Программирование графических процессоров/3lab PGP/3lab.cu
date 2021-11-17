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

typedef struct { double x, y, z; } LL3; 									    // +4 294 967 295 ~ 255 * 4100 * 4100
typedef struct { int w, h, n; } info;											// pic 4100 by 4100 of one claster max pixel
																				// mb need to be changed to higher
typedef struct { double x, y, z; } D3;
__constant__ D3 CLASSES[32];													// contant classes no more than 32 by default
__constant__ info inf[1];														// some else param that never changes

#define SIZE_OF_PIC sizeof(uchar4) * w * h
#define SIZE_OF_CL sizeof(D3) * n
#define SIZE_OF_CLUINT sizeof(LL3) * n
#define SIZE_OF_INFO sizeof(info)

// texture<uchar4, 2, cudaReadModeElementType> tex;

__global__ void Kmean(uchar4 * pic) {
    for (int y = blockDim.y * blockIdx.y + threadIdx.y; y < inf[0].h; y += blockDim.y * gridDim.y) {
        for (int x = blockDim.x * blockIdx.x + threadIdx.x; x < inf[0].w; x += blockDim.x * gridDim.x) {
            double maxDist = sqrt((double)3*(255*255))+(double)1;
			// uchar4 piv = tex2D(tex, x, y);
			uchar4 piv = pic[x + inf[0].w * y];
            for (int i = 0; i < inf[0].n; i++) {
                double pivDist = sqrt( 					   						      \
					(((double)piv.x-CLASSES[i].x) * ((double)piv.x-CLASSES[i].x)) + \
					(((double)piv.y-CLASSES[i].y) * ((double)piv.y-CLASSES[i].y)) + \
					(((double)piv.z-CLASSES[i].z) * ((double)piv.z-CLASSES[i].z))   \
				);
                if (pivDist < maxDist) {
					pic[x + y * inf[0].w].w = (unsigned char)i;
                    maxDist = pivDist;
                }
            }
        }
    }
}

int main() {
	double eps = 0.1;
    string filename1, filename2;
    int w, h, n, x, y, flag = 1;
    cin >> filename1 >> filename2 >> n;
	
	FILE * f = fopen(filename1.c_str(), "rb");									// read data section
	fread(&w, sizeof(int), 1, f);
	fread(&h, sizeof(int), 1, f);
	uchar4 * data = (uchar4 *)malloc(SIZE_OF_PIC); 								// malloc data
	fread(data, sizeof(uchar4), w * h, f);
	fclose(f);
	
	uchar4 * DEV_DATA;															// make dev struct for kernel
	CSC(cudaMalloc(&DEV_DATA, SIZE_OF_PIC));									// cuda malloc dev_pic
	CSC(cudaMemcpy(DEV_DATA, data, SIZE_OF_PIC, cudaMemcpyHostToDevice));
	
	info infHost[1];															// infHost
	infHost[0].w = w; infHost[0].h = h; infHost[0].n = n;						// copy info data and never touch again
	CSC(cudaMemcpyToSymbol(inf, infHost, SIZE_OF_INFO, 0, cudaMemcpyHostToDevice));
	
	D3 clHost[32];															// read data for classes
    for (int i = 0; i < n; i++) {
		cin >> x >> y;
		clHost[i].x = (double)data[x + w * y].x;
		clHost[i].y = (double)data[x + w * y].y;
		clHost[i].z = (double)data[x + w * y].z;
	} 																			// we ll touch it every cicle
	
	LL3 * CLASSES_NEW = 	(LL3 *)malloc(SIZE_OF_CLUINT);						// malloc CLASSES_NEW
	int * CLASSES_NEW_NUM = (int *)malloc(sizeof(int) * n);						// test
	
	// // Подготовка данных для текстуры
	// cudaArray *arr;
	// cudaChannelFormatDesc ch = cudaCreateChannelDesc<uchar4>();
	// CSC(cudaMallocArray(&arr, &ch, w, h));
	// CSC(cudaMemcpyToArray(arr, 0, 0, data, sizeof(uchar4) * w * h, cudaMemcpyHostToDevice));
	// // Подготовка текстурной ссылки, настройка интерфейса работы с данными
	// tex.addressMode[0] = cudaAddressModeClamp;	// Политика обработки выхода за границы по каждому измерению
	// tex.addressMode[1] = cudaAddressModeClamp;
	// tex.channelDesc = ch;
	// tex.filterMode = cudaFilterModePoint;		// Без интерполяции при обращении по дробным координатам
	// tex.normalized = false;						// Режим нормализации координат: без нормализации
	// 
	// // Связываем интерфейс с данными
	// CSC(cudaBindTextureToArray(tex, arr, ch));
	// 
	while (flag) {
		CSC(cudaMemcpyToSymbol(CLASSES, clHost, SIZE_OF_CL, 0, cudaMemcpyHostToDevice));
		Kmean <<<dim3(16, 16), dim3(32, 32)>>> (DEV_DATA);						// find new clasters
		CSC(cudaMemcpy(data, DEV_DATA, SIZE_OF_PIC, cudaMemcpyDeviceToHost));
		
		for (int i = 0; i < n; i++) {
			CLASSES_NEW_NUM[i] = 0;
			CLASSES_NEW[i].x = 0;
			CLASSES_NEW[i].y = 0;
			CLASSES_NEW[i].z = 0;
		}
		
		for (int y = 0; y < h; y++) {
			for (int x = 0; x < w; x++) {
				int idx = data[y * w + x].w;
				CLASSES_NEW[idx].x += data[y * w + x].x;
				CLASSES_NEW[idx].y += data[y * w + x].y;
				CLASSES_NEW[idx].z += data[y * w + x].z;
				CLASSES_NEW_NUM[idx] += 1;
			}
		}
	
		for (int i = 0; i < n; i++) {
			if (CLASSES_NEW_NUM[i] != 0) {
				CLASSES_NEW[i].x /= CLASSES_NEW_NUM[i];
				CLASSES_NEW[i].y /= CLASSES_NEW_NUM[i];
				CLASSES_NEW[i].z /= CLASSES_NEW_NUM[i];
			}
		}
		
		flag = 0;
		for (int i = 0; i < n; i++) {
			if (!(											\
				(abs(clHost[i].x - CLASSES_NEW[i].x) < eps) && \
				(abs(clHost[i].y - CLASSES_NEW[i].y) < eps) && \
				(abs(clHost[i].z - CLASSES_NEW[i].z) < eps)	\
				)) {
					flag = 1;
			}
		}
		
		for (int i = 0; i < n; i++) {
			clHost[i].x = CLASSES_NEW[i].x;
			clHost[i].y = CLASSES_NEW[i].y;
			clHost[i].z = CLASSES_NEW[i].z;
		}
	}
	// 
	// // Отвязываем данные от текстурной ссылки
	// CSC(cudaUnbindTexture(tex));
	// CSC(cudaFreeArray(arr));
	// 
	CSC(cudaFree(DEV_DATA));														// cuda free dev_pic

	f = fopen(filename2.c_str(), "wb");
	fwrite(&w, sizeof(int), 1, f);
	fwrite(&h, sizeof(int), 1, f);
	fwrite(data, sizeof(uchar4), w * h, f);
	fclose(f);

	free(data);																	// free data
	free(CLASSES_NEW_NUM);
	free(CLASSES_NEW);															// free CLASSES_NEW
	return 0;
}
